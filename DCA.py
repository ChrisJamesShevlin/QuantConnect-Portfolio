from AlgorithmImports import *
import numpy as np
from collections import deque

class RegimeRiskBudgetAllocator(QCAlgorithm):

    def Initialize(self):
        # -------------------------------------------------
        # Backtest window
        # -------------------------------------------------
        self.SetStartDate(2016, 1, 1)
        self.SetEndDate(2026, 1, 1)
        self.SetCash(5000)

        # -------------------------------------------------
        # Assets
        # -------------------------------------------------
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.tlt = self.AddEquity("TLT", Resolution.Daily).Symbol
        self.gld = self.AddEquity("GLD", Resolution.Daily).Symbol

        self.SetBrokerageModel(
            BrokerageName.InteractiveBrokersBrokerage,
            AccountType.Margin
        )

        # -------------------------------------------------
        # Leverage (IG proxy)
        # -------------------------------------------------
        self.leverage = 20.0
        for sym in [self.spy, self.tlt, self.gld]:
            self.Securities[sym].SetLeverage(self.leverage)

        # -------------------------------------------------
        # Monthly DCA (cash only)
        # -------------------------------------------------
        self.dca_amount = 200

        self.Schedule.On(
            self.DateRules.MonthStart(self.spy),
            self.TimeRules.AfterMarketOpen(self.spy, 5),
            self.AddMonthlyDCA
        )

        # -------------------------------------------------
        # Margin budgets by regime
        # -------------------------------------------------
        self.margin_budget_by_regime = {
            "Calm":   0.12,
            "Alert":  0.09,
            "Stress": 0.06,
            "Panic":  0.04
        }

        # -------------------------------------------------
        # Calm recovery floor
        # -------------------------------------------------
        self.calm_min_effective_margin_pct = 0.07

        # -------------------------------------------------
        # Structural weights
        # -------------------------------------------------
        self.weights = {
            "Calm":   {"spy": 0.55, "tlt": 0.35, "gld": 0.10},
            "Alert":  {"spy": 0.45, "tlt": 0.35, "gld": 0.10},
            "Stress": {"spy": 0.30, "tlt": 0.35, "gld": 0.10},
            "Panic":  {"spy": 0.20, "tlt": 0.30, "gld": 0.10}
        }

        # -------------------------------------------------
        # Drawdown governor
        # -------------------------------------------------
        self.dd_buffer = 0.12
        self.dd_target = 0.25
        self.min_risk_scale = 0.20

        self.peak_equity = None
        self.current_dd = 0.0

        # -------------------------------------------------
        # Indicators
        # -------------------------------------------------
        self.sma50  = self.SMA(self.spy, 50, Resolution.Daily)
        self.sma200 = self.SMA(self.spy, 200, Resolution.Daily)
        self.rsi    = self.RSI(self.spy, 14, MovingAverageType.Wilders, Resolution.Daily)

        self.close_window = deque(maxlen=252)
        self.vol_window   = deque(maxlen=252)
        self.days_below_sma50 = 0

        # -------------------------------------------------
        # Regime state
        # -------------------------------------------------
        self.regime = "Calm"
        self.regime_codes = {"Calm": 0, "Alert": 1, "Stress": 2, "Panic": 3}

        self.last_rebalance_week = -1
        self.last_regime = self.regime
        self.last_effective_pct = None
        self.last_spy_weight_multiplier = None

        # -------------------------------------------------
        # Warmup
        # -------------------------------------------------
        self.SetWarmup(250, Resolution.Daily)
        self.SeedCloseAndVolWindows()

        # -------------------------------------------------
        # Scheduling
        # -------------------------------------------------
        self.Schedule.On(
            self.DateRules.EveryDay(self.spy),
            self.TimeRules.BeforeMarketClose(self.spy, 10),
            self.UpdateSignals
        )

        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Friday),
            self.TimeRules.BeforeMarketClose(self.spy, 5),
            self.Rebalance
        )

    # =================================================
    # Monthly DCA
    # =================================================
    def AddMonthlyDCA(self):
        self.Portfolio.CashBook["USD"].AddAmount(self.dca_amount)

    # =================================================
    # Seed history
    # =================================================
    def SeedCloseAndVolWindows(self):
        hist = self.History(self.spy, 252, Resolution.Daily)
        if hist.empty:
            return

        closes = hist.loc[self.spy]["close"].values.astype(float)

        for c in closes[-252:]:
            if c > 0:
                self.close_window.append(c)

        if len(self.close_window) >= 21:
            prices = np.array(list(self.close_window))
            rets = np.diff(np.log(prices))
            for i in range(20, len(rets) + 1):
                rv20 = np.std(rets[i-20:i], ddof=1) * np.sqrt(252) * 100
                self.vol_window.append(rv20)

    # =================================================
    # Update signals
    # =================================================
    def UpdateSignals(self):
        close = float(self.Securities[self.spy].Close)
        if close > 0:
            self.close_window.append(close)

        equity = float(self.Portfolio.TotalPortfolioValue)
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity

        self.current_dd = 1.0 - equity / self.peak_equity
        self.Plot("Risk", "Drawdown", self.current_dd)

        if not (self.sma50.IsReady and self.sma200.IsReady and self.rsi.IsReady):
            return

        self.days_below_sma50 = self.days_below_sma50 + 1 if close < self.sma50.Current.Value else 0

        rv20 = self.RealizedVol20()
        dd20 = self.ReturnNDays(20)
        rsi  = self.rsi.Current.Value

        if rv20 is None or dd20 is None:
            return

        self.vol_window.append(rv20)
        median_vol = np.median(self.vol_window)
        vol_ratio = rv20 / median_vol if median_vol > 0 else 1.0

        new_regime = self.regime

        if vol_ratio > 1.60 or dd20 <= -0.08:
            new_regime = "Panic"
        elif vol_ratio >= 1.40 and self.days_below_sma50 >= 4 and rsi < 40:
            new_regime = "Stress"
        elif vol_ratio >= 1.25 and self.days_below_sma50 >= 2 and rsi < 45:
            new_regime = "Alert"
        elif vol_ratio < 1.15 and close >= self.sma50.Current.Value and rsi > 50:
            new_regime = "Calm"

        self.regime = self.ClampTier(self.regime, new_regime)

        self.Plot("Regime", "Stage", self.regime_codes[self.regime])
        self.Plot("Regime", "Vol20", rv20)
        self.Plot("Regime", "VolRatio", vol_ratio)

    # =================================================
    # Drawdown scaling
    # =================================================
    def RiskScaleFromDrawdown(self):
        if self.current_dd <= self.dd_buffer:
            return 1.0
        if self.current_dd >= self.dd_target:
            return self.min_risk_scale

        t = (self.current_dd - self.dd_buffer) / (self.dd_target - self.dd_buffer)
        return max(self.min_risk_scale, 1.0 - t * (1.0 - self.min_risk_scale))

    # =================================================
    # Weekly rebalance
    # =================================================
    def Rebalance(self):
        self.UpdateSignals()

        week = self.Time.isocalendar()[1]
        if week == self.last_rebalance_week:
            return

        equity = self.Portfolio.TotalPortfolioValue
        base_pct = self.margin_budget_by_regime[self.regime]
        risk_scale = self.RiskScaleFromDrawdown()
        effective_pct = base_pct * risk_scale

        if self.regime == "Calm":
            effective_pct = max(effective_pct, self.calm_min_effective_margin_pct)
            effective_pct = min(effective_pct, base_pct)

        close = self.Securities[self.spy].Close
        sma200 = self.sma200.Current.Value
        extended_calm = self.regime == "Calm" and close > 1.05 * sma200
        spy_mult = 0.85 if extended_calm else 1.0

        margin_budget = equity * effective_pct
        w = self.weights[self.regime].copy()
        w["spy"] *= spy_mult

        targets = [
            PortfolioTarget(self.spy, w["spy"] * margin_budget * self.leverage / equity),
            PortfolioTarget(self.tlt, w["tlt"] * margin_budget * self.leverage / equity),
            PortfolioTarget(self.gld, w["gld"] * margin_budget * self.leverage / equity)
        ]

        self.SetHoldings(targets, True)
        self.last_rebalance_week = week

    # =================================================
    # Helpers (FIXED deque slicing)
    # =================================================
    def RealizedVol20(self):
        if len(self.close_window) < 21:
            return None
        prices = np.array(list(self.close_window)[-21:], dtype=float)
        rets = np.diff(np.log(prices))
        return float(np.std(rets, ddof=1) * np.sqrt(252) * 100)

    def ReturnNDays(self, n):
        if len(self.close_window) < n + 1:
            return None
        closes = list(self.close_window)
        return float(closes[-1] / closes[-(n + 1)] - 1)

    def ClampTier(self, current, proposed):
        order = ["Calm", "Alert", "Stress", "Panic"]
        ci = order.index(current)
        pi = order.index(proposed)

        if pi > ci:
            return proposed
        if pi < ci - 1:
            return order[ci - 1]
        return proposed
