 Charts
 Statistics
 Code


from AlgorithmImports import *
import numpy as np
from collections import deque

class RegimeRiskBudgetAllocator(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2016, 1, 1)
        self.SetEndDate(2026, 1, 1)
        self.SetCash(5000)

        # Assets
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.tlt = self.AddEquity("TLT", Resolution.Daily).Symbol
        self.gld = self.AddEquity("GLD", Resolution.Daily).Symbol

        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)

        # Leverage (IG proxy: 1:20)
        self.leverage = 20.0
        for sym in [self.spy, self.tlt, self.gld]:
            self.Securities[sym].SetLeverage(self.leverage)

        # HARD regime ceilings (no drawdown governor)
        self.margin_budget_by_regime = {
            "Calm":   0.12,
            "Alert":  0.09,
            "Stress": 0.06,
            "Panic":  0.04
        }

        # Recovery participation floor (unchanged)
        self.calm_min_effective_margin_pct = 0.07

        # Structural weights
        self.weights = {
            "Calm":   {"spy": 0.55, "tlt": 0.35, "gld": 0.10},
            "Alert":  {"spy": 0.45, "tlt": 0.35, "gld": 0.10},
            "Stress": {"spy": 0.30, "tlt": 0.35, "gld": 0.10},
            "Panic":  {"spy": 0.20, "tlt": 0.30, "gld": 0.10}
        }

        # Indicators
        self.sma50  = self.SMA(self.spy, 50, Resolution.Daily)
        self.sma200 = self.SMA(self.spy, 200, Resolution.Daily)
        self.rsi    = self.RSI(self.spy, 14, MovingAverageType.Wilders, Resolution.Daily)

        self.close_window = deque(maxlen=252)
        self.vol_window   = deque(maxlen=252)
        self.days_below_sma50 = 0

        # Regime state
        self.regime = "Calm"
        self.regime_codes = {"Calm": 0, "Alert": 1, "Stress": 2, "Panic": 3}

        # Rebalance control
        self.last_regime = self.regime
        self.last_effective_pct = None
        self.last_spy_weight_multiplier = None

        self.SetWarmup(250, Resolution.Daily)
        self.SeedCloseAndVolWindows()

        self.Schedule.On(
            self.DateRules.EveryDay(self.spy),
            self.TimeRules.BeforeMarketClose(self.spy, 10),
            self.UpdateSignals
        )

    def SeedCloseAndVolWindows(self):
        hist = self.History(self.spy, 252, Resolution.Daily)
        if hist.empty:
            return

        try:
            closes = hist.loc[self.spy]["close"].values.astype(float)
        except Exception:
            closes = hist["close"].values.astype(float)

        closes = closes[-252:]
        for c in closes:
            if c > 0:
                self.close_window.append(float(c))

        if len(self.close_window) >= 21:
            prices = np.array(list(self.close_window), dtype=float)
            rets = np.diff(np.log(prices))
            for i in range(20, len(rets) + 1):
                window = rets[i-20:i]
                rv20 = float(np.std(window, ddof=1) * np.sqrt(252) * 100)
                self.vol_window.append(rv20)

    def UpdateSignals(self):
        close = float(self.Securities[self.spy].Close)
        if close > 0:
            self.close_window.append(close)

        if not (self.sma50.IsReady and self.sma200.IsReady and self.rsi.IsReady):
            return

        self.days_below_sma50 = self.days_below_sma50 + 1 if close < self.sma50.Current.Value else 0

        rv20 = self.RealizedVol20()
        rsi  = float(self.rsi.Current.Value)
        dd20 = self.ReturnNDays(20)

        if rv20 is None or dd20 is None:
            return

        self.vol_window.append(float(rv20))

        median_vol = float(np.median(self.vol_window)) if len(self.vol_window) >= 100 else float(rv20)
        if median_vol <= 0:
            median_vol = float(rv20)

        vol_ratio = float(rv20 / median_vol)

        new_regime = self.regime

        if vol_ratio > 1.60 or dd20 <= -0.08:
            new_regime = "Panic"
        elif vol_ratio >= 1.40 and self.days_below_sma50 >= 4 and rsi < 40:
            new_regime = "Stress"
        elif vol_ratio >= 1.25 and self.days_below_sma50 >= 2 and rsi < 45:
            new_regime = "Alert"
        elif vol_ratio < 1.15 and close >= self.sma50.Current.Value and rsi > 50:
            new_regime = "Calm"

        prev_regime = self.regime
        self.regime = self.ClampTier(self.regime, new_regime)

        self.Plot("Regime", "Stage", self.regime_codes[self.regime])
        self.Plot("Regime", "Vol20", rv20)
        self.Plot("Regime", "VolRatio", vol_ratio)

        if self.regime != prev_regime:
            self.Rebalance()

    def Rebalance(self):
        equity = float(self.Portfolio.TotalPortfolioValue)
        if equity <= 0:
            return

        base_pct = float(self.margin_budget_by_regime[self.regime])
        effective_pct = base_pct

        if self.regime == "Calm":
            effective_pct = max(effective_pct, self.calm_min_effective_margin_pct)
            effective_pct = min(effective_pct, base_pct)

        close = float(self.Securities[self.spy].Close)
        sma200 = float(self.sma200.Current.Value) if self.sma200.IsReady else close
        extended_calm = (self.regime == "Calm" and sma200 > 0 and close > 1.05 * sma200)

        spy_weight_multiplier = 0.85 if extended_calm else 1.0

        effective_changed = (
            self.last_effective_pct is None or
            abs(effective_pct - self.last_effective_pct) >= 0.01
        )
        regime_changed = (self.regime != self.last_regime)
        spy_mult_changed = (
            self.last_spy_weight_multiplier is None or
            abs(spy_weight_multiplier - self.last_spy_weight_multiplier) > 1e-9
        )

        if not (regime_changed or effective_changed or spy_mult_changed):
            return

        margin_budget = equity * effective_pct

        self.Plot("Risk", "EffectiveMarginPct", effective_pct * 100)
        self.Plot("Risk", "ExtendedCalm", 1 if extended_calm else 0)

        w = self.weights[self.regime].copy()
        w["spy"] *= spy_weight_multiplier

        targets_value = {
            self.spy: w["spy"] * margin_budget * self.leverage,
            self.tlt: w["tlt"] * margin_budget * self.leverage,
            self.gld: w["gld"] * margin_budget * self.leverage
        }

        targets = [PortfolioTarget(sym, val / equity) for sym, val in targets_value.items()]
        self.SetHoldings(targets, True)

        self.last_regime = self.regime
        self.last_effective_pct = effective_pct
        self.last_spy_weight_multiplier = spy_weight_multiplier

    def RealizedVol20(self):
        if len(self.close_window) < 21:
            return None
        prices = np.array(list(self.close_window)[-21:], dtype=float)
        rets = np.diff(np.log(prices))
        return float(np.std(rets[-20:], ddof=1) * np.sqrt(252) * 100)

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
