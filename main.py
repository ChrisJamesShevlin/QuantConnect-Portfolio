from AlgorithmImports import *
import numpy as np
from collections import deque

class RegimeRiskBudgetAllocator(QCAlgorithm):

    def Initialize(self):
        # Backtest window
        self.SetStartDate(2016, 1, 1)
        self.SetEndDate(2026, 1, 1)
        self.SetCash(5000)

        # -----------------------------
        # Assets
        # -----------------------------
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.tlt = self.AddEquity("TLT", Resolution.Daily).Symbol
        self.gld = self.AddEquity("GLD", Resolution.Daily).Symbol

        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)

        # -----------------------------
        # Leverage (IG proxy: 1:20)
        # -----------------------------
        self.leverage = 20.0
        for sym in [self.spy, self.tlt, self.gld]:
            self.Securities[sym].SetLeverage(self.leverage)

        # -----------------------------
        # Base margin deployment by regime
        # (Risk-off = deploy less, allow cash)
        # -----------------------------
        self.margin_budget_by_regime = {
            "Calm":   0.12,
            "Alert":  0.09,
            "Stress": 0.06,
            "Panic":  0.04
        }

        # -----------------------------
        # Recovery participation floor
        # -----------------------------
        self.calm_min_effective_margin_pct = 0.07

        # -----------------------------
        # Structural weights by regime
        # NOTE: As requested, bonds/gold remain the same; SPY reduces and the remainder becomes cash.
        # -----------------------------
        self.weights = {
            "Calm":   {"spy": 0.55, "tlt": 0.35, "gld": 0.10},
            "Alert":  {"spy": 0.45, "tlt": 0.35, "gld": 0.10},
            "Stress": {"spy": 0.30, "tlt": 0.35, "gld": 0.10},
            "Panic":  {"spy": 0.20, "tlt": 0.30, "gld": 0.10}
        }

        # -----------------------------
        # Drawdown governor (tweaked: earlier + smoother)
        # -----------------------------
        self.dd_buffer = 0.12        # start scaling earlier
        self.dd_target = 0.25        # aim to be meaningfully de-risked by ~25% DD
        self.min_risk_scale = 0.20   # avoid crushing to near-zero (helps recovery)

        self.peak_equity = None
        self.current_dd = 0.0

        # -----------------------------
        # Indicators (SPY drives regime)
        # -----------------------------
        self.sma50  = self.SMA(self.spy, 50, Resolution.Daily)
        self.sma200 = self.SMA(self.spy, 200, Resolution.Daily)
        self.rsi    = self.RSI(self.spy, 14, MovingAverageType.Wilders, Resolution.Daily)

        self.close_window = deque(maxlen=252)
        self.vol_window   = deque(maxlen=252)  # for volatility normalization
        self.days_below_sma50 = 0

        # Regime state
        self.regime = "Calm"
        self.regime_codes = {"Calm": 0, "Alert": 1, "Stress": 2, "Panic": 3}

        # Rebalance control
        self.last_rebalance_week = -1
        self.last_regime = self.regime
        self.last_effective_pct = None
        self.last_spy_weight_multiplier = None

        # Warmup + seed history
        self.SetWarmup(250, Resolution.Daily)
        self.SeedCloseAndVolWindows()

        # Scheduling
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

    # -----------------------------
    # Seed close + vol windows
    # -----------------------------
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

        # Seed vol_window by computing rolling 20d realized vol across the seeded closes
        if len(self.close_window) >= 21:
            prices = np.array(list(self.close_window), dtype=float)
            logp = np.log(prices)
            rets = np.diff(logp)  # daily log returns

            # rolling 20 std of returns (requires 20 returns -> 21 prices)
            # fill with the most recent windowed vols up to maxlen
            for i in range(20, len(rets) + 1):
                window = rets[i-20:i]
                if len(window) == 20:
                    rv20 = float(np.std(window, ddof=1) * np.sqrt(252) * 100)
                    self.vol_window.append(rv20)

    # -----------------------------
    # Update signals + drawdown tracking
    # -----------------------------
    def UpdateSignals(self):
        # Maintain close window
        close = float(self.Securities[self.spy].Close)
        if close > 0:
            self.close_window.append(close)

        # Track peak equity / drawdown (strategy equity)
        equity = float(self.Portfolio.TotalPortfolioValue)
        if self.peak_equity is None:
            self.peak_equity = equity

        if equity > self.peak_equity:
            self.peak_equity = equity

        self.current_dd = 0.0 if self.peak_equity <= 0 else 1.0 - (equity / self.peak_equity)
        self.Plot("Risk", "Drawdown", self.current_dd)

        # Regime classification needs indicators ready
        if not (self.sma50.IsReady and self.sma200.IsReady and self.rsi.IsReady):
            return

        # Persistence below 50-DMA
        self.days_below_sma50 = self.days_below_sma50 + 1 if close < self.sma50.Current.Value else 0

        rv20 = self.RealizedVol20()
        rsi  = float(self.rsi.Current.Value)
        dd20 = self.ReturnNDays(20)

        if rv20 is None or dd20 is None:
            return

        # Update vol normalization window
        self.vol_window.append(float(rv20))

        # Normalized volatility ratio vs trailing median (robust to changing vol regimes)
        median_vol = float(np.median(self.vol_window)) if len(self.vol_window) >= 100 else float(rv20)
        if median_vol <= 0:
            median_vol = float(rv20)

        vol_ratio = float(rv20 / median_vol)

        new_regime = self.regime

        # -----------------------------
        # Regime logic (tweaked: use vol_ratio; keep your dd20 crash trigger)
        # -----------------------------
        if vol_ratio > 1.60 or dd20 <= -0.08:
            new_regime = "Panic"
        elif vol_ratio >= 1.40 and self.days_below_sma50 >= 4 and rsi < 40:
            new_regime = "Stress"
        elif vol_ratio >= 1.25 and self.days_below_sma50 >= 2 and rsi < 45:
            new_regime = "Alert"
        elif vol_ratio < 1.15 and close >= self.sma50.Current.Value and rsi > 50:
            new_regime = "Calm"

        # Allow fast de-risking, slow re-risking
        self.regime = self.ClampTier(self.regime, new_regime)

        # Plots for intuition/debugging
        self.Plot("Regime", "Stage", self.regime_codes[self.regime])
        self.Plot("Regime", "Vol20", rv20)
        self.Plot("Regime", "VolRatio", vol_ratio)

    # -----------------------------
    # Drawdown governor: scale risk as DD rises (tweaked parameters set in Initialize)
    # -----------------------------
    def RiskScaleFromDrawdown(self) -> float:
        dd = float(self.current_dd)

        if dd <= self.dd_buffer:
            return 1.0

        if dd >= self.dd_target:
            return float(self.min_risk_scale)

        span = self.dd_target - self.dd_buffer
        if span <= 0:
            return float(self.min_risk_scale)

        t = (dd - self.dd_buffer) / span  # 0..1
        scale = 1.0 - t * (1.0 - self.min_risk_scale)
        return float(max(self.min_risk_scale, min(1.0, scale)))

    # -----------------------------
    # Rebalance (long-only), regime budgets + drawdown governor + Calm floor
    # Plus: (1) late-cycle Calm brake that reduces SPY only, (2) conditional rebalance gating
    # -----------------------------
    def Rebalance(self):
        # Refresh signals right before sizing
        self.UpdateSignals()

        week = self.Time.isocalendar()[1]
        if week == self.last_rebalance_week:
            return

        equity = float(self.Portfolio.TotalPortfolioValue)
        if equity <= 0:
            return

        base_pct = float(self.margin_budget_by_regime[self.regime])
        risk_scale = self.RiskScaleFromDrawdown()

        # Effective margin percentage after drawdown scaling
        effective_pct = base_pct * risk_scale

        # Recovery participation floor (Calm only)
        if self.regime == "Calm":
            effective_pct = max(effective_pct, float(self.calm_min_effective_margin_pct))
            effective_pct = min(effective_pct, base_pct)

        # Late-cycle Calm brake (reduce SPY only; bonds/gold unchanged; remainder becomes cash)
        # "Extended Calm" defined as SPY meaningfully above long trend.
        close = float(self.Securities[self.spy].Close)
        sma200 = float(self.sma200.Current.Value) if self.sma200.IsReady else close
        extended_calm = (self.regime == "Calm" and sma200 > 0 and close > 1.05 * sma200)

        spy_weight_multiplier = 0.85 if extended_calm else 1.0

        # Conditional rebalance gating:
        # Only rebalance if regime changed OR effective risk changed materially OR Calm-brake toggled.
        # (Still runs weekly schedule; this reduces churn.)
        effective_changed = (
            self.last_effective_pct is None or
            abs(effective_pct - float(self.last_effective_pct)) >= 0.01  # 1% of equity margin deployment
        )
        regime_changed = (self.regime != self.last_regime)
        spy_mult_changed = (
            self.last_spy_weight_multiplier is None or
            abs(spy_weight_multiplier - float(self.last_spy_weight_multiplier)) > 1e-9
        )

        if not (regime_changed or effective_changed or spy_mult_changed):
            self.last_rebalance_week = week
            return

        margin_budget = equity * effective_pct

        # Plots for debugging/intuition
        self.Plot("Risk", "RiskScale", risk_scale)
        self.Plot("Risk", "EffectiveMarginPct", effective_pct * 100)
        self.Plot("Risk", "ExtendedCalm", 1 if extended_calm else 0)

        w = self.weights[self.regime].copy()

        # Apply late-cycle brake to SPY only (bonds/gold weights unchanged; freed risk is cash)
        w["spy"] = float(w["spy"]) * float(spy_weight_multiplier)

        # Convert margin allocations to notional targets using leverage
        targets_value = {
            self.spy: float(w["spy"]) * margin_budget * self.leverage,
            self.tlt: float(w["tlt"]) * margin_budget * self.leverage,
            self.gld: float(w["gld"]) * margin_budget * self.leverage
        }

        targets = [PortfolioTarget(sym, val / equity) for sym, val in targets_value.items()]
        self.SetHoldings(targets, True)

        # Update state
        self.last_rebalance_week = week
        self.last_regime = self.regime
        self.last_effective_pct = effective_pct
        self.last_spy_weight_multiplier = spy_weight_multiplier

    # -----------------------------
    # Helpers
    # -----------------------------
    def RealizedVol20(self):
        if len(self.close_window) < 21:
            return None
        prices = np.array(list(self.close_window)[-21:], dtype=float)
        if np.any(prices <= 0):
            return None
        rets = np.diff(np.log(prices))
        if len(rets) < 20:
            return None
        return float(np.std(rets[-20:], ddof=1) * np.sqrt(252) * 100)

    def ReturnNDays(self, n):
        if len(self.close_window) < n + 1:
            return None
        closes = list(self.close_window)
        if closes[-(n + 1)] <= 0:
            return None
        return float(closes[-1] / closes[-(n + 1)] - 1)

    def ClampTier(self, current, proposed):
        """
        Tweaked behavior:
        - Allow fast de-risking (move to higher-risk tier immediately).
        - Slow re-risking (only step up one tier at a time on improvement).
        """
        order = ["Calm", "Alert", "Stress", "Panic"]
        ci = order.index(current)
        pi = order.index(proposed)

        # Fast de-risking
        if pi > ci:
            return proposed

        # Slow re-risking (no multi-tier jumps back to Calm)
        if pi < ci - 1:
            return order[ci - 1]

        return proposed
