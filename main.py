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

        self.SetBrokerageModel(
            BrokerageName.InteractiveBrokersBrokerage,
            AccountType.Margin
        )

        # Risk budget
        self.margin_budget_pct = 0.12

        self.assumed_leverage = {
            self.spy: 5.0,
            self.tlt: 5.0,
            self.gld: 5.0
        }

        for sym, lev in self.assumed_leverage.items():
            self.Securities[sym].SetLeverage(lev)

        # Indicators
        self.sma50  = self.SMA(self.spy, 50, Resolution.Daily)
        self.sma200 = self.SMA(self.spy, 200, Resolution.Daily)
        self.rsi    = self.RSI(self.spy, 14, MovingAverageType.Wilders, Resolution.Daily)

        self.close_window = deque(maxlen=252)
        self.days_below_sma50 = 0

        # Regime state
        self.regime = "Calm"
        self.prolonged_bear = False

        self.bear_dd_threshold = -0.12
        self.bear_lookback = 126

        self.regime_codes = {
            "Calm": 0,
            "Alert": 1,
            "Stress": 2,
            "Panic": 3,
            "Bear": 4
        }
        self.regime_numeric = 0

        # Base weights
        self.tiers = {
            "Calm":   {"spy": 0.55, "tlt": 0.35, "gld": 0.10},
            "Alert":  {"spy": 0.50, "tlt": 0.40, "gld": 0.10},
            "Stress": {"spy": 0.40, "tlt": 0.45, "gld": 0.15},
            "Panic":  {"spy": 0.35, "tlt": 0.45, "gld": 0.20}
        }

        self.last_rebalance_week = -1

        # DCA
        self.monthly_contribution = 200
        self.total_contributions = 0

        self.Schedule.On(
            self.DateRules.MonthStart(self.spy),
            self.TimeRules.AfterMarketOpen(self.spy, 5),
            self.AddMonthlyContribution
        )

        self.Schedule.On(
            self.DateRules.EveryDay(self.spy),
            self.TimeRules.AfterMarketClose(self.spy),
            self.UpdateSignals
        )

        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Friday),
            self.TimeRules.BeforeMarketClose(self.spy, 5),
            self.Rebalance
        )

        self.SetWarmup(250, Resolution.Daily)

    def AddMonthlyContribution(self):
        self.Portfolio.CashBook["USD"].AddAmount(self.monthly_contribution)
        self.total_contributions += self.monthly_contribution
        self.Plot("Contributions", "Total", float(self.total_contributions))

    def UpdateSignals(self):
        if self.IsWarmingUp:
            return

        close = self.Securities[self.spy].Close
        self.close_window.append(close)

        if self.sma50.IsReady:
            self.days_below_sma50 = self.days_below_sma50 + 1 if close < self.sma50.Current.Value else 0

        rv20 = self.RealizedVol20()
        rsi  = self.rsi.Current.Value if self.rsi.IsReady else None
        dd20 = self.ReturnNDays(20)

        if rv20 is None or rsi is None or not self.sma200.IsReady:
            return

        new_regime = self.regime

        if rv20 > 24 or (dd20 is not None and dd20 <= -0.08):
            new_regime = "Panic"
        elif rv20 >= 20 and self.days_below_sma50 >= 4 and rsi < 40:
            new_regime = "Stress"
        elif rv20 >= 18 and self.days_below_sma50 >= 2 and rsi < 45:
            new_regime = "Alert"
        elif rv20 < 18 and close >= self.sma50.Current.Value and rsi > 50:
            new_regime = "Calm"

        self.regime = self.ClampTier(self.regime, new_regime)

        dd_6m = self.DrawdownFromHigh(self.bear_lookback)
        below_200 = close < self.sma200.Current.Value

        self.prolonged_bear = (
            below_200 and dd_6m is not None and dd_6m <= self.bear_dd_threshold
        )

        if self.prolonged_bear and self.regime in ["Stress", "Panic"]:
            self.regime_numeric = self.regime_codes["Bear"]
        else:
            self.regime_numeric = self.regime_codes[self.regime]

        self.Plot("Regime", "Stage", self.regime_numeric)

    def Rebalance(self):
        if self.IsWarmingUp:
            return

        week = self.Time.isocalendar()[1]
        if week == self.last_rebalance_week:
            return

        weights = dict(self.tiers[self.regime])

        # ✅ Increased short sizing (ONLY CHANGE)
        if self.prolonged_bear and self.regime in ["Stress", "Panic"]:
            if self.regime == "Stress":
                weights["spy"] = -0.25
                weights["tlt"] = 0.60
                weights["gld"] = 0.20
            else:
                weights["spy"] = -0.40
                weights["tlt"] = 0.60
                weights["gld"] = 0.25

        equity = self.Portfolio.TotalPortfolioValue
        margin_budget = equity * self.margin_budget_pct

        targets_value = {}
        for sym, key in [(self.spy, "spy"), (self.tlt, "tlt"), (self.gld, "gld")]:
            lev = self.assumed_leverage[sym]
            alloc_margin = abs(weights[key]) * margin_budget
            value = alloc_margin * lev
            targets_value[sym] = value if weights[key] >= 0 else -value

        total_margin = sum(
            abs(v) / self.assumed_leverage[sym]
            for sym, v in targets_value.items()
        )

        if total_margin > margin_budget:
            scale = margin_budget / total_margin
            for sym in targets_value:
                targets_value[sym] *= scale

        targets = [
            PortfolioTarget(sym, value / equity)
            for sym, value in targets_value.items()
        ]

        self.SetHoldings(targets, True)
        self.last_rebalance_week = week

    # Helpers
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
        return closes[-1] / closes[-(n + 1)] - 1

    def DrawdownFromHigh(self, lookback):
        if len(self.close_window) < lookback:
            return None
        closes = list(self.close_window)[-lookback:]
        peak = max(closes)
        return closes[-1] / peak - 1

    def ClampTier(self, current, proposed):
        order = ["Calm", "Alert", "Stress", "Panic"]
        ci = order.index(current)
        pi = order.index(proposed)
        if pi > ci + 1:
            return order[ci + 1]
        if pi < ci - 1:
            return order[ci - 1]
        return proposed
