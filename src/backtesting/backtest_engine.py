"""Backtest engine: replay 12 months of market data through the rebalancing agent."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from src.monitoring.drift_calculator import DriftCalculator
from src.optimisation.portfolio_optimiser import PortfolioOptimiser

TARGET_ALLOCATIONS = {
    "ultra_conservative": np.array([0.10, 0.05, 0.45, 0.15, 0.10, 0.15]),
    "conservative": np.array([0.20, 0.10, 0.35, 0.10, 0.12, 0.13]),
    "balanced": np.array([0.35, 0.15, 0.20, 0.10, 0.12, 0.08]),
    "aggressive": np.array([0.50, 0.20, 0.10, 0.05, 0.10, 0.05]),
    "ultra_aggressive": np.array([0.60, 0.25, 0.05, 0.00, 0.07, 0.03]),
}

DRIFT_BANDS = {
    "ultra_conservative": 0.020,
    "conservative": 0.025,
    "balanced": 0.030,
    "aggressive": 0.040,
    "ultra_aggressive": 0.050,
}


@dataclass
class DailyPortfolioState:
    date: date
    weights: np.ndarray
    value_inr: float
    rebalanced_today: bool
    cost_inr: float = 0.0
    trigger_type: str | None = None


@dataclass
class BacktestResult:
    strategy_name: str
    portfolio_id: str
    risk_category: str
    initial_value_inr: float
    final_value_inr: float
    daily_states: list[DailyPortfolioState]

    # Performance metrics
    annualised_return: float = 0.0
    annualised_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    tracking_error: float = 0.0
    turnover: float = 0.0
    total_cost_inr: float = 0.0
    rebalance_count: int = 0


class BacktestEngine:
    """
    Simulate portfolio evolution over historical market data for one portfolio.
    Supports agent, legacy (quarterly), threshold-only, and buy-and-hold strategies.
    """

    TRANSACTION_COST_PCT = 0.0015  # 15 bps round-trip

    def __init__(
        self,
        market_returns: pd.DataFrame,
        risk_free_rate_annual: float = 0.065,
    ):
        self.market_returns = market_returns
        self.rf_daily = (1 + risk_free_rate_annual) ** (1 / 252) - 1
        self.calculator = DriftCalculator()
        self.optimiser = PortfolioOptimiser()

    def run(
        self,
        portfolio_id: str,
        risk_category: str,
        initial_weights: np.ndarray,
        initial_value_inr: float,
        strategy: str = "agent",
        quarterly_months: set | None = None,
    ) -> BacktestResult:
        """
        Replay the full backtest for one portfolio.

        strategy: "agent" | "legacy_quarterly" | "threshold_only" | "buy_and_hold"
        """
        target = TARGET_ALLOCATIONS[risk_category]
        band = DRIFT_BANDS[risk_category]
        dates = self.market_returns.index.tolist()

        weights = initial_weights.copy()
        value = float(initial_value_inr)
        states = []
        total_cost = 0.0
        rebalance_count = 0
        daily_returns = []
        quarter_months = quarterly_months or {1, 4, 7, 10}

        last_rebalance_date: date | None = None

        for i, dt in enumerate(dates):
            returns_today = self.market_returns.iloc[i].values

            # Apply market returns to weights
            new_values = weights * value * (1 + returns_today)
            new_value = new_values.sum()
            if new_value <= 0:
                break
            new_weights = new_values / new_value

            # Return for the day
            day_return = (new_value - value) / value
            daily_returns.append(day_return)

            # Rebalancing decision
            rebalanced = False
            cost = 0.0
            trigger = None

            should_rebalance = self._should_rebalance(
                strategy=strategy,
                weights=new_weights,
                target=target,
                band=band,
                dt=dt if isinstance(dt, date) else dt.date(),
                quarter_months=quarter_months,
                last_rebalance=last_rebalance_date,
            )

            if should_rebalance:
                cost = np.abs(new_weights - target).sum() * new_value * self.TRANSACTION_COST_PCT
                new_value -= cost
                total_cost += cost
                rebalance_count += 1
                new_weights = target.copy()
                rebalanced = True
                trigger = strategy
                last_rebalance_date = dt if isinstance(dt, date) else dt.date()

            states.append(
                DailyPortfolioState(
                    date=dt if isinstance(dt, date) else dt.date(),
                    weights=new_weights.copy(),
                    value_inr=round(new_value, 2),
                    rebalanced_today=rebalanced,
                    cost_inr=round(cost, 2),
                    trigger_type=trigger,
                )
            )

            weights = new_weights
            value = new_value

        result = BacktestResult(
            strategy_name=strategy,
            portfolio_id=portfolio_id,
            risk_category=risk_category,
            initial_value_inr=initial_value_inr,
            final_value_inr=value,
            daily_states=states,
            total_cost_inr=round(total_cost, 2),
            rebalance_count=rebalance_count,
        )
        self._compute_metrics(result, np.array(daily_returns), target)
        return result

    def _should_rebalance(
        self,
        strategy: str,
        weights: np.ndarray,
        target: np.ndarray,
        band: float,
        dt: date,
        quarter_months: set,
        last_rebalance: date | None,
    ) -> bool:
        max_drift = float(np.abs(weights - target).max())

        if strategy == "buy_and_hold":
            return False
        elif strategy == "agent":
            return max_drift > band
        elif strategy == "threshold_only":
            return max_drift > band * 1.5
        elif strategy == "legacy_quarterly":
            if dt.month in quarter_months and dt.day <= 5:
                # First 5 days of quarter
                if last_rebalance is None or (dt - last_rebalance).days > 60:
                    return True
            return False
        return False

    def _compute_metrics(
        self,
        result: BacktestResult,
        daily_returns: np.ndarray,
        target: np.ndarray,
    ) -> None:
        if len(daily_returns) == 0:
            return

        n_days = len(daily_returns)
        result.annualised_return = float((1 + daily_returns).prod() ** (252 / n_days) - 1)
        result.annualised_volatility = float(daily_returns.std() * np.sqrt(252))

        excess_returns = daily_returns - self.rf_daily
        if result.annualised_volatility > 0:
            result.sharpe_ratio = float(excess_returns.mean() / daily_returns.std() * np.sqrt(252))

        # Max drawdown
        cum = (1 + daily_returns).cumprod()
        running_max = np.maximum.accumulate(cum)
        drawdowns = (cum - running_max) / running_max
        result.max_drawdown = float(drawdowns.min())

        # Tracking error (vs target portfolio)
        final_weights = result.daily_states[-1].weights if result.daily_states else np.array([])
        if len(final_weights) > 0:
            result.tracking_error = float(np.abs(final_weights - target).max())

        # Turnover
        if result.rebalance_count > 0:
            result.turnover = float(result.total_cost_inr / result.initial_value_inr)
