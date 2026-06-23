"""Compute and compare performance metrics across backtesting strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.backtesting.backtest_engine import BacktestResult


class PerformanceAnalyser:
    """Aggregate and compare performance metrics from backtest runs."""

    def summarise(self, result: BacktestResult) -> dict:
        """Return a flat summary dict for a single backtest result."""
        return {
            "strategy": result.strategy_name,
            "portfolio_id": result.portfolio_id,
            "risk_category": result.risk_category,
            "initial_value_inr": result.initial_value_inr,
            "final_value_inr": round(result.final_value_inr, 2),
            "total_return_pct": round(
                (result.final_value_inr / result.initial_value_inr - 1) * 100, 2
            ),
            "annualised_return_pct": round(result.annualised_return * 100, 2),
            "annualised_volatility_pct": round(result.annualised_volatility * 100, 2),
            "sharpe_ratio": round(result.sharpe_ratio, 3),
            "max_drawdown_pct": round(result.max_drawdown * 100, 2),
            "tracking_error_pct": round(result.tracking_error * 100, 3),
            "total_cost_inr": result.total_cost_inr,
            "rebalance_count": result.rebalance_count,
            "turnover_pct": round(result.turnover * 100, 2),
        }

    def compare(self, results: list[BacktestResult]) -> pd.DataFrame:
        """Compare all strategies in a summary DataFrame."""
        rows = [self.summarise(r) for r in results]
        return pd.DataFrame(rows).set_index("strategy")

    def statistical_significance(
        self,
        agent_result: BacktestResult,
        legacy_result: BacktestResult,
        confidence: float = 0.95,
    ) -> dict:
        """Paired t-test: is the agent's Sharpe significantly better than legacy?"""
        agent_returns = np.array([s.value_inr for s in agent_result.daily_states])
        legacy_returns = np.array([s.value_inr for s in legacy_result.daily_states])

        min_len = min(len(agent_returns), len(legacy_returns))
        if min_len < 10:
            return {"error": "Insufficient data for significance test"}

        agent_r = np.diff(agent_returns[:min_len]) / agent_returns[:min_len - 1]
        legacy_r = np.diff(legacy_returns[:min_len]) / legacy_returns[:min_len - 1]

        t_stat, p_value = stats.ttest_rel(agent_r, legacy_r)
        return {
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_value), 4),
            "significant": bool(p_value < (1 - confidence)),
            "confidence_level": confidence,
            "agent_mean_daily_return": round(float(agent_r.mean()), 6),
            "legacy_mean_daily_return": round(float(legacy_r.mean()), 6),
        }

    def improvement_scorecard(
        self,
        agent: dict,
        legacy: dict,
    ) -> dict:
        """Score the agent vs legacy on 6 key metrics."""
        metrics = {
            "sharpe_ratio": "higher_is_better",
            "max_drawdown_pct": "lower_is_better",
            "tracking_error_pct": "lower_is_better",
            "turnover_pct": "lower_is_better",
            "total_cost_inr": "lower_is_better",
            "annualised_return_pct": "higher_is_better",
        }
        wins = 0
        comparison = {}
        for metric, direction in metrics.items():
            av = agent.get(metric, 0)
            lv = legacy.get(metric, 0)
            if direction == "higher_is_better":
                agent_wins = av > lv
            else:
                agent_wins = av < lv
            if agent_wins:
                wins += 1
            comparison[metric] = {
                "agent": av,
                "legacy": lv,
                "agent_wins": agent_wins,
            }
        return {
            "agent_wins": wins,
            "total_metrics": len(metrics),
            "target_met": wins >= 4,
            "details": comparison,
        }

    def rolling_metrics(
        self, result: BacktestResult, window: int = 63
    ) -> pd.DataFrame:
        """Compute rolling 63-day (3-month) Sharpe and volatility."""
        values = [s.value_inr for s in result.daily_states]
        returns = pd.Series(values).pct_change().dropna()
        rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        return pd.DataFrame({
            "date": [s.date for s in result.daily_states[window + 1:]],
            "rolling_sharpe": rolling_sharpe.values[window:],
            "rolling_volatility": rolling_vol.values[window:],
        })
