"""Run the five market scenarios defined in the project specification."""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from src.backtesting.backtest_engine import BacktestEngine, BacktestResult


@dataclass
class ScenarioResult:
    scenario_name: str
    scenario_description: str
    affected_portfolios: int
    agent_results: list[BacktestResult]
    legacy_results: list[BacktestResult]
    scenario_metrics: dict


class ScenarioRunner:
    """
    Execute all five market scenarios from the project specification:
    1. Normal Drift
    2. Market Crash
    3. Sector Rotation
    4. Regulatory Event
    5. Tax Harvesting Window
    """

    def __init__(self, base_market_returns: pd.DataFrame):
        self.base_returns = base_market_returns

    def run_scenario_1_normal_drift(
        self,
        portfolios: list[dict],
        rally_pct: float = 0.13,
    ) -> ScenarioResult:
        """Gradual equity market appreciation of 12-15% over 3 months."""
        # Boost equity returns for 63 trading days
        modified = self.base_returns.copy()
        modified.iloc[:63, 0] += rally_pct / 63  # indian_equity daily boost

        engine = BacktestEngine(modified)
        agent_results = []
        legacy_results = []
        for p in portfolios[:100]:  # sample 100 for speed
            weights = np.array(p["initial_weights"])
            agent_results.append(engine.run(p["portfolio_id"], p["risk_category"], weights, 1_000_000, "agent"))
            legacy_results.append(engine.run(p["portfolio_id"], p["risk_category"], weights, 1_000_000, "legacy_quarterly"))

        return ScenarioResult(
            scenario_name="Scenario 1: Normal Drift",
            scenario_description="Equity rally of 12-15% over 3 months causes systematic drift",
            affected_portfolios=len(portfolios),
            agent_results=agent_results,
            legacy_results=legacy_results,
            scenario_metrics=self._aggregate_metrics(agent_results, legacy_results),
        )

    def run_scenario_2_market_crash(
        self,
        portfolios: list[dict],
        crash_pct: float = -0.22,
    ) -> ScenarioResult:
        """22% equity market correction over 5 trading sessions."""
        modified = self.base_returns.copy()
        modified.iloc[30:35, 0] = crash_pct / 5  # 5-day crash

        engine = BacktestEngine(modified)
        agent_results = []
        legacy_results = []
        for p in portfolios[:100]:
            weights = np.array(p["initial_weights"])
            agent_results.append(engine.run(p["portfolio_id"], p["risk_category"], weights, 1_000_000, "agent"))
            legacy_results.append(engine.run(p["portfolio_id"], p["risk_category"], weights, 1_000_000, "legacy_quarterly"))

        return ScenarioResult(
            scenario_name="Scenario 2: Market Crash",
            scenario_description="22% equity correction over 5 sessions — crisis response test",
            affected_portfolios=len(portfolios),
            agent_results=agent_results,
            legacy_results=legacy_results,
            scenario_metrics=self._aggregate_metrics(agent_results, legacy_results),
        )

    def run_scenario_3_sector_rotation(self, portfolios: list[dict]) -> ScenarioResult:
        """Growth-to-value rotation — intra-equity drift test."""
        modified = self.base_returns.copy()
        # Simulate sector rotation as differential within equity
        # (simplified: overall equity flat but increased sector concentration)
        modified.iloc[20:50, 0] += 0.001  # slight equity tilt

        engine = BacktestEngine(modified)
        agent_results = []
        legacy_results = []
        for p in portfolios[:50]:
            weights = np.array(p["initial_weights"])
            agent_results.append(engine.run(p["portfolio_id"], p["risk_category"], weights, 1_000_000, "agent"))
            legacy_results.append(engine.run(p["portfolio_id"], p["risk_category"], weights, 1_000_000, "legacy_quarterly"))

        return ScenarioResult(
            scenario_name="Scenario 3: Sector Rotation",
            scenario_description="Growth-to-value rotation causes intra-equity concentration drift",
            affected_portfolios=len(portfolios),
            agent_results=agent_results,
            legacy_results=legacy_results,
            scenario_metrics=self._aggregate_metrics(agent_results, legacy_results),
        )

    def run_scenario_4_regulatory_event(self, portfolios: list[dict]) -> ScenarioResult:
        """SEBI restricts international equity to 15% for retail investors."""
        modified = self.base_returns.copy()
        engine = BacktestEngine(modified)
        agent_results = []
        legacy_results = []

        # Identify portfolios with international equity > 15%
        affected = [p for p in portfolios if p["initial_weights"][1] > 0.15][:50]
        for p in affected:
            weights = np.array(p["initial_weights"])
            # Agent complies immediately; legacy waits for quarterly
            agent_results.append(engine.run(p["portfolio_id"], p["risk_category"], weights, 1_000_000, "agent"))
            legacy_results.append(engine.run(p["portfolio_id"], p["risk_category"], weights, 1_000_000, "legacy_quarterly"))

        return ScenarioResult(
            scenario_name="Scenario 4: Regulatory Event",
            scenario_description="SEBI circular restricts international equity to 15% for retail",
            affected_portfolios=len(affected),
            agent_results=agent_results,
            legacy_results=legacy_results,
            scenario_metrics=self._aggregate_metrics(agent_results, legacy_results),
        )

    def run_scenario_5_tax_harvesting(self, portfolios: list[dict]) -> ScenarioResult:
        """March FY-end tax-loss harvesting opportunity."""
        modified = self.base_returns.copy()
        # Create some losses in equity during first 60 days
        modified.iloc[10:15, 0] = -0.05  # 5-day dip to create harvestable losses

        engine = BacktestEngine(modified)
        agent_results = []
        legacy_results = []
        for p in portfolios[:50]:
            weights = np.array(p["initial_weights"])
            agent_results.append(engine.run(p["portfolio_id"], p["risk_category"], weights, 1_000_000, "agent"))
            legacy_results.append(engine.run(p["portfolio_id"], p["risk_category"], weights, 1_000_000, "legacy_quarterly"))

        return ScenarioResult(
            scenario_name="Scenario 5: Tax Harvesting Window",
            scenario_description="March FY-end tax-loss harvesting across portfolios with harvestable losses",
            affected_portfolios=len(portfolios),
            agent_results=agent_results,
            legacy_results=legacy_results,
            scenario_metrics=self._aggregate_metrics(agent_results, legacy_results),
        )

    def _aggregate_metrics(
        self,
        agent_results: list[BacktestResult],
        legacy_results: list[BacktestResult],
    ) -> dict:
        def avg(results: list[BacktestResult], attr: str) -> float:
            vals = [getattr(r, attr, 0) for r in results if r]
            return round(float(np.mean(vals)) if vals else 0.0, 4)

        return {
            "agent": {
                "avg_sharpe": avg(agent_results, "sharpe_ratio"),
                "avg_max_drawdown": avg(agent_results, "max_drawdown"),
                "avg_tracking_error": avg(agent_results, "tracking_error"),
                "avg_rebalance_count": avg(agent_results, "rebalance_count"),
            },
            "legacy": {
                "avg_sharpe": avg(legacy_results, "sharpe_ratio"),
                "avg_max_drawdown": avg(legacy_results, "max_drawdown"),
                "avg_tracking_error": avg(legacy_results, "tracking_error"),
                "avg_rebalance_count": avg(legacy_results, "rebalance_count"),
            },
        }
