"""Run parallel backtests across strategies and produce comparison reports."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.backtesting.backtest_engine import BacktestEngine, BacktestResult
from src.backtesting.performance_analyser import PerformanceAnalyser

STRATEGIES = ["agent", "legacy_quarterly", "threshold_only", "buy_and_hold"]


class StrategyComparator:
    """Run all four strategies on the same portfolio and compare results."""

    def __init__(self, engine: BacktestEngine):
        self.engine = engine
        self.analyser = PerformanceAnalyser()

    def compare_all(
        self,
        portfolio_id: str,
        risk_category: str,
        initial_weights: np.ndarray,
        initial_value_inr: float = 1_000_000,
    ) -> dict:
        """Run all 4 strategies and produce the comparison report."""
        results: dict[str, BacktestResult] = {}
        for strategy in STRATEGIES:
            logger.debug(f"Running strategy: {strategy} for {portfolio_id}")
            results[strategy] = self.engine.run(
                portfolio_id=portfolio_id,
                risk_category=risk_category,
                initial_weights=initial_weights,
                initial_value_inr=initial_value_inr,
                strategy=strategy,
            )

        summaries = {s: self.analyser.summarise(r) for s, r in results.items()}
        self.analyser.compare(list(results.values()))

        significance = self.analyser.statistical_significance(
            results["agent"], results["legacy_quarterly"]
        )
        scorecard = self.analyser.improvement_scorecard(
            summaries["agent"], summaries["legacy_quarterly"]
        )

        return {
            "portfolio_id": portfolio_id,
            "risk_category": risk_category,
            "strategy_summaries": summaries,
            "statistical_significance": significance,
            "improvement_scorecard": scorecard,
            "agent_beats_legacy": scorecard["target_met"],
        }

    def compare_batch(
        self,
        portfolios: list[dict],
    ) -> pd.DataFrame:
        """Compare strategies across multiple portfolios."""
        rows = []
        for p in portfolios:
            result = self.compare_all(
                portfolio_id=p["portfolio_id"],
                risk_category=p["risk_category"],
                initial_weights=np.array(p["initial_weights"]),
                initial_value_inr=p.get("initial_value_inr", 1_000_000),
            )
            for strategy, summary in result["strategy_summaries"].items():
                rows.append({**summary, "portfolio_id": p["portfolio_id"]})
        return pd.DataFrame(rows)
