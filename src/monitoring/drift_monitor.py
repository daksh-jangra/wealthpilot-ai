"""DriftMonitor: orchestrates batch scans and produces prioritised rebalancing queues."""

from __future__ import annotations

import time
import heapq
from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger

from src.monitoring.drift_calculator import DriftCalculator, DriftResult, DriftSeverity
from src.monitoring.threshold_manager import ThresholdManager


class DriftMonitor:
    """
    Service that runs periodic drift scans across all portfolios and maintains
    a priority queue of portfolios requiring rebalancing attention.
    """

    def __init__(
        self,
        calculator: Optional[DriftCalculator] = None,
        threshold_manager: Optional[ThresholdManager] = None,
    ):
        self.calculator = calculator or DriftCalculator()
        self.threshold_manager = threshold_manager or ThresholdManager()
        self._last_scan_time: Optional[datetime] = None
        self._priority_queue: list[tuple] = []
        self._latest_results: list[DriftResult] = []

    def run_scan(
        self,
        current_weights: pd.DataFrame,
        risk_categories: pd.Series,
    ) -> dict:
        """
        Execute a full drift scan across all portfolios.

        Returns:
            Summary dict with scan metadata and actionable portfolio count.
        """
        t0 = time.perf_counter()
        logger.info(f"Starting drift scan for {len(current_weights):,} portfolios")

        results = self.calculator.compute_batch(current_weights, risk_categories)
        elapsed = time.perf_counter() - t0

        self._latest_results = results
        self._last_scan_time = datetime.utcnow()
        self._priority_queue = self.calculator.build_priority_queue(results)

        actionable = [r for r in results if r.severity != DriftSeverity.NONE]
        critical = [r for r in results if r.severity == DriftSeverity.CRITICAL]

        summary = {
            "scan_timestamp": self._last_scan_time.isoformat(),
            "total_portfolios": len(results),
            "elapsed_seconds": round(elapsed, 2),
            "actionable_count": len(actionable),
            "critical_count": len(critical),
            "high_count": sum(1 for r in results if r.severity == DriftSeverity.HIGH),
            "medium_count": sum(1 for r in results if r.severity == DriftSeverity.MEDIUM),
            "performance_ok": elapsed < 30.0,
        }

        if elapsed > 30.0:
            logger.warning(f"Scan took {elapsed:.1f}s — exceeds 30s target")
        else:
            logger.success(
                f"Scan complete in {elapsed:.2f}s: {len(actionable)} portfolios need attention"
            )

        return summary

    def get_next_portfolio(self) -> Optional[DriftResult]:
        """Pop the highest-priority portfolio from the queue."""
        if not self._priority_queue:
            return None
        _, portfolio_id = heapq.heappop(self._priority_queue)
        for r in self._latest_results:
            if r.portfolio_id == portfolio_id:
                return r
        return None

    def get_top_n(self, n: int = 100) -> list[DriftResult]:
        """Return top N portfolios by drift severity without consuming queue."""
        return [r for r in self._latest_results if r.severity != DriftSeverity.NONE][:n]

    def get_critical_portfolios(self) -> list[DriftResult]:
        return [r for r in self._latest_results if r.severity == DriftSeverity.CRITICAL]

    def get_results_dataframe(self) -> pd.DataFrame:
        return self.calculator.results_to_dataframe(self._latest_results)

    def drift_heatmap_data(self) -> pd.DataFrame:
        """Return per-category drift statistics for dashboard heatmap."""
        if not self._latest_results:
            return pd.DataFrame()
        df = self.get_results_dataframe()
        return (
            df.groupby("risk_category")
            .agg(
                mean_max_drift=("max_drift", "mean"),
                max_max_drift=("max_drift", "max"),
                actionable_count=("severity", lambda s: (s != "none").sum()),
                critical_count=("severity", lambda s: (s == "critical").sum()),
            )
            .reset_index()
        )
