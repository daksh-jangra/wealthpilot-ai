"""Tests for DriftMonitor, ThresholdManager, and monitoring pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_calculator import DriftSeverity
from src.monitoring.drift_monitor import DriftMonitor
from src.monitoring.threshold_manager import BASE_DRIFT_BANDS, ThresholdManager

ASSET_CLASSES = [
    "indian_equity",
    "international_equity",
    "indian_fixed_income",
    "international_fixed_income",
    "alternatives",
    "cash",
]


def _make_weights(n: int = 100, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(6), size=n)
    df = pd.DataFrame(w, columns=ASSET_CLASSES)
    df.index = [f"P{i:04d}" for i in range(n)]
    cats = pd.Series(["balanced"] * n, index=df.index)
    return df, cats


# ThresholdManager
def test_threshold_manager_base_bands():
    tm = ThresholdManager()
    assert tm.get_drift_band("balanced") == BASE_DRIFT_BANDS["balanced"]
    assert tm.get_drift_band("ultra_conservative") == BASE_DRIFT_BANDS["ultra_conservative"]


def test_threshold_manager_client_override():
    tm = ThresholdManager()
    tm.set_client_override("WP000001", 0.015)
    assert tm.get_drift_band("balanced", "WP000001") == 0.015


def test_threshold_manager_remove_override():
    tm = ThresholdManager()
    tm.set_client_override("WP000001", 0.015)
    tm.remove_client_override("WP000001")
    assert tm.get_drift_band("balanced", "WP000001") == BASE_DRIFT_BANDS["balanced"]


def test_threshold_manager_invalid_band_raises():
    tm = ThresholdManager()
    with pytest.raises(ValueError):
        tm.set_client_override("WP000001", -0.01)
    with pytest.raises(ValueError):
        tm.set_client_override("WP000001", 0.50)


def test_threshold_manager_tighten():
    tm = ThresholdManager()
    original_balanced = BASE_DRIFT_BANDS["balanced"]
    tm.tighten_bands(0.80)
    assert abs(tm.get_drift_band("balanced") - original_balanced * 0.80) < 1e-9
    tm.restore_bands()
    assert abs(tm.get_drift_band("balanced") - original_balanced) < 1e-9


def test_threshold_manager_bulk_resolve():
    tm = ThresholdManager()
    mapping = {f"P{i:04d}": "balanced" for i in range(5)}
    bands = tm.get_all_bands(mapping)
    assert len(bands) == 5
    assert all(v == BASE_DRIFT_BANDS["balanced"] for v in bands.values())


# DriftMonitor
def test_drift_monitor_run_scan():
    monitor = DriftMonitor()
    weights, cats = _make_weights(100)
    summary = monitor.run_scan(weights, cats)
    assert summary["total_portfolios"] == 100
    assert "actionable_count" in summary


def test_drift_monitor_scan_returns_results():
    monitor = DriftMonitor()
    weights, cats = _make_weights(50)
    monitor.run_scan(weights, cats)
    results = monitor.get_results_dataframe()
    assert len(results) == 50


def test_drift_monitor_get_critical_portfolios():
    monitor = DriftMonitor()
    # Create portfolios with extreme drift
    w = pd.DataFrame(
        [
            [0.80, 0.10, 0.05, 0.02, 0.02, 0.01],  # extreme equity overweight
            [0.35, 0.15, 0.20, 0.10, 0.12, 0.08],  # at target
        ],
        columns=ASSET_CLASSES,
        index=["P0000", "P0001"],
    )
    cats = pd.Series(["balanced", "balanced"], index=w.index)
    monitor.run_scan(w, cats)
    critical = monitor.get_critical_portfolios()
    assert any(r.portfolio_id == "P0000" for r in critical)


def test_drift_monitor_priority_queue_ordering():
    monitor = DriftMonitor()
    weights, cats = _make_weights(50)
    monitor.run_scan(weights, cats)

    top = monitor.get_top_n(n=10)
    if len(top) >= 2:
        # First should have higher or equal drift than second
        assert top[0].max_drift >= top[1].max_drift - 1e-9


def test_drift_monitor_heatmap_data():
    monitor = DriftMonitor()
    weights, cats = _make_weights(50)
    monitor.run_scan(weights, cats)
    heatmap = monitor.drift_heatmap_data()
    assert not heatmap.empty
    assert "risk_category" in heatmap.columns


def test_drift_monitor_get_next_portfolio():
    monitor = DriftMonitor()
    # Create drifted portfolio
    w = pd.DataFrame(
        [[0.80, 0.10, 0.05, 0.02, 0.02, 0.01]],
        columns=ASSET_CLASSES,
        index=["P_HIGH"],
    )
    cats = pd.Series(["balanced"], index=w.index)
    monitor.run_scan(w, cats)
    result = monitor.get_next_portfolio()
    # Should return the drifted portfolio or None if all NONE severity
    # (extreme weights should trigger CRITICAL)
    if result is not None:
        assert result.severity != DriftSeverity.NONE
