"""Tests for DriftCalculator — all drift metrics and edge cases."""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_calculator import DRIFT_BANDS, DriftCalculator, DriftSeverity

ASSET_CLASSES = [
    "indian_equity",
    "international_equity",
    "indian_fixed_income",
    "international_fixed_income",
    "alternatives",
    "cash",
]


@pytest.fixture
def calculator():
    return DriftCalculator()


def _make_balanced_weights(drift: float = 0.0) -> np.ndarray:
    """Balanced target with optional drift in equity."""
    w = np.array([0.35, 0.15, 0.20, 0.10, 0.12, 0.08])
    w[0] += drift
    w[2] -= drift
    return w / w.sum()


def test_zero_drift(calculator):
    """Portfolio exactly at target should have NONE severity."""
    w = np.array([0.35, 0.15, 0.20, 0.10, 0.12, 0.08])
    result = calculator.compute_single("TEST001", w, "balanced")
    assert result.severity == DriftSeverity.NONE
    assert result.max_drift < 1e-6
    assert result.sum_abs_drift < 1e-6


def test_medium_drift(calculator):
    """Drift exactly at band should trigger MEDIUM severity."""
    w = _make_balanced_weights(drift=DRIFT_BANDS["balanced"])
    result = calculator.compute_single("TEST002", w, "balanced")
    assert result.severity in (DriftSeverity.MEDIUM, DriftSeverity.HIGH)
    assert result.max_drift > 0


def test_critical_drift(calculator):
    """2x band drift should trigger CRITICAL."""
    w = _make_balanced_weights(drift=DRIFT_BANDS["balanced"] * 2.0)
    result = calculator.compute_single("TEST003", w, "balanced")
    assert result.severity == DriftSeverity.CRITICAL


def test_batch_vectorised(calculator):
    """Batch calculation returns correct count and is sorted by severity."""
    n = 100
    rng = np.random.default_rng(42)
    weights = rng.dirichlet(np.ones(6), size=n)
    df = pd.DataFrame(weights, columns=ASSET_CLASSES)
    df.index = [f"P{i:04d}" for i in range(n)]
    cats = pd.Series(["balanced"] * n, index=df.index)

    results = calculator.compute_batch(df, cats)
    assert len(results) == n

    # Verify sorted (critical/high before none)
    severities = [r.severity for r in results]

    order_map = {
        DriftSeverity.CRITICAL: 0,
        DriftSeverity.HIGH: 1,
        DriftSeverity.MEDIUM: 2,
        DriftSeverity.LOW: 3,
        DriftSeverity.NONE: 4,
    }
    orders = [order_map[s] for s in severities]
    assert orders == sorted(orders)


def test_missing_price_handled(calculator):
    """Zero-weight portfolio (degenerate case) should not crash."""
    w = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # After normalisation it becomes all zeros — just shouldn't raise
    result = calculator.compute_single("TEST_ZERO", w, "conservative")
    assert result is not None


def test_all_risk_categories(calculator):
    """Test each risk category processes without error."""
    categories = [
        ("ultra_conservative", [0.10, 0.05, 0.45, 0.15, 0.10, 0.15]),
        ("conservative", [0.20, 0.10, 0.35, 0.10, 0.12, 0.13]),
        ("balanced", [0.35, 0.15, 0.20, 0.10, 0.12, 0.08]),
        ("aggressive", [0.50, 0.20, 0.10, 0.05, 0.10, 0.05]),
        ("ultra_aggressive", [0.60, 0.25, 0.05, 0.00, 0.07, 0.03]),
    ]
    for cat, target in categories:
        w = np.array(target)
        result = calculator.compute_single(f"TEST_{cat}", w, cat)
        assert result.severity == DriftSeverity.NONE, f"{cat} at target should have no drift"


def test_breaching_asset_classes(calculator):
    """Breaching asset classes should match drifted positions."""
    w = np.array([0.50, 0.15, 0.15, 0.10, 0.02, 0.08])  # equity +15% from balanced target
    result = calculator.compute_single("TEST_BREACH", w, "balanced")
    assert "indian_equity" in result.breaching_asset_classes


def test_rmsd_calculation(calculator):
    """RMSD should be between 0 and max_drift."""
    w = _make_balanced_weights(drift=0.05)
    result = calculator.compute_single("TEST_RMSD", w, "balanced")
    assert result.rmsd >= 0
    assert result.rmsd <= result.max_drift + 1e-9


def test_priority_queue(calculator):
    """Priority queue should return most-drifted first."""
    n = 20
    rng = np.random.default_rng(123)
    weights = rng.dirichlet(np.ones(6), size=n)
    df = pd.DataFrame(weights, columns=ASSET_CLASSES)
    df.index = [f"P{i:04d}" for i in range(n)]
    cats = pd.Series(["balanced"] * n, index=df.index)

    results = calculator.compute_batch(df, cats)
    queue = calculator.build_priority_queue(results)

    import heapq

    drifts = []
    while queue:
        neg_drift, _ = heapq.heappop(queue)
        drifts.append(-neg_drift)
    assert drifts == sorted(drifts, reverse=True)
