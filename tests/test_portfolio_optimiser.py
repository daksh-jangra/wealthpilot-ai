"""Tests for PortfolioOptimiser: convergence, constraint satisfaction, edge cases."""

import numpy as np
import pytest

pytest.importorskip("cvxpy")

from src.optimisation.portfolio_optimiser import PortfolioOptimiser
from src.optimisation.constraint_manager import ConstraintManager


@pytest.fixture
def optimiser():
    return PortfolioOptimiser()


def test_optimiser_returns_valid_weights(optimiser):
    current = np.array([0.40, 0.15, 0.15, 0.10, 0.12, 0.08])
    target = np.array([0.35, 0.15, 0.20, 0.10, 0.12, 0.08])
    result = optimiser.optimise(current, target)

    assert result.status in ("optimal", "optimal_inaccurate")
    post = result.post_trade_weights
    assert abs(post.sum() - 1.0) < 0.01, "Weights must sum to 1"
    assert (post >= -1e-6).all(), "Weights must be non-negative"


def test_optimiser_reduces_tracking_error(optimiser):
    current = np.array([0.50, 0.20, 0.10, 0.05, 0.10, 0.05])
    target = np.array([0.35, 0.15, 0.20, 0.10, 0.12, 0.08])
    result = optimiser.optimise(current, target)

    pre_te = float(np.abs(current - target).max())
    post_te = float(np.abs(result.post_trade_weights - target).max())
    assert post_te < pre_te, "Optimiser should reduce tracking error"


def test_turnover_constraint_respected(optimiser):
    current = np.array([0.50, 0.20, 0.10, 0.05, 0.10, 0.05])
    target = np.array([0.35, 0.15, 0.20, 0.10, 0.12, 0.08])
    budget = 0.10
    result = optimiser.optimise(current, target, turnover_budget=budget)
    assert (
        result.turnover <= budget + 1e-3
    ), f"Turnover {result.turnover:.4f} exceeds budget {budget}"


def test_international_equity_sebi_limit(optimiser):
    current = np.array([0.30, 0.30, 0.15, 0.05, 0.12, 0.08])
    target = np.array([0.35, 0.30, 0.20, 0.00, 0.07, 0.08])  # intl equity = 30%
    result = optimiser.optimise(current, target)
    assert (
        result.post_trade_weights[1] <= 0.25 + 1e-3
    ), "International equity must not exceed SEBI 25% limit"


def test_long_only_constraint(optimiser):
    current = np.array([0.40, 0.15, 0.15, 0.10, 0.12, 0.08])
    target = np.array([0.00, 0.00, 0.50, 0.20, 0.20, 0.10])  # sell all equity
    result = optimiser.optimise(current, target)
    assert (result.post_trade_weights >= -1e-6).all(), "No short positions allowed"


def test_exact_target_is_optimal(optimiser):
    """If current == target, trades should be zero."""
    target = np.array([0.35, 0.15, 0.20, 0.10, 0.12, 0.08])
    result = optimiser.optimise(target.copy(), target.copy())
    assert np.abs(result.trade_weights).sum() < 0.02  # near-zero turnover


def test_partial_rebalance(optimiser):
    current = np.array([0.45, 0.15, 0.15, 0.10, 0.07, 0.08])
    target = np.array([0.35, 0.15, 0.20, 0.10, 0.12, 0.08])
    result = optimiser.optimise_partial(current, target, rebalance_fraction=0.5)
    assert result.post_trade_weights is not None
    # Post-trade should be between current and target (closer to target than current)
    post_drift = np.abs(result.post_trade_weights - target).max()
    current_drift = np.abs(current - target).max()
    assert post_drift <= current_drift + 1e-3


def test_cash_flow_handling(optimiser):
    """Cash inflow should be invested according to target weights."""
    current = np.array([0.35, 0.15, 0.20, 0.10, 0.12, 0.08])
    target = np.array([0.35, 0.15, 0.20, 0.10, 0.12, 0.08])
    result = optimiser.optimise(current, target, cash_flow_fraction=0.05)
    assert result.post_trade_weights is not None
    assert abs(result.post_trade_weights.sum() - 1.0) < 0.02
