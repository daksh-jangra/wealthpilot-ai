"""Tests for BacktestEngine and PerformanceAnalyser."""

import numpy as np
import pandas as pd
import pytest

from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.performance_analyser import PerformanceAnalyser


@pytest.fixture
def market_returns():
    """Minimal 50-day market return series for testing."""
    rng = np.random.default_rng(42)
    n_days = 50
    returns = rng.normal(0, 0.01, size=(n_days, 6))
    cols = [
        "indian_equity",
        "international_equity",
        "indian_fixed_income",
        "international_fixed_income",
        "alternatives",
        "cash",
    ]
    dates = pd.bdate_range(start="2024-04-01", periods=n_days, freq="B")
    return pd.DataFrame(returns, index=dates, columns=cols)


@pytest.fixture
def engine(market_returns):
    return BacktestEngine(market_returns)


@pytest.fixture
def balanced_weights():
    return np.array([0.35, 0.15, 0.20, 0.10, 0.12, 0.08])


def test_agent_strategy_completes(engine, balanced_weights):
    result = engine.run("TEST001", "balanced", balanced_weights, 1_000_000, "agent")
    assert result is not None
    assert result.final_value_inr > 0


def test_all_strategies_complete(engine, balanced_weights):
    for strategy in ["agent", "legacy_quarterly", "threshold_only", "buy_and_hold"]:
        result = engine.run("TEST002", "balanced", balanced_weights, 1_000_000, strategy)
        assert result.strategy_name == strategy
        assert len(result.daily_states) > 0


def test_buy_and_hold_never_rebalances(engine, balanced_weights):
    result = engine.run("TEST003", "balanced", balanced_weights, 1_000_000, "buy_and_hold")
    assert result.rebalance_count == 0
    assert result.total_cost_inr == 0.0


def test_agent_rebalances_on_drift(engine):
    """Agent should rebalance when drift exceeds band."""
    # Start with significantly drifted weights
    drifted = np.array([0.55, 0.15, 0.10, 0.05, 0.07, 0.08])
    result = engine.run("TEST004", "balanced", drifted, 1_000_000, "agent")
    assert result.rebalance_count >= 1


def test_final_value_positive(engine, balanced_weights):
    result = engine.run("TEST005", "balanced", balanced_weights, 500_000, "agent")
    assert result.final_value_inr > 0


def test_metrics_computed(engine, balanced_weights):
    result = engine.run("TEST006", "balanced", balanced_weights, 1_000_000, "agent")
    assert isinstance(result.sharpe_ratio, float)
    assert isinstance(result.max_drawdown, float)
    assert result.max_drawdown <= 0.0  # max drawdown is non-positive


def test_performance_analyser_summarise(engine, balanced_weights):
    result = engine.run("TEST007", "balanced", balanced_weights, 1_000_000, "agent")
    analyser = PerformanceAnalyser()
    summary = analyser.summarise(result)
    assert "sharpe_ratio" in summary
    assert "max_drawdown_pct" in summary
    assert "annualised_return_pct" in summary


def test_performance_comparison(engine, balanced_weights):
    analyser = PerformanceAnalyser()
    agent = engine.run("TEST008", "balanced", balanced_weights, 1_000_000, "agent")
    legacy = engine.run("TEST009", "balanced", balanced_weights, 1_000_000, "legacy_quarterly")
    comparison = analyser.compare([agent, legacy])
    assert "agent" in comparison.index
    assert "legacy_quarterly" in comparison.index


def test_all_risk_categories(engine):
    categories = [
        ("ultra_conservative", [0.10, 0.05, 0.45, 0.15, 0.10, 0.15]),
        ("conservative", [0.20, 0.10, 0.35, 0.10, 0.12, 0.13]),
        ("balanced", [0.35, 0.15, 0.20, 0.10, 0.12, 0.08]),
        ("aggressive", [0.50, 0.20, 0.10, 0.05, 0.10, 0.05]),
        ("ultra_aggressive", [0.60, 0.25, 0.05, 0.00, 0.07, 0.03]),
    ]
    for cat, weights in categories:
        result = engine.run(f"TEST_{cat}", cat, np.array(weights), 1_000_000, "agent")
        assert result.final_value_inr > 0, f"Strategy failed for {cat}"
