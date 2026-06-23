"""Tests for scenario runner, strategy comparator, and performance analyser."""

import numpy as np
import pandas as pd
import pytest

from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.performance_analyser import PerformanceAnalyser
from src.backtesting.scenario_runner import ScenarioRunner
from src.backtesting.strategy_comparator import StrategyComparator


ASSET_CLASSES = [
    "indian_equity", "international_equity",
    "indian_fixed_income", "international_fixed_income",
    "alternatives", "cash",
]
BALANCED_WEIGHTS = np.array([0.35, 0.15, 0.20, 0.10, 0.12, 0.08])


def _make_market_returns(n: int = 252, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.01, size=(n, len(ASSET_CLASSES)))
    index = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame(returns, columns=ASSET_CLASSES, index=index)


def _make_portfolios(n: int = 5) -> list[dict]:
    return [
        {
            "portfolio_id": f"WP{i:06d}",
            "risk_category": "balanced",
            "initial_weights": BALANCED_WEIGHTS.tolist(),
            "initial_value_inr": 1_000_000,
        }
        for i in range(n)
    ]


def _make_portfolios_with_intl(n: int = 5) -> list[dict]:
    weights_intl = np.array([0.30, 0.20, 0.15, 0.10, 0.12, 0.13])  # intl > 15%
    return [
        {
            "portfolio_id": f"WP{i:06d}",
            "risk_category": "balanced",
            "initial_weights": weights_intl.tolist(),
            "initial_value_inr": 1_000_000,
        }
        for i in range(n)
    ]


# ── ScenarioRunner ────────────────────────────────────────────────────────────

def test_scenario_1_normal_drift():
    market = _make_market_returns()
    runner = ScenarioRunner(market)
    result = runner.run_scenario_1_normal_drift(_make_portfolios(3))
    assert result.scenario_name == "Scenario 1: Normal Drift"
    assert len(result.agent_results) == 3
    assert len(result.legacy_results) == 3
    assert "agent" in result.scenario_metrics
    assert "legacy" in result.scenario_metrics


def test_scenario_2_market_crash():
    market = _make_market_returns()
    runner = ScenarioRunner(market)
    result = runner.run_scenario_2_market_crash(_make_portfolios(3))
    assert result.scenario_name == "Scenario 2: Market Crash"
    assert result.affected_portfolios == 3


def test_scenario_3_sector_rotation():
    market = _make_market_returns()
    runner = ScenarioRunner(market)
    result = runner.run_scenario_3_sector_rotation(_make_portfolios(3))
    assert "Sector Rotation" in result.scenario_name


def test_scenario_4_regulatory_event():
    market = _make_market_returns()
    runner = ScenarioRunner(market)
    portfolios = _make_portfolios_with_intl(5)
    result = runner.run_scenario_4_regulatory_event(portfolios)
    assert "Regulatory" in result.scenario_name
    assert result.affected_portfolios <= len(portfolios)


def test_scenario_5_tax_harvesting():
    market = _make_market_returns()
    runner = ScenarioRunner(market)
    result = runner.run_scenario_5_tax_harvesting(_make_portfolios(3))
    assert "Tax Harvesting" in result.scenario_name


def test_scenario_metrics_structure():
    market = _make_market_returns()
    runner = ScenarioRunner(market)
    result = runner.run_scenario_1_normal_drift(_make_portfolios(2))
    m = result.scenario_metrics
    assert "avg_sharpe" in m["agent"]
    assert "avg_max_drawdown" in m["agent"]
    assert "avg_tracking_error" in m["agent"]


def test_scenario_does_not_mutate_original_returns():
    market = _make_market_returns()
    original_first = market.iloc[0, 0]
    runner = ScenarioRunner(market)
    runner.run_scenario_2_market_crash(_make_portfolios(2))
    assert market.iloc[0, 0] == pytest.approx(original_first)


# ── StrategyComparator ────────────────────────────────────────────────────────

def test_strategy_comparator_compare_all():
    market = _make_market_returns()
    engine = BacktestEngine(market)
    comparator = StrategyComparator(engine)
    result = comparator.compare_all("WP000001", "balanced", BALANCED_WEIGHTS)
    assert "strategy_summaries" in result
    assert len(result["strategy_summaries"]) == 4
    assert "agent" in result["strategy_summaries"]
    assert "legacy_quarterly" in result["strategy_summaries"]
    assert isinstance(result["agent_beats_legacy"], bool)


def test_strategy_comparator_improvement_scorecard():
    market = _make_market_returns()
    engine = BacktestEngine(market)
    comparator = StrategyComparator(engine)
    result = comparator.compare_all("WP000001", "balanced", BALANCED_WEIGHTS)
    sc = result["improvement_scorecard"]
    assert "agent_wins" in sc
    assert "total_metrics" in sc
    assert sc["total_metrics"] == 6
    assert "target_met" in sc


def test_strategy_comparator_statistical_significance():
    market = _make_market_returns()
    engine = BacktestEngine(market)
    comparator = StrategyComparator(engine)
    result = comparator.compare_all("WP000001", "balanced", BALANCED_WEIGHTS)
    sig = result["statistical_significance"]
    assert "t_statistic" in sig or "error" in sig


def test_strategy_comparator_batch():
    market = _make_market_returns()
    engine = BacktestEngine(market)
    comparator = StrategyComparator(engine)
    portfolios = _make_portfolios(2)
    df = comparator.compare_batch(portfolios)
    assert not df.empty
    assert "portfolio_id" in df.columns


# ── PerformanceAnalyser ───────────────────────────────────────────────────────

def test_performance_analyser_summarise():
    market = _make_market_returns()
    engine = BacktestEngine(market)
    result = engine.run("WP000001", "balanced", BALANCED_WEIGHTS, 1_000_000, "agent")
    analyser = PerformanceAnalyser()
    summary = analyser.summarise(result)
    assert "strategy" in summary
    assert "sharpe_ratio" in summary
    assert "max_drawdown_pct" in summary
    assert "tracking_error_pct" in summary


def test_performance_analyser_compare_dataframe():
    market = _make_market_returns()
    engine = BacktestEngine(market)
    results = [engine.run("WP000001", "balanced", BALANCED_WEIGHTS, 1_000_000, s)
               for s in ["agent", "legacy_quarterly"]]
    analyser = PerformanceAnalyser()
    df = analyser.compare(results)
    assert "agent" in df.index
    assert "legacy_quarterly" in df.index


def test_performance_analyser_rolling_metrics():
    market = _make_market_returns()
    engine = BacktestEngine(market)
    result = engine.run("WP000001", "balanced", BALANCED_WEIGHTS, 1_000_000, "agent")
    analyser = PerformanceAnalyser()
    rolling = analyser.rolling_metrics(result, window=30)
    assert not rolling.empty
    assert "rolling_sharpe" in rolling.columns
    assert "rolling_volatility" in rolling.columns


def test_performance_analyser_improvement_scorecard_all_better():
    analyser = PerformanceAnalyser()
    agent = {
        "sharpe_ratio": 1.5, "max_drawdown_pct": 5.0, "tracking_error_pct": 1.0,
        "turnover_pct": 10.0, "total_cost_inr": 5000, "annualised_return_pct": 12.0,
    }
    legacy = {
        "sharpe_ratio": 1.0, "max_drawdown_pct": 8.0, "tracking_error_pct": 2.0,
        "turnover_pct": 20.0, "total_cost_inr": 10000, "annualised_return_pct": 10.0,
    }
    sc = analyser.improvement_scorecard(agent, legacy)
    assert sc["agent_wins"] == 6
    assert sc["target_met"] is True


def test_performance_analyser_significance_insufficient_data():
    market = _make_market_returns(n=5)  # very short, < 10 days
    engine = BacktestEngine(market)
    r1 = engine.run("WP000001", "balanced", BALANCED_WEIGHTS, 1_000_000, "agent")
    r2 = engine.run("WP000001", "balanced", BALANCED_WEIGHTS, 1_000_000, "legacy_quarterly")
    analyser = PerformanceAnalyser()
    result = analyser.statistical_significance(r1, r2)
    # May return error or valid result depending on days
    assert isinstance(result, dict)
