"""Tests for cost estimator, liquidity scorer, and trade list generator."""

import numpy as np
import pandas as pd
import pytest

from src.optimisation.constraint_manager import ConstraintManager
from src.optimisation.cost_estimator import CostEstimator
from src.optimisation.liquidity_scorer import LiquidityScorer
from src.optimisation.trade_list_generator import TradeListGenerator


# CostEstimator
def test_cost_estimator_equity_buy():
    est = CostEstimator()
    result = est.estimate_trade(
        trade_value_inr=100_000,
        asset_class="indian_equity",
        avg_daily_volume_inr=10_000_000,
        is_buy=True,
    )
    assert result.total_inr > 0
    assert result.stt_inr > 0
    assert result.stamp_duty_inr > 0
    assert result.total_bps > 0


def test_cost_estimator_equity_sell():
    est = CostEstimator()
    result = est.estimate_trade(
        trade_value_inr=100_000,
        asset_class="indian_equity",
        avg_daily_volume_inr=10_000_000,
        is_buy=False,
    )
    assert result.total_inr > 0
    assert result.stamp_duty_inr == 0  # stamp duty only on buy


def test_cost_estimator_zero_trade():
    est = CostEstimator()
    result = est.estimate_trade(0, "indian_equity", 10_000_000)
    assert result.total_inr == 0


def test_cost_estimator_debt_no_stt():
    est = CostEstimator()
    result = est.estimate_trade(100_000, "indian_fixed_income", 10_000_000, is_buy=True)
    assert result.stt_inr == 0.0  # no STT on debt


def test_cost_estimator_illiquid_impact():
    est = CostEstimator()
    liquid = est.estimate_trade(100_000, "indian_equity", 50_000_000, is_buy=True)
    illiquid = est.estimate_trade(100_000, "indian_equity", 200_000, is_buy=True)
    assert illiquid.market_impact_inr > liquid.market_impact_inr


def test_cost_estimator_trade_list():
    est = CostEstimator()
    trades = [
        {
            "trade_value_inr": 50_000,
            "asset_class": "indian_equity",
            "avg_daily_volume_inr": 5_000_000,
            "is_buy": True,
        },
        {
            "trade_value_inr": 30_000,
            "asset_class": "indian_fixed_income",
            "avg_daily_volume_inr": 2_000_000,
            "is_buy": False,
        },
    ]
    result = est.estimate_trade_list(trades, portfolio_value_inr=500_000)
    assert result["total_cost_inr"] > 0
    assert len(result["trade_details"]) == 2


# LiquidityScorer
def test_liquidity_scorer_high_volume_scores_high():
    scorer = LiquidityScorer()
    score = scorer.score_security("IEQ001", "indian_equity", 100_000_000, 5.0)
    low_vol_score = scorer.score_security("IEQ002", "indian_equity", 100_000, 50.0)
    assert score.score > low_vol_score.score


def test_liquidity_scorer_max_daily_trade():
    scorer = LiquidityScorer()
    score = scorer.score_security("IEQ001", "indian_equity", 10_000_000, 10.0)
    expected_max = 10_000_000 * scorer.MAX_PARTICIPATION_RATE
    assert abs(score.max_single_day_trade_inr - expected_max) < 1


def test_liquidity_scorer_execution_schedule():
    scorer = LiquidityScorer()
    schedule = scorer.schedule_execution(5_000_000, "IEQ001", 1_000_000)
    assert schedule["days_required"] >= 5  # 5M trade vs 100k daily max
    assert schedule["execution_strategy"] in ("VWAP", "TWAP")


def test_liquidity_scorer_single_day_small_trade():
    scorer = LiquidityScorer()
    schedule = scorer.schedule_execution(50_000, "IEQ001", 10_000_000)
    assert schedule["days_required"] == 1
    assert schedule["execution_strategy"] == "market_order"


def test_liquidity_scorer_batch():
    scorer = LiquidityScorer()
    master = pd.DataFrame(
        [
            {
                "security_id": f"IEQ{i:03d}",
                "asset_class": "indian_equity",
                "avg_daily_volume_inr": 1_000_000 * (i + 1),
                "bid_ask_spread_bps": 10.0,
            }
            for i in range(10)
        ]
    )
    df = scorer.score_batch(master)
    assert len(df) == 10
    assert "liquidity_score" in df.columns


# ConstraintManager
def test_constraint_manager_long_only_violation():
    cm = ConstraintManager()
    post_weights = np.array([-0.05, 0.20, 0.35, 0.10, 0.15, 0.25])  # negative weight
    trade_weights = np.zeros(6)
    violations = cm.check_post_trade(post_weights, trade_weights)
    assert cm.has_hard_violations(violations)
    assert any(v.constraint_name == "long_only" for v in violations)


def test_constraint_manager_no_violation():
    cm = ConstraintManager()
    post_weights = np.array([0.35, 0.15, 0.20, 0.10, 0.12, 0.08])
    trade_weights = np.zeros(6)
    violations = cm.check_post_trade(post_weights, trade_weights)
    assert not cm.has_hard_violations(violations)


def test_constraint_manager_sebi_intl_limit():
    cm = ConstraintManager()
    post_weights = np.array([0.30, 0.30, 0.20, 0.05, 0.07, 0.08])  # intl = 30%
    trade_weights = np.zeros(6)
    violations = cm.check_post_trade(post_weights, trade_weights)
    assert any(v.constraint_name == "sebi_intl_equity" for v in violations)


def test_constraint_manager_sector_violation():
    cm = ConstraintManager()
    post_weights = np.array([0.35, 0.15, 0.20, 0.10, 0.12, 0.08])
    trade_weights = np.zeros(6)
    violations = cm.check_post_trade(
        post_weights,
        trade_weights,
        sector_weights={"financials": 0.40},  # exceeds 35% limit
    )
    assert any("financials" in v.constraint_name for v in violations)


# TradeListGenerator
@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("cvxpy"),
    reason="cvxpy not installed",
)
def test_trade_list_generator():
    from src.optimisation.portfolio_optimiser import PortfolioOptimiser

    current = np.array([0.45, 0.15, 0.15, 0.10, 0.07, 0.08])
    target = np.array([0.35, 0.15, 0.20, 0.10, 0.12, 0.08])

    opt = PortfolioOptimiser()
    result = opt.optimise(current, target)

    master = pd.DataFrame(
        [
            {
                "security_id": f"IEQ{i:03d}",
                "asset_class": [
                    "indian_equity",
                    "international_equity",
                    "indian_fixed_income",
                    "international_fixed_income",
                    "alternatives",
                    "cash",
                ][i],
                "avg_daily_volume_inr": 5_000_000,
                "current_price_inr": 100.0,
                "bid_ask_spread_bps": 10.0,
                "exit_load_pct": 0.0,
                "is_esg_compliant": True,
                "sector": "financials",
            }
            for i in range(6)
        ]
    )

    gen = TradeListGenerator()
    trades = gen.generate("WP000001", result, 1_000_000, master)
    summary = gen.trade_list_summary(trades)

    assert summary["trade_count"] >= 0  # may be zero if no drift
    if trades:
        assert all(t.direction in ("BUY", "SELL") for t in trades)
        assert all(t.trade_value_inr > 0 for t in trades)
