"""End-to-end integration tests: drift detection → trigger → optimisation → explanation."""

import numpy as np
import pandas as pd
import pytest
from datetime import date
from unittest.mock import MagicMock


# ── Data layer ──────────────────────────────────────────────────────────────────
def test_data_generation_produces_50k_portfolios():
    from src.data.client_profile_generator import ClientProfileGenerator
    gen = ClientProfileGenerator(seed=42)
    profiles = gen.generate_all()
    assert len(profiles) == 50_000
    assert "risk_category" in profiles.columns
    assert "client_id" in profiles.columns


def test_market_data_generates_252_days():
    from src.data.market_data_simulator import MarketDataSimulator
    sim = MarketDataSimulator(seed=42)
    returns = sim.simulate_returns()
    assert len(returns) == 252
    assert returns.shape[1] == 6  # 6 asset classes


def test_portfolio_values_generated():
    from src.data.portfolio_generator import PortfolioGenerator
    from src.data.client_profile_generator import ClientProfileGenerator
    gen = PortfolioGenerator(seed=42)
    clients = ClientProfileGenerator(seed=42).generate_all()
    values = gen.generate_portfolio_values(clients)
    assert len(values) == 50_000
    assert (values > 0).all()


# ── Monitoring layer ─────────────────────────────────────────────────────────────
def test_drift_monitor_scan_performance():
    """50,000 portfolios should be scanned in <30 seconds."""
    import time
    from src.data.portfolio_generator import PortfolioGenerator
    from src.data.client_profile_generator import ClientProfileGenerator
    from src.monitoring.drift_monitor import DriftMonitor

    gen = PortfolioGenerator(seed=42)
    clients = ClientProfileGenerator(seed=42).generate_all()
    weights = gen.generate_current_allocations(clients)
    risk_cats = clients.set_index("client_id")["risk_category"]

    monitor = DriftMonitor()
    t0 = time.perf_counter()
    summary = monitor.run_scan(weights, risk_cats)
    elapsed = time.perf_counter() - t0

    assert elapsed < 30.0, f"Scan took {elapsed:.1f}s — target is <30s"
    assert summary["total_portfolios"] == 50_000


# ── Trigger layer ────────────────────────────────────────────────────────────────
def test_trigger_pipeline_for_drifted_portfolio():
    from src.triggers.trigger_consolidator import TriggerConsolidator
    from src.monitoring.drift_calculator import DriftResult, DriftSeverity

    drift = DriftResult(
        portfolio_id="WP000001",
        risk_category="balanced",
        current_weights=np.array([0.45, 0.15, 0.15, 0.08, 0.09, 0.08]),
        target_weights=np.array([0.35, 0.15, 0.20, 0.10, 0.12, 0.08]),
        abs_drift=np.array([0.10, 0.0, 0.05, 0.02, 0.03, 0.0]),
        max_drift=0.10,
        sum_abs_drift=0.20,
        rmsd=0.065,
        drift_band=0.030,
        severity=DriftSeverity.CRITICAL,
        breaching_asset_classes=["indian_equity"],
    )

    consolidator = TriggerConsolidator()
    ctx = {
        "WP000001": {
            "drift_result": drift,
            "date": date(2025, 4, 1),
            "sector_weights": {},
            "issuer_weights": {},
            "benchmark_drawdown_from_high": -0.05,
            "harvestable_loss_inr": 0,
            "sip_inflow_inr": 0,
            "uninvested_cash_inr": 0,
            "regulatory_events": [],
            "client_life_events": [],
            "factor_tilts": {},
            "pending_cash_flow_inr": 0,
        }
    }
    results = consolidator.evaluate_batch(["WP000001"], ctx)
    assert len(results) == 1
    assert results[0].portfolio_id == "WP000001"


# ── Optimisation layer ─────────────────────────────────────────────────────────
@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("cvxpy"),
    reason="cvxpy not installed",
)
def test_optimiser_constraint_satisfaction():
    from src.optimisation.portfolio_optimiser import PortfolioOptimiser
    current = np.array([0.45, 0.15, 0.15, 0.10, 0.07, 0.08])
    target = np.array([0.35, 0.15, 0.20, 0.10, 0.12, 0.08])
    optimiser = PortfolioOptimiser()
    result = optimiser.optimise(current, target, turnover_budget=0.20)

    assert result.post_trade_weights is not None
    assert abs(result.post_trade_weights.sum() - 1.0) < 0.01
    assert (result.post_trade_weights >= -1e-6).all()
    assert result.turnover <= 0.20 + 1e-3


# ── Explanation layer ──────────────────────────────────────────────────────────
def test_explanation_pipeline_with_mock_llm():
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text="Test rebalancing explanation.")]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_msg

    from src.explainability.explanation_generator import ExplanationGenerator
    gen = ExplanationGenerator(client=mock_client)
    meta = {
        "portfolio_id": "WP000001",
        "risk_category": "balanced",
        "trigger_type": "threshold_asset_class",
        "max_drift_pct": 5.0,
        "sum_abs_drift_pct": 10.0,
        "total_cost_inr": 3500,
        "tax_impact_inr": 0,
        "tracking_error_before": 4.0,
        "tracking_error_after": 0.8,
        "trade_summary": {},
    }
    outputs = gen.generate_all_tiers(meta)
    assert len(outputs) == 3
    for audience in ("client", "advisor", "compliance"):
        assert outputs[audience].narrative is not None


# ── Override layer ─────────────────────────────────────────────────────────────
def test_override_audit_trail():
    from src.override.override_capture import OverrideCapture
    capture = OverrideCapture()
    record = capture.capture(
        decision_id="DEC00000001",
        portfolio_id="WP000001",
        advisor_id="ADV001",
        original_recommendation={"action": "full_rebalance"},
        modified_recommendation={"action": "partial_rebalance"},
        reason_category="client_preference",
        reason_free_text="Client wants minimal activity",
    )
    assert record.override_id is not None
    assert capture.get_decision_override("DEC00000001") is not None


# ── Compliance layer ──────────────────────────────────────────────────────────
def test_bias_detector_runs_on_sample():
    from src.compliance.bias_detector import BiasDetector
    decision_log = [
        {"decision_metadata": {"risk_category": "balanced", "trigger_type": "threshold", "max_drift_pct": 4.5, "vix": 18.0}},
        {"decision_metadata": {"risk_category": "aggressive", "trigger_type": "calendar", "max_drift_pct": 5.0, "vix": 20.0}},
        {"decision_metadata": {"risk_category": "balanced", "trigger_type": "threshold", "max_drift_pct": 3.8, "vix": 19.0}},
    ]
    detector = BiasDetector()
    report = detector.full_bias_report(decision_log)
    assert "category_bias" in report
    assert "momentum_bias" in report
