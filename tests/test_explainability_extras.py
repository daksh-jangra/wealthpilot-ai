"""Tests for SHAP, LIME, counterfactual, advisor, and compliance explainers."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.explainability.counterfactual_generator import CounterfactualGenerator
from src.explainability.advisor_explainer import AdvisorExplainer
from src.explainability.compliance_explainer import ComplianceExplainer
from src.explainability.explanation_generator import ExplanationGenerator, ExplanationOutput

SAMPLE_FEATURES = {
    "max_drift_pct": 6.5,
    "sum_abs_drift_pct": 12.0,
    "days_since_last_rebalance": 90,
    "vix": 22.0,
    "risk_score": 3.0,
    "ltcg_lot_fraction": 0.6,
    "sector_concentration_max": 0.25,
    "portfolio_value_log": 14.5,
}

SAMPLE_FEATURE_ARRAY = np.array([6.5, 12.0, 90, 22.0, 3.0, 0.6, 0.25, 14.5])
SAMPLE_SHAP_ARRAY = np.array([0.15, 0.08, 0.03, 0.12, -0.02, 0.01, -0.05, 0.02])


# ── CounterfactualGenerator ──────────────────────────────────────────────────


def test_counterfactual_rebalance_to_hold():
    gen = CounterfactualGenerator()
    cf = gen.generate(
        "WP000001",
        SAMPLE_FEATURE_ARRAY,
        SAMPLE_SHAP_ARRAY,
        decision="rebalance",
    )
    assert cf.original_decision == "rebalance"
    assert cf.flipped_decision == "hold"
    assert isinstance(cf.natural_language, str)
    assert len(cf.natural_language) > 20


def test_counterfactual_hold_to_rebalance():
    gen = CounterfactualGenerator()
    # Negative SHAP values → hold decision
    neg_shap = -SAMPLE_SHAP_ARRAY
    cf = gen.generate("WP000002", SAMPLE_FEATURE_ARRAY, neg_shap, decision="hold")
    assert cf.original_decision == "hold"
    assert cf.flipped_decision == "rebalance"


def test_counterfactual_key_feature_is_top_shap():
    gen = CounterfactualGenerator()
    cf = gen.generate("WP000003", SAMPLE_FEATURE_ARRAY, SAMPLE_SHAP_ARRAY, decision="rebalance")
    # Highest |SHAP| is index 0 (max_drift_pct = 0.15)
    assert cf.key_feature == "max_drift_pct"


def test_counterfactual_natural_language_contains_feature():
    gen = CounterfactualGenerator()
    cf = gen.generate("WP000004", SAMPLE_FEATURE_ARRAY, SAMPLE_SHAP_ARRAY, decision="rebalance")
    assert (
        "equity allocation drift" in cf.natural_language or "drift" in cf.natural_language.lower()
    )


def test_counterfactual_batch():
    gen = CounterfactualGenerator()
    portfolio_ids = ["WP000001", "WP000002"]
    feature_arrays = [SAMPLE_FEATURE_ARRAY, SAMPLE_FEATURE_ARRAY]
    shap_arrays = [SAMPLE_SHAP_ARRAY, SAMPLE_SHAP_ARRAY]
    decisions = ["rebalance", "hold"]
    results = gen.generate_batch(portfolio_ids, feature_arrays, shap_arrays, decisions)
    assert len(results) == 2
    assert results[0].original_decision == "rebalance"
    assert results[1].original_decision == "hold"


def test_counterfactual_days_since_rebalance_unit():
    gen = CounterfactualGenerator()
    # Make days_since_last_rebalance the top SHAP feature
    shap = np.zeros(8)
    shap[2] = 0.50  # days_since_last_rebalance
    cf = gen.generate("WP000005", SAMPLE_FEATURE_ARRAY, shap, decision="rebalance")
    assert cf.key_feature == "days_since_last_rebalance"
    assert "days" in cf.natural_language


# ── AdvisorExplainer ─────────────────────────────────────────────────────────


def _mock_generator() -> ExplanationGenerator:
    gen = MagicMock(spec=ExplanationGenerator)
    gen.generate.return_value = ExplanationOutput(
        audience="advisor",
        narrative="Tracking error of 3.5% detected. Rebalancing reduces risk.",
        trigger_summary="Threshold drift trigger",
        key_numbers={"max_drift_pct": 6.5, "total_cost_inr": 3500},
        word_count=11,
    )
    return gen


def test_advisor_explainer_returns_output():
    gen = _mock_generator()
    explainer = AdvisorExplainer(gen)
    result = explainer.explain({"max_drift_pct": 6.5, "risk_category": "balanced"})
    assert isinstance(result, ExplanationOutput)
    assert result.word_count > 0


def test_advisor_explainer_dashboard_widget():
    gen = _mock_generator()
    explainer = AdvisorExplainer(gen)
    output = explainer.explain({})
    widget = explainer.format_dashboard_widget(output)
    assert "trigger" in widget
    assert "narrative" in widget
    assert "key_metrics" in widget
    assert "word_count" in widget


# ── ComplianceExplainer ───────────────────────────────────────────────────────


def test_compliance_explainer_generates_output():
    gen = _mock_generator()
    with patch("src.explainability.compliance_explainer.SHAPIntegration") as mock_shap_cls:
        mock_shap = MagicMock()
        mock_shap.explain.return_value = MagicMock(
            top_features=[{"feature": "max_drift_pct", "shap_value": 0.15}],
            counterfactual="If drift were 4.0 instead of 6.5, the agent would hold.",
        )
        mock_shap_cls.return_value = mock_shap
        explainer = ComplianceExplainer(gen, shap=mock_shap)
        result = explainer.explain({"portfolio_id": "WP000001", "max_drift_pct": 6.5, "vix": 22.0})
        assert isinstance(result, ExplanationOutput)


def test_compliance_explainer_full_audit_record():
    gen = _mock_generator()
    with patch("src.explainability.compliance_explainer.SHAPIntegration") as mock_shap_cls:
        mock_shap = MagicMock()
        mock_shap.explain.return_value = MagicMock(
            top_features=[],
            counterfactual=None,
        )
        mock_shap_cls.return_value = mock_shap
        explainer = ComplianceExplainer(gen, shap=mock_shap)
        output = explainer.explain({"portfolio_id": "WP000001", "max_drift_pct": 5.0})
        audit = explainer.generate_full_audit_record(
            {"portfolio_id": "WP000001", "decision_id": "DEC00000001", "trigger_type": "threshold"},
            output,
            trade_list=[{"trade_id": "TRD001"}],
        )
        assert audit["portfolio_id"] == "WP000001"
        assert "audit_timestamp" in audit
        assert len(audit["trade_list"]) == 1


def test_compliance_explainer_sebi_ref_regulatory_trigger():
    gen = _mock_generator()
    with patch("src.explainability.compliance_explainer.SHAPIntegration"):
        explainer = ComplianceExplainer(gen)
        # Access the private method directly
        refs = explainer._get_sebi_refs(
            {"trigger_type": "regulatory_change", "constraint_checks": {}}
        )
        assert any("SEBI" in r for r in refs)


# ── SHAPIntegration (integration test, skipped if shap not installed) ─────────


def _shap_available() -> bool:
    try:
        from src.explainability.shap_integration import SHAP_AVAILABLE

        return SHAP_AVAILABLE
    except Exception:
        return False


def _lime_available() -> bool:
    try:
        from src.explainability.lime_integration import LIME_AVAILABLE
        from src.explainability.shap_integration import SHAP_AVAILABLE

        return LIME_AVAILABLE and SHAP_AVAILABLE
    except Exception:
        return False


@pytest.mark.skipif(not _shap_available(), reason="shap/xgboost not available")
def test_shap_integration_explain():
    from src.explainability.shap_integration import SHAPIntegration

    shap_int = SHAPIntegration()
    exp = shap_int.explain("WP000001", SAMPLE_FEATURES)
    assert exp.portfolio_id == "WP000001"
    assert len(exp.top_features) > 0
    assert len(exp.shap_values) == 8
    assert isinstance(exp.prediction, float)
    assert isinstance(exp.counterfactual, str)


@pytest.mark.skipif(not _shap_available(), reason="shap/xgboost not available")
def test_shap_format_compliance():
    from src.explainability.shap_integration import SHAPIntegration

    shap_int = SHAPIntegration()
    exp = shap_int.explain("WP000001", SAMPLE_FEATURES)
    text = shap_int.format_for_compliance(exp)
    assert "SHAP FEATURE ATTRIBUTION" in text
    assert "Baseline" in text


@pytest.mark.skipif(not _shap_available(), reason="shap/xgboost not available")
def test_surrogate_model_trainer_generates_data():
    from src.explainability.shap_integration import SurrogateModelTrainer, SHAP_AVAILABLE

    if not SHAP_AVAILABLE:
        pytest.skip("xgboost not available")
    trainer = SurrogateModelTrainer()
    X, y = trainer.generate_training_data(n_samples=500)
    assert X.shape == (500, 8)
    assert len(y) == 500
    assert set(y.unique()).issubset({0, 1})


# ── LIMEIntegration ──────────────────────────────────────────────────────────


@pytest.mark.skipif(not _lime_available(), reason="lime not available")
def test_lime_integration_explain():
    from src.explainability.lime_integration import LIMEIntegration

    lime_int = LIMEIntegration()
    exp = lime_int.explain("WP000001", SAMPLE_FEATURES, num_features=5)
    assert exp.portfolio_id == "WP000001"
    assert isinstance(exp.prediction, float)
    assert isinstance(exp.explanation_text, str)
    assert "Decision:" in exp.explanation_text


@pytest.mark.skipif(not _lime_available(), reason="lime not available")
def test_lime_integration_batch():
    from src.explainability.lime_integration import LIMEIntegration

    lime_int = LIMEIntegration()
    results = lime_int.explain_batch(["WP000001", "WP000002"], [SAMPLE_FEATURES, SAMPLE_FEATURES])
    assert len(results) == 2
