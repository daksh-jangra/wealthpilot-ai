"""Tests for ExplanationGenerator — template selection, quality checks, mocked LLM."""

import pytest
from unittest.mock import MagicMock, patch

from src.explainability.explanation_generator import ExplanationGenerator, ExplanationOutput
from src.explainability.client_explainer import ClientExplainer
from src.compliance.explainability_scorecard import ExplainabilityScorecard

SAMPLE_METADATA = {
    "portfolio_id": "WP000001",
    "risk_category": "balanced",
    "trigger_type": "threshold_asset_class",
    "max_drift_pct": 4.5,
    "sum_abs_drift_pct": 9.0,
    "total_cost_inr": 3500,
    "tax_impact_inr": 0,
    "tracking_error_before": 4.2,
    "tracking_error_after": 0.8,
    "trade_summary": {"trade_count": 3, "total_value_inr": 85000},
    "decision_id": "DEC00000001",
    "model_version": "1.0.0",
    "constraint_checks": {"hard": 0, "soft": 1, "details": []},
}


@pytest.fixture
def mock_anthropic():
    """Mock Anthropic client to avoid API calls in tests."""
    mock_msg = MagicMock()
    mock_msg.content = [
        MagicMock(
            text="Your portfolio has drifted 4.5% from target. We are rebalancing to restore your risk level. Cost: INR 3,500."
        )
    ]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_msg
    return mock_client


@pytest.fixture
def generator(mock_anthropic):
    return ExplanationGenerator(model="claude-sonnet-4-6", client=mock_anthropic)


def test_generate_client_explanation(generator):
    output = generator.generate("client", SAMPLE_METADATA)
    assert isinstance(output, ExplanationOutput)
    assert output.audience == "client"
    assert len(output.narrative) > 0
    assert output.word_count > 0


def test_generate_advisor_explanation(generator):
    output = generator.generate("advisor", SAMPLE_METADATA)
    assert output.audience == "advisor"
    assert output.narrative is not None


def test_generate_compliance_explanation(generator):
    output = generator.generate("compliance", SAMPLE_METADATA)
    assert output.audience == "compliance"


def test_generate_all_tiers(generator):
    outputs = generator.generate_all_tiers(SAMPLE_METADATA)
    assert set(outputs.keys()) == {"client", "advisor", "compliance"}
    for audience, output in outputs.items():
        assert output.audience == audience
        assert output.narrative is not None


def test_fallback_on_api_error():
    """Generator should return fallback narrative when API fails."""
    broken_client = MagicMock()
    broken_client.messages.create.side_effect = Exception("API Error")
    gen = ExplanationGenerator(client=broken_client)
    output = gen.generate("client", SAMPLE_METADATA)
    assert output.narrative is not None
    assert len(output.narrative) > 10


def test_trigger_summary_extraction(generator):
    output = generator.generate("client", SAMPLE_METADATA)
    assert "4.5" in output.trigger_summary or "threshold" in output.trigger_summary.lower()


def test_key_numbers_extracted(generator):
    output = generator.generate("client", SAMPLE_METADATA)
    assert "max_drift_pct" in output.key_numbers
    assert output.key_numbers["max_drift_pct"] == 4.5
    assert output.key_numbers["total_cost_inr"] == 3500


def test_client_explainer_formats_email(generator):
    explainer = ClientExplainer(generator)
    output = explainer.explain(SAMPLE_METADATA)
    email = explainer.format_email(output, client_name="Rajesh Kumar")
    assert "Rajesh Kumar" in email
    assert "WealthPilot AI" in email


def test_explainability_scorecard():
    scorecard = ExplainabilityScorecard()
    narrative = (
        "Your portfolio has drifted 4.5% from its target due to recent market movements. "
        "We are rebalancing it back to your chosen risk level. "
        "This will cost approximately INR 3,500. "
        "Your portfolio will be back on track."
    )
    score = scorecard.score(
        explanation_id="TEST001_client",
        audience="client",
        narrative=narrative,
        metadata=SAMPLE_METADATA,
    )
    assert 0.0 <= score.overall_score <= 1.0
    assert isinstance(score.passed, bool)


def test_scorecard_compliance_requires_sebi_keywords():
    scorecard = ExplainabilityScorecard()
    narrative = "This is a regulatory compliance record. The decision ID is 001. Constraint check passed. Trigger: threshold. Audit trail complete. SEBI regulations followed."
    score = scorecard.score("TEST_COMP", "compliance", narrative, SAMPLE_METADATA)
    assert score.regulatory_score > 0.5


def test_word_count_tracking(generator):
    output = generator.generate("client", SAMPLE_METADATA)
    expected_words = len(output.narrative.split())
    assert output.word_count == expected_words
