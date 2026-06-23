"""Tests for compliance layer: auditor, regulatory reporter, bias detector, scorecard."""

import numpy as np
import pandas as pd
import pytest

from src.compliance.compliance_auditor import ComplianceAuditor
from src.compliance.regulatory_reporter import RegulatoryReporter
from src.compliance.bias_detector import BiasDetector
from src.compliance.explainability_scorecard import ExplainabilityScorecard


SAMPLE_DECISIONS = [
    {
        "decision_id": f"DEC{i:08d}",
        "decision_metadata": {
            "portfolio_id": f"WP{i:06d}",
            "risk_category": ["balanced", "aggressive", "conservative"][i % 3],
            "trigger_type": ["threshold_asset_class", "calendar_quarterly", "event_market_crash"][i % 3],
            "max_drift_pct": 4.5 + i * 0.1,
            "sum_abs_drift_pct": 9.0,
            "total_cost_inr": 3500,
            "tax_impact_inr": 0,
            "vix": 18.0 + i,
            "constraint_checks": {"hard": 0, "soft": 1, "details": []},
            "timestamp": "2025-03-14T09:23:45Z",
        },
        "explanations": {
            "client": {"narrative": "Your portfolio has drifted 4.5% from target. We are rebalancing. Cost: INR 3,500."},
            "advisor": {"narrative": "Drift 4.5% detected. Tracking error reduced. Trade cost 3,500. Cost analysis complete."},
            "compliance": {"narrative": "Decision audit: constraint check passed. Trigger threshold detected. SEBI rules followed. Model v1.0."},
        },
        "trades": [
            {"trade_id": f"TRD{i:08d}", "security_id": "IEQ001", "asset_class": "indian_equity",
             "direction": "BUY", "quantity": 100, "trade_value_inr": 10000, "estimated_cost_inr": 150}
        ],
        "override_history": [],
    }
    for i in range(30)
]


# ComplianceAuditor
def test_compliance_auditor_stratified_sample():
    auditor = ComplianceAuditor(seed=42)
    sample = auditor.stratified_sample(SAMPLE_DECISIONS, n=20)
    assert len(sample) <= 20


def test_compliance_auditor_explanation_quality():
    auditor = ComplianceAuditor()
    narrative = "Your portfolio has drifted 4.5% from target. We are rebalancing. Cost: INR 3,500."
    quality = auditor.evaluate_explanation_quality(narrative, "client")
    assert "word_count" in quality
    assert "completeness_score" in quality
    assert isinstance(quality["readability_pass"], bool)


def test_compliance_auditor_detects_category_bias():
    auditor = ComplianceAuditor()
    # All decisions for one category — detect_systematic_bias expects flat dicts
    biased_log = [
        {"risk_category": "balanced", "trigger_type": "threshold"}
        for _ in range(20)
    ]
    bias = auditor.detect_systematic_bias(biased_log)
    # One category dominates — should appear in bias report
    assert "balanced" in bias


def test_compliance_auditor_full_report():
    auditor = ComplianceAuditor(seed=42)
    report = auditor.generate_audit_report(SAMPLE_DECISIONS, quarter="Q1")
    assert "total_decisions" in report
    assert "explanation_quality" in report
    assert "bias_analysis" in report
    assert isinstance(report["audit_passed"], bool)


# RegulatoryReporter
def test_regulatory_reporter_transaction_report():
    reporter = RegulatoryReporter()
    df = reporter.generate_transaction_report(SAMPLE_DECISIONS)
    assert not df.empty
    assert "trade_id" in df.columns
    assert "decision_rationale" in df.columns


def test_regulatory_reporter_suitability():
    reporter = RegulatoryReporter()
    records = reporter.generate_suitability_report(SAMPLE_DECISIONS[:5])
    assert len(records) == 5
    assert all("sebi_compliant" in r for r in records)


def test_regulatory_reporter_audit_trail():
    reporter = RegulatoryReporter()
    trail = reporter.generate_algo_audit_trail(SAMPLE_DECISIONS[0])
    assert "ALGORITHMIC TRADING AUDIT TRAIL" in trail
    assert "Trigger" in trail


def test_regulatory_reporter_exception_report():
    reporter = RegulatoryReporter()
    # Add decision with violation
    decisions_with_violation = SAMPLE_DECISIONS[:5].copy()
    decisions_with_violation[0] = {
        **SAMPLE_DECISIONS[0],
        "decision_metadata": {**SAMPLE_DECISIONS[0]["decision_metadata"], "constraint_checks": {"hard": 1, "soft": 0, "details": []}},
        "override_history": [],
    }
    df = reporter.generate_exception_report(decisions_with_violation)
    assert not df.empty


# BiasDetector
def test_bias_detector_category_bias():
    detector = BiasDetector()
    # detect_category_bias flattens the list into a DataFrame directly
    log = [{"risk_category": "balanced", "trigger_type": "t", "max_drift_pct": 4.0, "vix": 18.0}
           for _ in range(20)]
    result = detector.detect_category_bias(log)
    assert "chi2_statistic" in result
    assert "significant_bias" in result


def test_bias_detector_security_bias():
    detector = BiasDetector()
    trades = [{"security_id": "IEQ001", "direction": "BUY"} for _ in range(15)]
    trades += [{"security_id": "IEQ002", "direction": "BUY"} for _ in range(5)]
    result = detector.detect_security_bias(trades)
    assert "top_5_securities" in result
    assert "concentration_flag" in result


def test_bias_detector_momentum_bias():
    detector = BiasDetector()
    log = [
        {"decision_metadata": {"risk_category": "balanced", "trigger_type": "t",
                                "max_drift_pct": 3.0 + i * 0.5, "vix": 18.0 + i * 2}}
        for i in range(15)
    ]
    result = detector.detect_momentum_bias(log)
    assert "drift_vix_correlation" in result


def test_bias_detector_insufficient_data():
    detector = BiasDetector()
    result = detector.detect_momentum_bias([{"decision_metadata": {"max_drift_pct": 4.0, "vix": 18.0}}])
    assert "error" in result


# ExplainabilityScorecard
def test_scorecard_client_pass():
    scorecard = ExplainabilityScorecard()
    narrative = (
        "Your portfolio has drifted 4.5% from its target due to market changes. "
        "We are rebalancing to restore your chosen risk level. "
        "This will cost INR 3,500. Your portfolio will be back on track."
    )
    score = scorecard.score("TEST001", "client", narrative, {"max_drift_pct": 4.5, "total_cost_inr": 3500})
    assert score.overall_score > 0.5


def test_scorecard_compliance_requires_keywords():
    scorecard = ExplainabilityScorecard()
    narrative = "Decision audit log. Constraint check: passed. SEBI regulations. Trigger: threshold. Model v1.0."
    score = scorecard.score("TEST002", "compliance", narrative, {"max_drift_pct": 4.0, "total_cost_inr": 3000})
    assert score.regulatory_score > 0.5


def test_scorecard_batch():
    scorecard = ExplainabilityScorecard()
    scores = scorecard.batch_score(SAMPLE_DECISIONS[:5])
    assert len(scores) == 15  # 5 decisions × 3 audiences
    assert all(0 <= s.overall_score <= 1 for s in scores)


def test_scorecard_empty_narrative():
    scorecard = ExplainabilityScorecard()
    score = scorecard.score("TEST003", "client", "", {"max_drift_pct": 4.0, "total_cost_inr": 3000})
    assert score.overall_score < 0.5
    assert not score.passed
