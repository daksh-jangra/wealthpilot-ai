"""Tests for override system: intervention classification, capture, kill switch."""

import pytest
from src.override.intervention_classifier import InterventionClassifier, InterventionLevel
from src.override.override_capture import OverrideCapture
from src.override.kill_switch import KillSwitch


@pytest.fixture
def classifier():
    return InterventionClassifier()


@pytest.fixture
def capture():
    return OverrideCapture()


@pytest.fixture
def kill_switch():
    return KillSwitch()


# Intervention Classifier Tests
def test_informational_low_impact_high_confidence(classifier):
    decision = classifier.classify(trade_impact_pct=3.0, decision_confidence=0.92)
    assert decision.level == InterventionLevel.INFORMATIONAL


def test_advisory_medium_impact(classifier):
    decision = classifier.classify(trade_impact_pct=10.0, decision_confidence=0.85)
    assert decision.level == InterventionLevel.ADVISORY
    assert decision.waiting_period_hours == 24


def test_approval_required_high_impact(classifier):
    decision = classifier.classify(trade_impact_pct=20.0, decision_confidence=0.90)
    assert decision.level == InterventionLevel.APPROVAL_REQUIRED


def test_approval_required_low_confidence(classifier):
    decision = classifier.classify(trade_impact_pct=5.0, decision_confidence=0.60)
    assert decision.level == InterventionLevel.APPROVAL_REQUIRED


def test_escalation_compliance_exception(classifier):
    decision = classifier.classify(
        trade_impact_pct=3.0,
        decision_confidence=0.95,
        has_compliance_exception=True,
    )
    assert decision.level == InterventionLevel.ESCALATION


def test_escalation_client_restriction(classifier):
    decision = classifier.classify(
        trade_impact_pct=3.0,
        decision_confidence=0.90,
        has_client_restriction_override=True,
    )
    assert decision.level == InterventionLevel.ESCALATION


def test_escalation_novel_trigger(classifier):
    decision = classifier.classify(
        trade_impact_pct=5.0,
        decision_confidence=0.85,
        is_novel_trigger=True,
    )
    assert decision.level == InterventionLevel.ESCALATION


def test_confidence_computation(classifier):
    conf_optimal = classifier.compute_confidence("optimal", 0.05)
    conf_infeasible = classifier.compute_confidence("infeasible", 0.0)
    assert conf_optimal > conf_infeasible
    assert 0 <= conf_optimal <= 1
    assert 0 <= conf_infeasible <= 1


# Override Capture Tests
def test_capture_records_override(capture):
    record = capture.capture(
        decision_id="DEC00000001",
        portfolio_id="WP000001",
        advisor_id="ADV001",
        original_recommendation={"action": "rebalance", "turnover": 0.12},
        modified_recommendation={"action": "partial_rebalance", "turnover": 0.06},
        reason_category="client_preference",
        reason_free_text="Client requested minimal trading this month",
    )
    assert record.override_id.startswith("OVR")
    assert record.portfolio_id == "WP000001"


def test_capture_invalid_reason_raises(capture):
    with pytest.raises(ValueError):
        capture.capture(
            decision_id="DEC00000002",
            portfolio_id="WP000001",
            advisor_id="ADV001",
            original_recommendation={},
            modified_recommendation={},
            reason_category="invalid_reason",
            reason_free_text="",
        )


def test_get_portfolio_overrides(capture):
    capture.capture("DEC001", "WP000001", "ADV001", {}, {}, "client_preference", "test")
    capture.capture("DEC002", "WP000001", "ADV001", {}, {}, "tax_consideration", "test2")
    capture.capture("DEC003", "WP000002", "ADV002", {}, {}, "other", "test3")

    p1_overrides = capture.get_portfolio_overrides("WP000001")
    assert len(p1_overrides) == 2

    p2_overrides = capture.get_portfolio_overrides("WP000002")
    assert len(p2_overrides) == 1


def test_override_rate_by_category(capture):
    for _ in range(3):
        capture.capture("DEC_A", "WP0001", "ADV1", {}, {}, "client_preference", "")
    capture.capture("DEC_B", "WP0002", "ADV1", {}, {}, "tax_consideration", "")
    rates = capture.override_rate_by_category()
    assert rates.get("client_preference", 0) >= 3


# Kill Switch Tests
def test_kill_switch_initially_inactive(kill_switch):
    assert not kill_switch.is_active


def test_kill_switch_activate(kill_switch):
    event = kill_switch.activate("Manual test", triggered_by="manual")
    assert kill_switch.is_active
    assert event.action == "activated"


def test_kill_switch_deactivate(kill_switch):
    kill_switch.activate("Test")
    event = kill_switch.deactivate("All clear", triggered_by="manual")
    assert not kill_switch.is_active
    assert event.action == "deactivated"


def test_kill_switch_auto_vix_trigger(kill_switch):
    activated = kill_switch.check_auto_triggers(vix=45.0, error_count=0, processed=100)
    assert activated
    assert kill_switch.is_active


def test_kill_switch_no_auto_trigger_normal_vix(kill_switch):
    activated = kill_switch.check_auto_triggers(vix=20.0, error_count=0, processed=100)
    assert not activated


def test_kill_switch_auto_error_rate_trigger(kill_switch):
    kill_switch._processed_count = 100
    kill_switch._error_count = 2  # 2% > 1% threshold
    activated = kill_switch.check_auto_triggers(vix=18.0, error_count=2, processed=100)
    assert activated


def test_kill_switch_event_log(kill_switch):
    kill_switch.activate("Test 1")
    kill_switch.deactivate("Test 2")
    log = kill_switch.get_event_log()
    assert len(log) == 2
    assert log[0]["action"] == "activated"
    assert log[1]["action"] == "deactivated"
