"""Tests for EscalationManager."""

import pytest
from src.override.escalation_manager import EscalationManager, EscalationBriefing

SAMPLE_METADATA = {
    "trigger_type": "threshold_asset_class",
    "max_drift_pct": 7.5,
    "constraint_checks": {
        "details": [
            {"name": "sebi_intl_equity", "message": "International equity exceeds 25%"},
        ]
    },
    "risk_category": "balanced",
}


def test_escalation_manager_creates_briefing():
    manager = EscalationManager()
    briefing = manager.create_briefing("WP000001", "DEC00000001", SAMPLE_METADATA)
    assert isinstance(briefing, EscalationBriefing)
    assert briefing.portfolio_id == "WP000001"
    assert briefing.decision_id == "DEC00000001"


def test_escalation_briefing_has_situation():
    manager = EscalationManager()
    briefing = manager.create_briefing("WP000001", "DEC00000001", SAMPLE_METADATA)
    assert "WP000001" in briefing.situation_summary
    assert "7.5" in briefing.situation_summary


def test_escalation_briefing_has_three_options():
    manager = EscalationManager()
    briefing = manager.create_briefing("WP000001", "DEC00000001", SAMPLE_METADATA)
    assert len(briefing.options) == 3
    assert all("description" in opt for opt in briefing.options)


def test_escalation_briefing_has_questions():
    manager = EscalationManager()
    briefing = manager.create_briefing("WP000001", "DEC00000001", SAMPLE_METADATA)
    assert len(briefing.specific_questions) >= 2


def test_escalation_briefing_urgency_levels():
    manager = EscalationManager()
    for level in ("critical", "high", "medium", "low"):
        b = manager.create_briefing("WP000001", f"DEC_{level}", {}, urgency=level)
        assert b.urgency == EscalationManager.URGENCY_LEVELS[level]


def test_escalation_briefing_unique_id():
    manager = EscalationManager()
    b1 = manager.create_briefing("WP000001", "DEC001", {})
    b2 = manager.create_briefing("WP000002", "DEC002", {})
    assert b1.escalation_id != b2.escalation_id
    assert b1.escalation_id.startswith("ESC")


def test_escalation_get_open_escalations():
    manager = EscalationManager()
    manager.create_briefing("WP000001", "DEC001", {})
    manager.create_briefing("WP000002", "DEC002", {})
    open_list = manager.get_open_escalations()
    assert len(open_list) == 2


def test_escalation_resolve_removes_briefing():
    manager = EscalationManager()
    b = manager.create_briefing("WP000001", "DEC001", {})
    manager.resolve_escalation(b.escalation_id)
    open_list = manager.get_open_escalations()
    assert all(e.escalation_id != b.escalation_id for e in open_list)


def test_escalation_resolve_nonexistent_no_error():
    manager = EscalationManager()
    manager.create_briefing("WP000001", "DEC001", {})
    manager.resolve_escalation("ESCNONEXISTENT")  # Should not raise
    assert len(manager.get_open_escalations()) == 1


def test_escalation_no_violations_message():
    manager = EscalationManager()
    b = manager.create_briefing(
        "WP000001", "DEC001", {"max_drift_pct": 5.0, "constraint_checks": {"details": []}}
    )
    assert "No constraint violations" in b.situation_summary


def test_escalation_analysis_contains_judgment():
    manager = EscalationManager()
    b = manager.create_briefing("WP000001", "DEC001", SAMPLE_METADATA)
    assert "Human judgment" in b.agent_analysis


def test_escalation_recommended_action_set():
    manager = EscalationManager()
    b = manager.create_briefing("WP000001", "DEC001", SAMPLE_METADATA)
    assert len(b.recommended_action) > 5


def test_escalation_created_at_is_string():
    manager = EscalationManager()
    b = manager.create_briefing("WP000001", "DEC001", {})
    assert isinstance(b.created_at, str)
    assert "T" in b.created_at  # ISO format


def test_escalation_batch_and_resolve_all():
    manager = EscalationManager()
    briefings = [manager.create_briefing(f"WP{i:06d}", f"DEC{i:06d}", {}) for i in range(5)]
    for b in briefings:
        manager.resolve_escalation(b.escalation_id)
    assert len(manager.get_open_escalations()) == 0
