"""Generate escalation briefings for edge cases requiring human judgment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class EscalationBriefing:
    escalation_id: str
    portfolio_id: str
    decision_id: str
    situation_summary: str
    agent_analysis: str
    options: list[dict]
    recommended_action: str
    specific_questions: list[str]
    urgency: str
    created_at: str


class EscalationManager:
    """Generate structured escalation briefings for human review."""

    URGENCY_LEVELS = {
        "critical": "Immediate action required (same session)",
        "high": "Action required within 24 hours",
        "medium": "Action required within 3 trading sessions",
        "low": "Action required within 5 trading sessions",
    }

    def __init__(self):
        self._escalations: list[EscalationBriefing] = []

    def create_briefing(
        self,
        portfolio_id: str,
        decision_id: str,
        decision_metadata: dict,
        urgency: str = "high",
    ) -> EscalationBriefing:
        """Create a human-readable escalation briefing."""
        import uuid

        trigger = decision_metadata.get("trigger_type", "unknown")
        drift = decision_metadata.get("max_drift_pct", 0)
        violations = decision_metadata.get("constraint_checks", {}).get("details", [])

        situation = (
            f"Portfolio {portfolio_id} triggered by {trigger.replace('_', ' ')}. "
            f"Max drift: {drift:.1f}%. "
            f"{'Constraint violations detected.' if violations else 'No constraint violations.'}"
        )

        analysis = (
            f"The agent evaluated rebalancing and found that standard optimisation "
            f"{'could not satisfy all constraints simultaneously' if violations else 'completed successfully'}. "
            f"Human judgment is required to resolve the conflict between "
            f"{', '.join(v.get('name', 'unknown') for v in violations[:2]) if violations else 'competing objectives'}."
        )

        options = [
            {
                "id": "A",
                "description": "Proceed with agent's best-effort trade list despite constraint violations",
                "risk": "May breach hard limits temporarily",
            },
            {
                "id": "B",
                "description": "Execute partial rebalance focusing on most critical drift only",
                "risk": "Leaves portfolio partially unrebalanced",
            },
            {
                "id": "C",
                "description": "Hold position and monitor — schedule full review in 5 days",
                "risk": "Portfolio remains out of balance",
            },
        ]

        questions = [
            f"Should the international equity limit be temporarily relaxed for this client?",
            f"Is there a client-specific reason to delay rebalancing despite {drift:.1f}% drift?",
            f"Should the trade be split across multiple days to reduce market impact?",
        ]

        briefing = EscalationBriefing(
            escalation_id=f"ESC{uuid.uuid4().hex[:8].upper()}",
            portfolio_id=portfolio_id,
            decision_id=decision_id,
            situation_summary=situation,
            agent_analysis=analysis,
            options=options,
            recommended_action="Option B — partial rebalance to the band edge",
            specific_questions=questions,
            urgency=self.URGENCY_LEVELS.get(urgency, urgency),
            created_at=datetime.utcnow().isoformat(),
        )
        self._escalations.append(briefing)
        return briefing

    def get_open_escalations(self) -> list[EscalationBriefing]:
        return list(self._escalations)

    def resolve_escalation(self, escalation_id: str) -> None:
        self._escalations = [e for e in self._escalations if e.escalation_id != escalation_id]
