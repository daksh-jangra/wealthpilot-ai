"""Classify rebalancing decisions into intervention levels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class InterventionLevel(str, Enum):
    INFORMATIONAL = "informational"       # post-execution notification
    ADVISORY = "advisory"                 # proceeds unless advisor objects within 4-24h
    APPROVAL_REQUIRED = "approval_required"  # waits for explicit approval
    ESCALATION = "escalation"             # needs human intervention


@dataclass
class InterventionDecision:
    level: InterventionLevel
    trade_impact_pct: float
    decision_confidence: float
    reason: str
    waiting_period_hours: Optional[int] = None


class InterventionClassifier:
    """
    Assign each rebalancing decision to an intervention level based on
    trade impact and agent decision confidence.
    """

    INFORMATIONAL_MAX_TRADE_PCT = 5.0
    ADVISORY_MAX_TRADE_PCT = 15.0
    MIN_CONFIDENCE_ADVISORY = 0.75

    def classify(
        self,
        trade_impact_pct: float,
        decision_confidence: float,
        has_compliance_exception: bool = False,
        has_client_restriction_override: bool = False,
        is_novel_trigger: bool = False,
    ) -> InterventionDecision:
        """
        Determine the intervention level.

        Args:
            trade_impact_pct: total trade value as % of portfolio
            decision_confidence: agent's self-assessed confidence (0-1)
            has_compliance_exception: True if any compliance rule was violated
            has_client_restriction_override: True if restricted security is involved
            is_novel_trigger: True if trigger type not seen before
        """
        # Always escalate edge cases
        if has_compliance_exception or has_client_restriction_override or is_novel_trigger:
            return InterventionDecision(
                level=InterventionLevel.ESCALATION,
                trade_impact_pct=trade_impact_pct,
                decision_confidence=decision_confidence,
                reason=(
                    "Compliance exception" if has_compliance_exception else
                    "Client restriction override" if has_client_restriction_override else
                    "Novel trigger type"
                ),
            )

        # High impact + low confidence → approval required
        if trade_impact_pct > self.ADVISORY_MAX_TRADE_PCT or decision_confidence < self.MIN_CONFIDENCE_ADVISORY:
            return InterventionDecision(
                level=InterventionLevel.APPROVAL_REQUIRED,
                trade_impact_pct=trade_impact_pct,
                decision_confidence=decision_confidence,
                reason=f"Trade impact {trade_impact_pct:.1f}% or confidence {decision_confidence:.2f} below threshold",
            )

        # Medium impact → advisory (24h window)
        if trade_impact_pct > self.INFORMATIONAL_MAX_TRADE_PCT:
            return InterventionDecision(
                level=InterventionLevel.ADVISORY,
                trade_impact_pct=trade_impact_pct,
                decision_confidence=decision_confidence,
                reason=f"Trade impact {trade_impact_pct:.1f}% in advisory range",
                waiting_period_hours=24,
            )

        # Low impact + high confidence → informational
        return InterventionDecision(
            level=InterventionLevel.INFORMATIONAL,
            trade_impact_pct=trade_impact_pct,
            decision_confidence=decision_confidence,
            reason=f"Low-impact ({trade_impact_pct:.1f}%) routine rebalance, confidence {decision_confidence:.2f}",
        )

    def compute_confidence(self, opt_result_status: str, tracking_error_reduction: float) -> float:
        """Derive confidence from optimiser quality and tracking error improvement."""
        if opt_result_status == "optimal":
            base = 0.90
        elif opt_result_status == "optimal_inaccurate":
            base = 0.75
        else:
            base = 0.50
        # Boost for large TE reduction
        te_bonus = min(0.10, tracking_error_reduction * 0.5)
        return min(1.0, base + te_bonus)
