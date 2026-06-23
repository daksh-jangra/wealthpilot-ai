"""Capture, store, and retrieve human override records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json
from pathlib import Path

OVERRIDE_REASON_CATEGORIES = [
    "client_preference",
    "tax_consideration",
    "liquidity_concern",
    "market_timing",
    "regulatory_concern",
    "model_disagreement",
    "other",
]


@dataclass
class OverrideRecord:
    override_id: str
    decision_id: str
    portfolio_id: str
    advisor_id: str
    timestamp: datetime
    original_recommendation: dict
    modified_recommendation: dict
    reason_category: str
    reason_free_text: str
    override_trade_list: Optional[list] = None
    outcome_tracked: bool = False
    outcome: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "override_id": self.override_id,
            "decision_id": self.decision_id,
            "portfolio_id": self.portfolio_id,
            "advisor_id": self.advisor_id,
            "timestamp": self.timestamp.isoformat(),
            "reason_category": self.reason_category,
            "reason_free_text": self.reason_free_text,
            "original_recommendation": self.original_recommendation,
            "modified_recommendation": self.modified_recommendation,
        }


class OverrideCapture:
    """
    API for capturing advisor overrides and maintaining the override audit trail.
    In production this would persist to a database; here we use in-memory + JSONL.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self._records: list[OverrideRecord] = []
        self._storage_path = storage_path

    def capture(
        self,
        decision_id: str,
        portfolio_id: str,
        advisor_id: str,
        original_recommendation: dict,
        modified_recommendation: dict,
        reason_category: str,
        reason_free_text: str,
        override_trade_list: Optional[list] = None,
    ) -> OverrideRecord:
        """Record a human override of an agent recommendation."""
        if reason_category not in OVERRIDE_REASON_CATEGORIES:
            raise ValueError(f"reason_category must be one of {OVERRIDE_REASON_CATEGORIES}")

        import uuid

        record = OverrideRecord(
            override_id=f"OVR{uuid.uuid4().hex[:8].upper()}",
            decision_id=decision_id,
            portfolio_id=portfolio_id,
            advisor_id=advisor_id,
            timestamp=datetime.utcnow(),
            original_recommendation=original_recommendation,
            modified_recommendation=modified_recommendation,
            reason_category=reason_category,
            reason_free_text=reason_free_text,
            override_trade_list=override_trade_list,
        )
        self._records.append(record)
        self._persist(record)
        return record

    def get_portfolio_overrides(self, portfolio_id: str) -> list[OverrideRecord]:
        return [r for r in self._records if r.portfolio_id == portfolio_id]

    def get_decision_override(self, decision_id: str) -> Optional[OverrideRecord]:
        for r in self._records:
            if r.decision_id == decision_id:
                return r
        return None

    def override_rate_by_category(self) -> dict[str, int]:
        from collections import Counter

        return dict(Counter(r.reason_category for r in self._records))

    def generate_override_quality_report(self) -> dict:
        """Quarterly report comparing override vs agent outcomes."""
        tracked = [r for r in self._records if r.outcome_tracked]
        return {
            "total_overrides": len(self._records),
            "tracked_outcomes": len(tracked),
            "override_rate_by_category": self.override_rate_by_category(),
            "report_generated": datetime.utcnow().isoformat(),
        }

    def _persist(self, record: OverrideRecord) -> None:
        if self._storage_path is None:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._storage_path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
