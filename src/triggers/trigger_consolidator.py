"""Consolidate multiple simultaneous triggers into a single rebalancing event."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.triggers.trigger_evaluator import (
    TriggerEvent,
    TriggerPriority,
    PRIORITY_ORDER,
)
from src.triggers.threshold_trigger import (
    ThresholdTrigger,
    ConcentrationTrigger,
    FactorExposureTrigger,
)
from src.triggers.calendar_trigger import (
    MonthlyCalendarTrigger,
    QuarterlyCalendarTrigger,
    AnnualCalendarTrigger,
)
from src.triggers.event_trigger import (
    MarketCrashTrigger,
    RegulatoryChangeTrigger,
    ClientLifeEventTrigger,
    TaxHarvestingTrigger,
    CashFlowTrigger,
)


@dataclass
class ConsolidatedTrigger:
    portfolio_id: str
    primary_trigger: TriggerEvent
    all_triggers: list[TriggerEvent]
    consolidated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def priority(self) -> TriggerPriority:
        return self.primary_trigger.priority

    @property
    def response_timeline(self) -> str:
        return self.primary_trigger.response_timeline

    def to_dict(self) -> dict:
        return {
            "portfolio_id": self.portfolio_id,
            "primary_trigger": self.primary_trigger.to_dict(),
            "contributing_triggers": [t.trigger_type.value for t in self.all_triggers],
            "priority": self.priority.value,
            "response_timeline": self.response_timeline,
            "consolidated_at": self.consolidated_at.isoformat(),
        }


ALL_EVALUATORS = [
    ThresholdTrigger(),
    ConcentrationTrigger(),
    FactorExposureTrigger(),
    MonthlyCalendarTrigger(),
    QuarterlyCalendarTrigger(),
    AnnualCalendarTrigger(),
    MarketCrashTrigger(),
    RegulatoryChangeTrigger(),
    ClientLifeEventTrigger(),
    TaxHarvestingTrigger(),
    CashFlowTrigger(),
]


class TriggerConsolidator:
    """
    Evaluate all trigger types for a portfolio and merge simultaneous fires
    into a single ConsolidatedTrigger with the highest priority timeline.
    """

    def __init__(self, evaluators: Optional[list] = None):
        self.evaluators = evaluators or ALL_EVALUATORS

    def evaluate_portfolio(self, portfolio_id: str, context: dict) -> Optional[ConsolidatedTrigger]:
        """Run all evaluators and consolidate fired triggers."""
        fired: list[TriggerEvent] = []
        for evaluator in self.evaluators:
            try:
                event = evaluator.evaluate(portfolio_id, context)
                if event is not None:
                    fired.append(event)
            except Exception:
                continue

        if not fired:
            return None

        # Sort by priority (critical first)
        fired.sort(key=lambda e: PRIORITY_ORDER[e.priority])
        primary = fired[0]

        return ConsolidatedTrigger(
            portfolio_id=portfolio_id,
            primary_trigger=primary,
            all_triggers=fired,
        )

    def evaluate_batch(
        self,
        portfolio_ids: list[str],
        contexts: dict[str, dict],
    ) -> list[ConsolidatedTrigger]:
        """Evaluate triggers for a batch of portfolios."""
        results = []
        for pid in portfolio_ids:
            ctx = contexts.get(pid, {})
            consolidated = self.evaluate_portfolio(pid, ctx)
            if consolidated is not None:
                results.append(consolidated)

        # Sort by priority
        results.sort(key=lambda ct: PRIORITY_ORDER[ct.priority])
        return results
