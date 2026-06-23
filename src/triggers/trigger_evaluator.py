"""Base trigger evaluator and trigger event model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class TriggerType(str, Enum):
    THRESHOLD_ASSET_CLASS = "threshold_asset_class"
    THRESHOLD_FACTOR = "threshold_factor"
    THRESHOLD_CONCENTRATION = "threshold_concentration"
    CALENDAR_MONTHLY = "calendar_monthly"
    CALENDAR_QUARTERLY = "calendar_quarterly"
    CALENDAR_ANNUAL = "calendar_annual"
    EVENT_MARKET_CRASH = "event_market_crash"
    EVENT_REGULATORY = "event_regulatory"
    EVENT_CLIENT_LIFE = "event_client_life"
    EVENT_CORPORATE_ACTION = "event_corporate_action"
    EVENT_TAX_HARVESTING = "event_tax_harvesting"
    EVENT_CASH_FLOW = "event_cash_flow"


class TriggerPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


TRIGGER_PRIORITY_MAP: dict[TriggerType, TriggerPriority] = {
    TriggerType.THRESHOLD_ASSET_CLASS: TriggerPriority.HIGH,
    TriggerType.THRESHOLD_FACTOR: TriggerPriority.MEDIUM,
    TriggerType.THRESHOLD_CONCENTRATION: TriggerPriority.CRITICAL,
    TriggerType.CALENDAR_MONTHLY: TriggerPriority.LOW,
    TriggerType.CALENDAR_QUARTERLY: TriggerPriority.MEDIUM,
    TriggerType.CALENDAR_ANNUAL: TriggerPriority.MEDIUM,
    TriggerType.EVENT_MARKET_CRASH: TriggerPriority.CRITICAL,
    TriggerType.EVENT_REGULATORY: TriggerPriority.CRITICAL,
    TriggerType.EVENT_CLIENT_LIFE: TriggerPriority.HIGH,
    TriggerType.EVENT_CORPORATE_ACTION: TriggerPriority.HIGH,
    TriggerType.EVENT_TAX_HARVESTING: TriggerPriority.MEDIUM,
    TriggerType.EVENT_CASH_FLOW: TriggerPriority.LOW,
}

RESPONSE_TIMELINE_MAP: dict[TriggerPriority, str] = {
    TriggerPriority.CRITICAL: "immediate",
    TriggerPriority.HIGH: "within_1_session",
    TriggerPriority.MEDIUM: "within_3_sessions",
    TriggerPriority.LOW: "within_5_sessions",
}

PRIORITY_ORDER = {
    TriggerPriority.CRITICAL: 0,
    TriggerPriority.HIGH: 1,
    TriggerPriority.MEDIUM: 2,
    TriggerPriority.LOW: 3,
}


@dataclass
class TriggerEvent:
    portfolio_id: str
    trigger_type: TriggerType
    priority: TriggerPriority
    timestamp: datetime
    details: dict
    response_timeline: str = field(init=False)

    def __post_init__(self):
        self.response_timeline = RESPONSE_TIMELINE_MAP[self.priority]

    def to_dict(self) -> dict:
        return {
            "portfolio_id": self.portfolio_id,
            "trigger_type": self.trigger_type.value,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "response_timeline": self.response_timeline,
            "details": self.details,
        }


class TriggerEvaluator(ABC):
    """Abstract base class for all trigger evaluators."""

    @abstractmethod
    def evaluate(self, portfolio_id: str, context: dict) -> Optional[TriggerEvent]:
        """Evaluate whether a trigger fires for a given portfolio and context."""

    def _make_event(
        self,
        portfolio_id: str,
        trigger_type: TriggerType,
        details: dict,
        timestamp: Optional[datetime] = None,
    ) -> TriggerEvent:
        return TriggerEvent(
            portfolio_id=portfolio_id,
            trigger_type=trigger_type,
            priority=TRIGGER_PRIORITY_MAP[trigger_type],
            timestamp=timestamp or datetime.utcnow(),
            details=details,
        )
