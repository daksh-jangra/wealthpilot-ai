"""Circuit-breaker kill switch for halting all autonomous rebalancing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class KillSwitchEvent:
    event_id: str
    action: str  # "activated" | "deactivated"
    reason: str
    triggered_by: str  # "manual" | "auto_vix" | "auto_error_rate"
    timestamp: str
    vix_at_trigger: float | None = None
    error_rate_at_trigger: float | None = None


class KillSwitch:
    """
    Manual and automatic circuit breaker for the rebalancing agent.
    Automatic triggers: VIX > 40 or system error rate > 1%.
    """

    VIX_THRESHOLD = 40.0
    ERROR_RATE_THRESHOLD = 0.01

    def __init__(self):
        self._active = False
        self._events: list[KillSwitchEvent] = []
        self._current_vix: float = 18.0
        self._error_count: int = 0
        self._processed_count: int = 0

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self, reason: str, triggered_by: str = "manual") -> KillSwitchEvent:
        import uuid

        self._active = True
        event = KillSwitchEvent(
            event_id=f"KS{uuid.uuid4().hex[:8].upper()}",
            action="activated",
            reason=reason,
            triggered_by=triggered_by,
            timestamp=datetime.utcnow().isoformat(),
            vix_at_trigger=self._current_vix,
            error_rate_at_trigger=self._error_rate,
        )
        self._events.append(event)
        return event

    def deactivate(self, reason: str, triggered_by: str = "manual") -> KillSwitchEvent:
        import uuid

        self._active = False
        event = KillSwitchEvent(
            event_id=f"KS{uuid.uuid4().hex[:8].upper()}",
            action="deactivated",
            reason=reason,
            triggered_by=triggered_by,
            timestamp=datetime.utcnow().isoformat(),
        )
        self._events.append(event)
        return event

    def check_auto_triggers(self, vix: float, error_count: int, processed: int) -> bool:
        """Check automatic activation criteria. Returns True if kill switch was activated."""
        self._current_vix = vix
        self._error_count = error_count
        self._processed_count = processed

        if vix > self.VIX_THRESHOLD and not self._active:
            self.activate(
                reason=f"VIX {vix:.1f} exceeded threshold {self.VIX_THRESHOLD}",
                triggered_by="auto_vix",
            )
            return True

        if processed > 0 and self._error_rate > self.ERROR_RATE_THRESHOLD and not self._active:
            self.activate(
                reason=f"Error rate {self._error_rate:.2%} exceeded threshold {self.ERROR_RATE_THRESHOLD:.2%}",
                triggered_by="auto_error_rate",
            )
            return True

        return False

    def record_error(self) -> None:
        self._error_count += 1

    def record_processed(self) -> None:
        self._processed_count += 1

    @property
    def _error_rate(self) -> float:
        if self._processed_count == 0:
            return 0.0
        return self._error_count / self._processed_count

    def get_event_log(self) -> list[dict]:
        return [
            {
                "event_id": e.event_id,
                "action": e.action,
                "reason": e.reason,
                "triggered_by": e.triggered_by,
                "timestamp": e.timestamp,
            }
            for e in self._events
        ]

    def status(self) -> dict:
        return {
            "active": self._active,
            "vix": self._current_vix,
            "error_rate": round(self._error_rate, 4),
            "total_events": len(self._events),
        }
