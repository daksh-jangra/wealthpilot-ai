"""Event-driven triggers: market crashes, regulatory changes, life events, tax windows."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from src.triggers.trigger_evaluator import TriggerEvaluator, TriggerEvent, TriggerType


class MarketCrashTrigger(TriggerEvaluator):
    """Fires when the benchmark index declines > 10% from its recent high."""

    def __init__(self, crash_threshold: float = 0.10):
        self.crash_threshold = crash_threshold

    def evaluate(self, portfolio_id: str, context: dict) -> Optional[TriggerEvent]:
        drawdown: Optional[float] = context.get("benchmark_drawdown_from_high")
        if drawdown is None or abs(drawdown) < self.crash_threshold:
            return None
        return self._make_event(
            portfolio_id=portfolio_id,
            trigger_type=TriggerType.EVENT_MARKET_CRASH,
            details={
                "drawdown_pct": round(drawdown * 100, 2),
                "threshold_pct": round(self.crash_threshold * 100, 2),
                "benchmark": context.get("benchmark_name", "NIFTY_50"),
            },
        )


class RegulatoryChangeTrigger(TriggerEvaluator):
    """Fires when a new SEBI circular affects this portfolio's allocations."""

    def evaluate(self, portfolio_id: str, context: dict) -> Optional[TriggerEvent]:
        regulatory_events: list[dict] = context.get("regulatory_events", [])
        affected = [e for e in regulatory_events if portfolio_id in e.get("affected_portfolios", [])]
        if not affected:
            return None
        event = affected[0]
        return self._make_event(
            portfolio_id=portfolio_id,
            trigger_type=TriggerType.EVENT_REGULATORY,
            details={
                "circular_ref": event.get("circular_ref", "SEBI/UNKNOWN"),
                "rule_description": event.get("rule_description", ""),
                "compliance_deadline": event.get("deadline", ""),
                "affected_asset_class": event.get("affected_asset_class", ""),
            },
        )


class ClientLifeEventTrigger(TriggerEvaluator):
    """Fires on retirement, inheritance, large deposit/withdrawal, or goal change."""

    SIGNIFICANT_FLOW_THRESHOLD = 500_000  # INR 5 lakh

    def evaluate(self, portfolio_id: str, context: dict) -> Optional[TriggerEvent]:
        life_events: list[dict] = context.get("client_life_events", [])
        client_events = [e for e in life_events if e.get("portfolio_id") == portfolio_id]
        if not client_events:
            # Also check for large cash flows
            cash_flow = abs(context.get("pending_cash_flow_inr", 0))
            if cash_flow < self.SIGNIFICANT_FLOW_THRESHOLD:
                return None
            client_events = [{"event_type": "large_cash_flow", "amount_inr": cash_flow}]

        event = client_events[0]
        return self._make_event(
            portfolio_id=portfolio_id,
            trigger_type=TriggerType.EVENT_CLIENT_LIFE,
            details=event,
        )


class TaxHarvestingTrigger(TriggerEvaluator):
    """Fires during the March FY-end tax-loss harvesting window."""

    WINDOW_MONTH = 3
    WINDOW_START_DAY = 1
    WINDOW_END_DAY = 31

    def evaluate(self, portfolio_id: str, context: dict) -> Optional[TriggerEvent]:
        today = context.get("date", date.today())
        if isinstance(today, datetime):
            today = today.date()

        if today.month != self.WINDOW_MONTH:
            return None

        harvestable_loss_inr: float = context.get("harvestable_loss_inr", 0.0)
        if harvestable_loss_inr < 5000:
            return None

        return self._make_event(
            portfolio_id=portfolio_id,
            trigger_type=TriggerType.EVENT_TAX_HARVESTING,
            details={
                "harvestable_loss_inr": round(harvestable_loss_inr, 2),
                "window": f"{today.year}-03-{self.WINDOW_START_DAY:02d} to {today.year}-03-{self.WINDOW_END_DAY:02d}",
                "fy_end": f"{today.year}-03-31",
            },
        )


class CashFlowTrigger(TriggerEvaluator):
    """Fires when SIP inflow or dividend needs to be invested."""

    MIN_INVESTABLE_INR = 5_000

    def evaluate(self, portfolio_id: str, context: dict) -> Optional[TriggerEvent]:
        cash_available = context.get("uninvested_cash_inr", 0.0)
        sip_inflow = context.get("sip_inflow_inr", 0.0)
        total_investable = cash_available + sip_inflow

        if total_investable < self.MIN_INVESTABLE_INR:
            return None

        return self._make_event(
            portfolio_id=portfolio_id,
            trigger_type=TriggerType.EVENT_CASH_FLOW,
            details={
                "uninvested_cash_inr": round(cash_available, 2),
                "sip_inflow_inr": round(sip_inflow, 2),
                "total_investable_inr": round(total_investable, 2),
            },
        )
