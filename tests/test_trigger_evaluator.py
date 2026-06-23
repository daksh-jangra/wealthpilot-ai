"""Tests for all trigger types and consolidation logic."""

import pytest
from datetime import date, datetime

from src.triggers.trigger_evaluator import TriggerType, TriggerPriority
from src.triggers.threshold_trigger import ThresholdTrigger, ConcentrationTrigger
from src.triggers.calendar_trigger import MonthlyCalendarTrigger, QuarterlyCalendarTrigger
from src.triggers.event_trigger import MarketCrashTrigger, TaxHarvestingTrigger, CashFlowTrigger
from src.triggers.trigger_consolidator import TriggerConsolidator
from src.monitoring.drift_calculator import DriftResult, DriftSeverity
import numpy as np


def _make_drift_result(severity: DriftSeverity, max_drift: float = 0.05) -> DriftResult:
    w_current = np.array([0.40, 0.15, 0.20, 0.10, 0.07, 0.08])
    w_target = np.array([0.35, 0.15, 0.20, 0.10, 0.12, 0.08])
    return DriftResult(
        portfolio_id="TEST001",
        risk_category="balanced",
        current_weights=w_current,
        target_weights=w_target,
        abs_drift=np.abs(w_current - w_target),
        max_drift=max_drift,
        sum_abs_drift=0.10,
        rmsd=0.04,
        drift_band=0.03,
        severity=severity,
        breaching_asset_classes=["indian_equity"],
    )


def test_threshold_trigger_fires_on_drift():
    trigger = ThresholdTrigger()
    ctx = {"drift_result": _make_drift_result(DriftSeverity.HIGH)}
    event = trigger.evaluate("TEST001", ctx)
    assert event is not None
    assert event.trigger_type == TriggerType.THRESHOLD_ASSET_CLASS
    assert event.priority == TriggerPriority.HIGH


def test_threshold_trigger_none_on_no_drift():
    trigger = ThresholdTrigger()
    ctx = {"drift_result": _make_drift_result(DriftSeverity.NONE, max_drift=0.01)}
    event = trigger.evaluate("TEST001", ctx)
    assert event is None


def test_threshold_trigger_critical_on_critical_severity():
    trigger = ThresholdTrigger()
    ctx = {"drift_result": _make_drift_result(DriftSeverity.CRITICAL, max_drift=0.08)}
    event = trigger.evaluate("TEST001", ctx)
    assert event is not None
    assert event.trigger_type == TriggerType.THRESHOLD_CONCENTRATION
    assert event.priority == TriggerPriority.CRITICAL


def test_concentration_trigger_fires_on_violation():
    trigger = ConcentrationTrigger()
    ctx = {"sector_weights": {"financials": 0.40}, "issuer_weights": {}}
    event = trigger.evaluate("TEST002", ctx)
    assert event is not None
    assert event.trigger_type == TriggerType.THRESHOLD_CONCENTRATION


def test_concentration_trigger_none_within_limits():
    trigger = ConcentrationTrigger()
    ctx = {"sector_weights": {"financials": 0.20}, "issuer_weights": {"AAAA": 0.05}}
    event = trigger.evaluate("TEST002", ctx)
    assert event is None


def test_monthly_calendar_fires_on_first_business_day():
    trigger = MonthlyCalendarTrigger()
    # 2025-01-01 is a Wednesday
    ctx = {"date": date(2025, 1, 1)}
    event = trigger.evaluate("TEST003", ctx)
    assert event is not None
    assert event.trigger_type == TriggerType.CALENDAR_MONTHLY


def test_monthly_calendar_no_fire_on_other_days():
    trigger = MonthlyCalendarTrigger()
    ctx = {"date": date(2025, 1, 15)}
    event = trigger.evaluate("TEST003", ctx)
    assert event is None


def test_quarterly_trigger_fires_in_first_week_of_quarter():
    trigger = QuarterlyCalendarTrigger()
    ctx = {"date": date(2025, 4, 1)}  # April 1 = Q1 start
    event = trigger.evaluate("TEST004", ctx)
    assert event is not None
    assert event.trigger_type == TriggerType.CALENDAR_QUARTERLY


def test_market_crash_trigger_fires_on_large_drop():
    trigger = MarketCrashTrigger(crash_threshold=0.10)
    ctx = {"benchmark_drawdown_from_high": -0.15, "benchmark_name": "NIFTY_50"}
    event = trigger.evaluate("TEST005", ctx)
    assert event is not None
    assert event.trigger_type == TriggerType.EVENT_MARKET_CRASH
    assert event.priority == TriggerPriority.CRITICAL


def test_market_crash_no_fire_on_small_drop():
    trigger = MarketCrashTrigger(crash_threshold=0.10)
    ctx = {"benchmark_drawdown_from_high": -0.05}
    event = trigger.evaluate("TEST005", ctx)
    assert event is None


def test_tax_harvesting_fires_in_march():
    trigger = TaxHarvestingTrigger()
    ctx = {"date": date(2025, 3, 15), "harvestable_loss_inr": 50000}
    event = trigger.evaluate("TEST006", ctx)
    assert event is not None
    assert event.trigger_type == TriggerType.EVENT_TAX_HARVESTING


def test_tax_harvesting_no_fire_outside_march():
    trigger = TaxHarvestingTrigger()
    ctx = {"date": date(2025, 6, 15), "harvestable_loss_inr": 50000}
    event = trigger.evaluate("TEST006", ctx)
    assert event is None


def test_cash_flow_trigger_fires_on_large_sip():
    trigger = CashFlowTrigger()
    ctx = {"sip_inflow_inr": 25000, "uninvested_cash_inr": 0}
    event = trigger.evaluate("TEST007", ctx)
    assert event is not None
    assert event.trigger_type == TriggerType.EVENT_CASH_FLOW


def test_trigger_consolidator_merges_multiple():
    """Multiple triggers for same portfolio should consolidate to highest priority."""
    consolidator = TriggerConsolidator()
    ctx = {
        "TEST008": {
            "drift_result": _make_drift_result(DriftSeverity.CRITICAL, max_drift=0.08),
            "date": date(2025, 4, 1),
            "sector_weights": {},
            "issuer_weights": {},
            "benchmark_drawdown_from_high": -0.05,
            "harvestable_loss_inr": 0,
            "sip_inflow_inr": 0,
            "uninvested_cash_inr": 0,
            "regulatory_events": [],
            "client_life_events": [],
            "factor_tilts": {},
            "pending_cash_flow_inr": 0,
        }
    }
    results = consolidator.evaluate_batch(["TEST008"], ctx)
    assert len(results) == 1
    consolidated = results[0]
    # Critical drift should be primary
    assert consolidated.priority == TriggerPriority.CRITICAL
    assert len(consolidated.all_triggers) >= 1


def test_trigger_event_response_timeline():
    trigger = MarketCrashTrigger()
    ctx = {"benchmark_drawdown_from_high": -0.15}
    event = trigger.evaluate("TEST009", ctx)
    assert event is not None
    assert event.response_timeline == "immediate"
