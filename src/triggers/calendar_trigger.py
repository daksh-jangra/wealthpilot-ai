"""Calendar-based triggers: monthly, quarterly, and annual review schedules."""

from __future__ import annotations

from datetime import date, datetime

from src.triggers.trigger_evaluator import TriggerEvaluator, TriggerEvent, TriggerType


def _is_business_day(d: date) -> bool:
    return d.weekday() < 5


def _next_business_day(d: date) -> date:
    while not _is_business_day(d):
        from datetime import timedelta

        d += timedelta(days=1)
    return d


class MonthlyCalendarTrigger(TriggerEvaluator):
    """Fires on the first business day of every month."""

    def evaluate(self, portfolio_id: str, context: dict) -> TriggerEvent | None:
        today = context.get("date", date.today())
        if isinstance(today, datetime):
            today = today.date()
        first = _next_business_day(today.replace(day=1))
        if today != first:
            return None
        return self._make_event(
            portfolio_id=portfolio_id,
            trigger_type=TriggerType.CALENDAR_MONTHLY,
            details={"review_date": today.isoformat(), "review_type": "monthly"},
        )


class QuarterlyCalendarTrigger(TriggerEvaluator):
    """Fires in the first week of each quarter (Jan, Apr, Jul, Oct)."""

    QUARTER_MONTHS = {1, 4, 7, 10}

    def evaluate(self, portfolio_id: str, context: dict) -> TriggerEvent | None:
        today = context.get("date", date.today())
        if isinstance(today, datetime):
            today = today.date()

        if today.month not in self.QUARTER_MONTHS:
            return None
        first = _next_business_day(today.replace(day=1))
        from datetime import timedelta

        week_end = first + timedelta(days=4)
        if not (first <= today <= week_end):
            return None

        quarter = {1: "Q4", 4: "Q1", 7: "Q2", 10: "Q3"}[today.month]
        return self._make_event(
            portfolio_id=portfolio_id,
            trigger_type=TriggerType.CALENDAR_QUARTERLY,
            details={
                "review_date": today.isoformat(),
                "quarter": quarter,
                "review_type": "quarterly",
            },
        )


class AnnualCalendarTrigger(TriggerEvaluator):
    """Fires on the client's annual review anniversary (or start of April FY)."""

    def evaluate(self, portfolio_id: str, context: dict) -> TriggerEvent | None:
        today = context.get("date", date.today())
        if isinstance(today, datetime):
            today = today.date()

        anniversary: date | None = context.get("client_anniversary")
        if anniversary is None:
            # Default: April 1 (Indian FY start)
            anniversary = date(today.year, 4, 1)

        anniversary_this_year = anniversary.replace(year=today.year)
        anniversary_this_year = _next_business_day(anniversary_this_year)

        if today != anniversary_this_year:
            return None

        return self._make_event(
            portfolio_id=portfolio_id,
            trigger_type=TriggerType.CALENDAR_ANNUAL,
            details={
                "review_date": today.isoformat(),
                "review_type": "annual_strategic",
            },
        )
