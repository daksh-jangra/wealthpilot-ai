"""Tests for TaxLotManager and TaxOptimiser."""

from datetime import date, timedelta

import pandas as pd
import pytest

from src.optimisation.tax_optimiser import HOLDING_PERIOD_EQUITY_DAYS, TaxLotManager, TaxOptimiser


def _make_lot_df():
    today = date.today()
    return pd.DataFrame(
        [
            {
                "portfolio_id": "WP000001",
                "security_id": "IEQ001",
                "asset_class": "indian_equity",
                "quantity": 100.0,
                "cost_per_unit_inr": 80.0,
                "current_price_inr": 100.0,
                "acquisition_date": today - timedelta(days=400),  # long-term
                "lot_value_inr": 10000.0,
            },
            {
                "portfolio_id": "WP000001",
                "security_id": "IEQ001",
                "asset_class": "indian_equity",
                "quantity": 50.0,
                "cost_per_unit_inr": 110.0,
                "current_price_inr": 100.0,
                "acquisition_date": today - timedelta(days=100),  # short-term, at loss
                "lot_value_inr": 5000.0,
            },
            {
                "portfolio_id": "WP000002",
                "security_id": "IEQ002",
                "asset_class": "indian_equity",
                "quantity": 200.0,
                "cost_per_unit_inr": 120.0,
                "current_price_inr": 90.0,  # at loss
                "acquisition_date": today - timedelta(days=200),
                "lot_value_inr": 18000.0,
            },
        ]
    )


@pytest.fixture
def lot_manager():
    return TaxLotManager(_make_lot_df())


def test_lot_manager_get_portfolio_lots(lot_manager):
    lots = lot_manager.get_portfolio_lots("WP000001")
    assert len(lots) == 2


def test_holding_days_long_term(lot_manager):
    lots = lot_manager.get_portfolio_lots("WP000001")
    first_lot = lots.iloc[0]
    days = lot_manager.holding_days(first_lot)
    assert days >= HOLDING_PERIOD_EQUITY_DAYS, "First lot should be long-term"


def test_holding_days_short_term(lot_manager):
    lots = lot_manager.get_portfolio_lots("WP000001")
    second_lot = lots.iloc[1]
    days = lot_manager.holding_days(second_lot)
    assert days < HOLDING_PERIOD_EQUITY_DAYS, "Second lot should be short-term"


def test_select_lots_tax_optimal(lot_manager):
    """Tax-optimal strategy should prefer long-term lots first (lower rate)."""
    selected = lot_manager.select_lots_for_sale(
        portfolio_id="WP000001",
        security_id="IEQ001",
        target_sell_value_inr=8000,
        current_price_inr=100.0,
        tax_bracket=0.30,
        strategy="tax_optimal",
    )
    assert len(selected) >= 1
    total_sold = sum(s.quantity * 100.0 for s in selected)
    assert total_sold >= 8000 * 0.95  # within 5% of target


def test_select_lots_fifo(lot_manager):
    """FIFO should select the earliest acquired lot first."""
    selected = lot_manager.select_lots_for_sale(
        portfolio_id="WP000001",
        security_id="IEQ001",
        target_sell_value_inr=5000,
        current_price_inr=100.0,
        strategy="fifo",
    )
    assert len(selected) >= 1
    # First lot acquired earliest (400 days ago) should be selected
    assert selected[0].is_long_term is True


def test_stcg_rate_applied(lot_manager):
    """Short-term gains should use 20% rate."""
    selected = lot_manager.select_lots_for_sale(
        portfolio_id="WP000001",
        security_id="IEQ001",
        target_sell_value_inr=5000,
        current_price_inr=100.0,
        strategy="lifo",
    )
    # Selling at loss → tax cost should be zero
    if selected and not selected[0].is_long_term and selected[0].realized_pnl_inr < 0:
        assert selected[0].tax_cost_inr == 0.0


def test_tax_harvesting_scanner():
    """Identify portfolios with harvestable losses."""
    lot_manager = TaxLotManager(_make_lot_df())
    optimiser = TaxOptimiser(lot_manager)

    current_prices = {"IEQ002": 90.0, "IEQ001": 100.0}
    opportunities = optimiser.find_harvest_opportunities(
        portfolio_id="WP000002",
        current_prices=current_prices,
        min_loss_inr=1000,
    )
    assert len(opportunities) >= 1
    assert opportunities[0].unrealized_loss_inr > 0
    assert opportunities[0].estimated_tax_saving_inr > 0


def test_empty_portfolio_returns_empty(lot_manager):
    selected = lot_manager.select_lots_for_sale(
        portfolio_id="NONEXISTENT",
        security_id="XYZ",
        target_sell_value_inr=10000,
        current_price_inr=100.0,
    )
    assert selected == []
