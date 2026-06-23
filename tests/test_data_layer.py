"""Tests for data generation layer."""

import numpy as np
import pandas as pd
import pytest

from src.data.market_data_simulator import MarketDataSimulator, MarketParameters
from src.data.client_profile_generator import ClientProfileGenerator
from src.data.portfolio_generator import PortfolioGenerator


# Market Data Simulator
def test_market_data_returns_252_days():
    sim = MarketDataSimulator(seed=42)
    returns = sim.simulate_returns()
    assert returns.shape == (252, 6)


def test_market_data_columns_correct():
    sim = MarketDataSimulator(seed=42)
    returns = sim.simulate_returns()
    assert "indian_equity" in returns.columns
    assert "cash" in returns.columns


def test_vix_series_length():
    sim = MarketDataSimulator(seed=42)
    vix = sim.get_vix_series()
    assert len(vix) == 252
    assert (vix > 0).all()


def test_price_levels_positive():
    sim = MarketDataSimulator(seed=42)
    prices = sim.simulate_price_levels()
    assert (prices > 0).all().all()


def test_nifty_returns_series():
    sim = MarketDataSimulator(seed=42)
    nifty = sim.get_nifty_returns()
    assert len(nifty) == 252
    assert nifty.name == "NIFTY_50"


def test_custom_params_seed_reproducible():
    params = MarketParameters()
    sim1 = MarketDataSimulator(params=params, seed=77)
    sim2 = MarketDataSimulator(params=params, seed=77)
    r1 = sim1.simulate_returns()
    r2 = sim2.simulate_returns()
    assert np.allclose(r1.values, r2.values)


# Client Profile Generator
def test_client_profiles_count():
    gen = ClientProfileGenerator(seed=42)
    profiles = gen.generate_all()
    assert len(profiles) == 50_000


def test_client_risk_categories_correct():
    gen = ClientProfileGenerator(seed=42)
    profiles = gen.generate_all()
    expected_cats = {
        "ultra_conservative",
        "conservative",
        "balanced",
        "aggressive",
        "ultra_aggressive",
    }
    assert set(profiles["risk_category"].unique()) == expected_cats


def test_client_risk_category_counts():
    gen = ClientProfileGenerator(seed=42)
    profiles = gen.generate_all()
    counts = profiles["risk_category"].value_counts()
    assert counts["ultra_conservative"] == 5000
    assert counts["conservative"] == 12000
    assert counts["balanced"] == 18000
    assert counts["aggressive"] == 10000
    assert counts["ultra_aggressive"] == 5000


def test_client_tax_bracket_valid():
    gen = ClientProfileGenerator(seed=42)
    profiles = gen.generate_all()
    assert profiles["tax_bracket"].isin([0.0, 0.20, 0.30]).all()


def test_client_risk_score_range():
    gen = ClientProfileGenerator(seed=42)
    profiles = gen.generate_all()
    assert profiles["risk_score"].between(1, 5).all()


# Portfolio Generator
def test_portfolio_allocations_sum_to_one():
    gen = PortfolioGenerator(seed=42)
    clients = ClientProfileGenerator(seed=42).generate_all()
    weights = gen.generate_current_allocations(clients)
    sums = weights.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6), "All portfolio weights must sum to 1"


def test_portfolio_weights_non_negative():
    gen = PortfolioGenerator(seed=42)
    clients = ClientProfileGenerator(seed=42).generate_all()
    weights = gen.generate_current_allocations(clients)
    assert (weights >= 0).all().all()


def test_portfolio_values_positive():
    gen = PortfolioGenerator(seed=42)
    clients = ClientProfileGenerator(seed=42).generate_all()
    values = gen.generate_portfolio_values(clients)
    assert (values > 0).all()


def test_securities_master_count():
    gen = PortfolioGenerator(seed=42)
    master = gen.get_securities_master(n_securities=500)
    assert len(master) == 500


def test_securities_master_asset_classes():
    gen = PortfolioGenerator(seed=42)
    master = gen.get_securities_master()
    expected = {
        "indian_equity",
        "international_equity",
        "indian_fixed_income",
        "international_fixed_income",
        "alternatives",
        "cash",
    }
    assert set(master["asset_class"].unique()) == expected


def test_portfolio_generator_small_client_set():
    """Verify generator works on a small custom client set."""
    from src.data.client_profile_generator import ClientProfileGenerator

    gen = PortfolioGenerator(seed=99)
    clients = ClientProfileGenerator(seed=99).generate_all()
    small = clients.iloc[:100]
    weights = gen.generate_current_allocations(small)
    assert weights.shape == (100, 6)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-6)


def test_portfolio_generator_tax_lots_have_required_columns():
    from src.data.client_profile_generator import ClientProfileGenerator

    gen = PortfolioGenerator(seed=42)
    clients = ClientProfileGenerator(seed=42).generate_all().iloc[:20]
    lots = gen.generate_tax_lots(clients)
    required = {
        "portfolio_id",
        "security_id",
        "asset_class",
        "quantity",
        "cost_per_unit_inr",
        "current_price_inr",
        "acquisition_date",
    }
    assert required.issubset(set(lots.columns))


def test_portfolio_generator_tax_lots_non_empty():
    from src.data.client_profile_generator import ClientProfileGenerator

    gen = PortfolioGenerator(seed=42)
    clients = ClientProfileGenerator(seed=42).generate_all().iloc[:10]
    lots = gen.generate_tax_lots(clients)
    assert len(lots) > 0
    assert (lots["quantity"] > 0).all()


def test_tax_lot_dataclass_properties():
    from src.data.portfolio_generator import TaxLot
    from datetime import date, timedelta

    old_date = date.today() - timedelta(days=400)
    lot = TaxLot(
        security_id="IEQ001",
        asset_class="indian_equity",
        quantity=10.0,
        cost_per_unit_inr=100.0,
        acquisition_date=old_date,
        portfolio_id="WP000001",
    )
    assert lot.cost_basis_inr == pytest.approx(1000.0)
    assert lot.holding_days >= 400
    assert lot.is_long_term is True


def test_securities_master_has_adv_column():
    gen = PortfolioGenerator(seed=42)
    master = gen.get_securities_master(n_securities=50)
    assert "avg_daily_volume_inr" in master.columns or "bid_ask_spread_bps" in master.columns
