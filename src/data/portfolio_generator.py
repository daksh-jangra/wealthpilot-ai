"""Generate 50,000 synthetic portfolio holdings with realistic tax lots."""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from src.data.client_profile_generator import ClientProfileGenerator

ASSET_CLASSES = [
    "indian_equity",
    "international_equity",
    "indian_fixed_income",
    "international_fixed_income",
    "alternatives",
    "cash",
]

TARGET_ALLOCATIONS = {
    "ultra_conservative": [0.10, 0.05, 0.45, 0.15, 0.10, 0.15],
    "conservative": [0.20, 0.10, 0.35, 0.10, 0.12, 0.13],
    "balanced": [0.35, 0.15, 0.20, 0.10, 0.12, 0.08],
    "aggressive": [0.50, 0.20, 0.10, 0.05, 0.10, 0.05],
    "ultra_aggressive": [0.60, 0.25, 0.05, 0.00, 0.07, 0.03],
}


@dataclass
class TaxLot:
    security_id: str
    asset_class: str
    quantity: float
    cost_per_unit_inr: float
    acquisition_date: date
    portfolio_id: str

    @property
    def cost_basis_inr(self) -> float:
        return self.quantity * self.cost_per_unit_inr

    @property
    def holding_days(self) -> int:
        return (date.today() - self.acquisition_date).days

    @property
    def is_long_term(self) -> bool:
        return self.holding_days >= 365


class PortfolioGenerator:
    """Generate portfolio holdings and tax lots for all clients."""

    def __init__(self, seed: int = 42, portfolio_value_inr: float = 5_000_000):
        self.rng = np.random.default_rng(seed)
        self.default_portfolio_value = portfolio_value_inr
        self.client_gen = ClientProfileGenerator(seed=seed)

    def generate_current_allocations(
        self,
        client_profiles: Optional[pd.DataFrame] = None,
        drift_noise_std: float = 0.04,
    ) -> pd.DataFrame:
        """
        Return DataFrame of shape (50000, 6) with current asset-class weights,
        perturbed from target by realistic drift noise.
        """
        if client_profiles is None:
            client_profiles = self.client_gen.generate_all()

        targets = np.array([TARGET_ALLOCATIONS[cat] for cat in client_profiles["risk_category"]])

        # Add realistic drift noise (correlated across asset classes)
        noise = self.rng.normal(0, drift_noise_std, size=targets.shape)
        # Positive noise in equity  implies negative in fixed income (they offset)
        noise[:, 2] -= noise[:, 0] * 0.5
        noise[:, 3] -= noise[:, 1] * 0.5

        current = targets + noise
        # Clip negatives and renormalise
        current = np.clip(current, 0.0, None)
        row_sums = current.sum(axis=1, keepdims=True)
        current = current / np.where(row_sums == 0, 1, row_sums)

        df = pd.DataFrame(current, columns=ASSET_CLASSES, index=client_profiles["client_id"])
        df.index.name = "client_id"
        return df

    def generate_portfolio_values(
        self,
        client_profiles: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Return portfolio total value in INR for each client."""
        if client_profiles is None:
            client_profiles = self.client_gen.generate_all()

        # Log-normal distributed portfolio values: mean ~50L, median ~10L
        log_mean = np.log(1_000_000)
        log_std = 1.5
        values = self.rng.lognormal(log_mean, log_std, size=len(client_profiles))
        values = np.clip(values, 50_000, 500_000_000)
        return pd.Series(values, index=client_profiles["client_id"], name="portfolio_value_inr")

    def generate_tax_lots(
        self,
        client_profiles: Optional[pd.DataFrame] = None,
        current_prices: Optional[pd.Series] = None,
        n_lots_per_holding: int = 3,
    ) -> pd.DataFrame:
        """Generate tax lots for every portfolio holding."""
        if client_profiles is None:
            client_profiles = self.client_gen.generate_all()

        portfolio_values = self.generate_portfolio_values(client_profiles)
        current_allocs = self.generate_current_allocations(client_profiles)

        rows = []
        for _, client in client_profiles.iterrows():
            cid = client["client_id"]
            pv = portfolio_values[cid]

            for ac_idx, ac in enumerate(ASSET_CLASSES):
                ac_value = pv * current_allocs.loc[cid, ac]
                if ac_value < 1000:
                    continue

                # Spread across 1-4 tax lots with different acquisition dates
                lots = self.rng.integers(1, n_lots_per_holding + 1)
                lot_fractions = self.rng.dirichlet(np.ones(lots))

                for lot_frac in lot_fractions:
                    lot_value = ac_value * lot_frac
                    # Random acquisition date: 0-3 years ago
                    days_ago = int(self.rng.integers(30, 1100))
                    acq_date = date.today() - timedelta(days=days_ago)

                    # Cost per unit: current price ± some appreciation
                    current_price = (
                        100.0 if current_prices is None else float(current_prices.get(ac, 100.0))
                    )
                    appreciation = self.rng.uniform(0.70, 1.50)
                    cost_per_unit = current_price / appreciation

                    quantity = lot_value / current_price
                    security_id = f"{ac.upper()[:3]}{self.rng.integers(100, 999):03d}"

                    rows.append(
                        {
                            "portfolio_id": cid,
                            "security_id": security_id,
                            "asset_class": ac,
                            "quantity": round(quantity, 4),
                            "cost_per_unit_inr": round(cost_per_unit, 2),
                            "current_price_inr": round(current_price, 2),
                            "acquisition_date": acq_date.isoformat(),
                            "lot_value_inr": round(lot_value, 2),
                        }
                    )

        return pd.DataFrame(rows)

    def get_securities_master(self, n_securities: int = 500) -> pd.DataFrame:
        """Return a master table of tradeable securities."""
        sectors = [
            "financials",
            "technology",
            "energy",
            "healthcare",
            "consumer",
            "industrials",
            "materials",
            "utilities",
            "real_estate",
        ]
        ac_dist = {
            "indian_equity": 200,
            "international_equity": 100,
            "indian_fixed_income": 100,
            "international_fixed_income": 50,
            "alternatives": 30,
            "cash": 20,
        }
        records = []
        for ac, count in ac_dist.items():
            for i in range(count):
                sector = self.rng.choice(sectors)
                avg_daily_vol = float(self.rng.lognormal(np.log(1e7), 1.0))
                bid_ask_bps = float(self.rng.uniform(3, 50))
                records.append(
                    {
                        "security_id": f"{ac.upper()[:3]}{i:03d}",
                        "asset_class": ac,
                        "sector": sector,
                        "avg_daily_volume_inr": avg_daily_vol,
                        "bid_ask_spread_bps": bid_ask_bps,
                        "current_price_inr": float(self.rng.uniform(10, 5000)),
                        "is_esg_compliant": bool(self.rng.random() < 0.60),
                        "exit_load_pct": 0.01 if ac == "indian_equity" else 0.0,
                    }
                )
        return pd.DataFrame(records)
