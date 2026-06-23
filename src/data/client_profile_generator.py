"""Generate synthetic client profiles for 50,000 WealthPilot AI portfolios."""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClientProfile:
    client_id: str
    risk_category: str
    risk_score: int  # 1-5
    tax_bracket: float  # 0.0, 0.20, or 0.30
    monthly_sip_inr: float  # recurring inflow
    planned_withdrawal_inr: float  # upcoming cash need
    restricted_securities: list[str]
    esg_screen: bool
    sector_preferences: dict[str, float]  # sector -> tilt (-1 to +1)
    age: int
    investment_horizon_years: int


class ClientProfileGenerator:
    """Generate realistic client profiles distributed across risk categories."""

    RISK_CATEGORY_MAP = {
        "ultra_conservative": (1, 5000),
        "conservative": (2, 12000),
        "balanced": (3, 18000),
        "aggressive": (4, 10000),
        "ultra_aggressive": (5, 5000),
    }

    SECTORS = [
        "financials",
        "technology",
        "energy",
        "healthcare",
        "consumer",
        "industrials",
        "materials",
        "utilities",
    ]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_all(self) -> pd.DataFrame:
        """Generate all 50,000 client profiles as a DataFrame."""
        profiles = []
        client_idx = 0
        for category, (risk_score_center, count) in self.RISK_CATEGORY_MAP.items():
            for _ in range(count):
                profile = self._generate_one(client_idx, category, risk_score_center)
                profiles.append(vars(profile))
                client_idx += 1
        df = pd.DataFrame(profiles)
        return df

    def _generate_one(self, idx: int, category: str, risk_score_center: int) -> ClientProfile:
        score = int(
            np.clip(
                self.rng.integers(max(1, risk_score_center - 1), min(5, risk_score_center + 1) + 1),
                1,
                5,
            )
        )

        # Tax bracket distribution (30% high-income, 50% mid, 20% zero)
        tax_bracket = self.rng.choice([0.0, 0.20, 0.30], p=[0.20, 0.50, 0.30])

        # SIP amounts (INR): 5k to 1 lakh/month
        sip = float(
            self.rng.choice([5000, 10000, 25000, 50000, 100000], p=[0.25, 0.35, 0.25, 0.12, 0.03])
        )

        # Planned withdrawal: 60% have none, rest have 1-20L
        withdrawal = 0.0
        if self.rng.random() < 0.40:
            withdrawal = float(self.rng.integers(100000, 2000000))

        # Restricted securities: 0-3 random ISIN-like codes
        n_restricted = int(self.rng.integers(0, 4))
        restricted = [
            f"INE{self.rng.integers(100, 999):03d}A01{self.rng.integers(10, 99):02d}"
            for _ in range(n_restricted)
        ]

        esg = bool(self.rng.random() < 0.15)

        # Sector preferences: random tilts for 0-2 sectors
        prefs: dict[str, float] = {}
        n_prefs = int(self.rng.integers(0, 3))
        for sector in self.rng.choice(self.SECTORS, size=n_prefs, replace=False):
            prefs[str(sector)] = float(self.rng.uniform(-0.5, 0.5))

        age_ranges = {
            "ultra_conservative": (55, 80),
            "conservative": (45, 65),
            "balanced": (35, 55),
            "aggressive": (25, 45),
            "ultra_aggressive": (22, 35),
        }
        lo, hi = age_ranges[category]
        age = int(self.rng.integers(lo, hi + 1))
        horizon = max(1, 65 - age + int(self.rng.integers(-5, 6)))

        return ClientProfile(
            client_id=f"WP{idx:06d}",
            risk_category=category,
            risk_score=score,
            tax_bracket=float(tax_bracket),
            monthly_sip_inr=sip,
            planned_withdrawal_inr=withdrawal,
            restricted_securities=restricted,
            esg_screen=esg,
            sector_preferences=prefs,
            age=age,
            investment_horizon_years=horizon,
        )
