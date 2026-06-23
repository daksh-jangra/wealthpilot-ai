"""Manage and report on portfolio rebalancing constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ConstraintViolation:
    constraint_name: str
    description: str
    current_value: float
    limit: float
    severity: str  # "hard" | "soft"


class ConstraintManager:
    """
    Maintain the full constraint set for portfolio optimisation and
    report violations on proposed trade lists.
    """

    def __init__(
        self,
        turnover_budget: float = 0.20,
        min_trade_size_inr: float = 1_000,
        sector_limits: Optional[dict[str, float]] = None,
        issuer_limit: float = 0.10,
        international_equity_limit: float = 0.25,
    ):
        self.turnover_budget = turnover_budget
        self.min_trade_size_inr = min_trade_size_inr
        self.sector_limits = sector_limits or {
            "financials": 0.35,
            "technology": 0.30,
            "energy": 0.25,
            "healthcare": 0.25,
            "consumer": 0.25,
            "industrials": 0.20,
            "materials": 0.15,
            "utilities": 0.15,
            "real_estate": 0.15,
        }
        self.issuer_limit = issuer_limit
        self.international_equity_limit = international_equity_limit

    def check_post_trade(
        self,
        post_trade_weights: np.ndarray,
        trade_weights: np.ndarray,
        sector_weights: Optional[dict[str, float]] = None,
        issuer_weights: Optional[dict[str, float]] = None,
        portfolio_value_inr: float = 1_000_000,
    ) -> list[ConstraintViolation]:
        """Return list of constraint violations for a proposed trade."""
        violations = []

        # Long-only constraint
        if (post_trade_weights < -1e-6).any():
            neg_idxs = np.where(post_trade_weights < -1e-6)[0]
            violations.append(
                ConstraintViolation(
                    constraint_name="long_only",
                    description=f"Negative weights in asset classes {neg_idxs.tolist()}",
                    current_value=float(post_trade_weights.min()),
                    limit=0.0,
                    severity="hard",
                )
            )

        # Budget constraint: weights sum to ~1
        weight_sum = post_trade_weights.sum()
        if abs(weight_sum - 1.0) > 0.01:
            violations.append(
                ConstraintViolation(
                    constraint_name="budget",
                    description="Weights do not sum to 1",
                    current_value=float(weight_sum),
                    limit=1.0,
                    severity="hard",
                )
            )

        # Turnover limit
        turnover = float(np.abs(trade_weights).sum())
        if turnover > self.turnover_budget:
            violations.append(
                ConstraintViolation(
                    constraint_name="turnover",
                    description=f"Turnover {turnover:.2%} exceeds budget {self.turnover_budget:.2%}",
                    current_value=turnover,
                    limit=self.turnover_budget,
                    severity="soft",
                )
            )

        # International equity SEBI limit
        INTL_IDX = 1
        if post_trade_weights[INTL_IDX] > self.international_equity_limit:
            violations.append(
                ConstraintViolation(
                    constraint_name="sebi_intl_equity",
                    description="International equity exceeds SEBI retail limit",
                    current_value=float(post_trade_weights[INTL_IDX]),
                    limit=self.international_equity_limit,
                    severity="hard",
                )
            )

        # Sector concentration
        if sector_weights:
            for sector, weight in sector_weights.items():
                limit = self.sector_limits.get(sector, 0.20)
                if weight > limit:
                    violations.append(
                        ConstraintViolation(
                            constraint_name=f"sector_{sector}",
                            description=f"Sector {sector} weight {weight:.2%} > limit {limit:.2%}",
                            current_value=weight,
                            limit=limit,
                            severity="hard",
                        )
                    )

        # Single issuer concentration
        if issuer_weights:
            for issuer, weight in issuer_weights.items():
                if weight > self.issuer_limit:
                    violations.append(
                        ConstraintViolation(
                            constraint_name=f"issuer_{issuer}",
                            description=f"Issuer {issuer} weight {weight:.2%} > {self.issuer_limit:.2%}",
                            current_value=weight,
                            limit=self.issuer_limit,
                            severity="hard",
                        )
                    )

        return violations

    def has_hard_violations(self, violations: list[ConstraintViolation]) -> bool:
        return any(v.severity == "hard" for v in violations)

    def violations_summary(self, violations: list[ConstraintViolation]) -> dict:
        return {
            "total": len(violations),
            "hard": sum(1 for v in violations if v.severity == "hard"),
            "soft": sum(1 for v in violations if v.severity == "soft"),
            "details": [
                {"name": v.constraint_name, "description": v.description, "severity": v.severity}
                for v in violations
            ],
        }
