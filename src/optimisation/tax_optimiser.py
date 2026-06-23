"""Tax-aware lot selection, loss harvesting, and wash-sale avoidance for Indian portfolios."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional
import pandas as pd
import numpy as np


TAX_RATE_LTCG_EQUITY = 0.125    # 12.5% post-2024 budget (grandfathering pre-2018)
TAX_RATE_STCG_EQUITY = 0.20    # 20% post-2024 budget
TAX_RATE_LTCG_DEBT = 0.125     # post-2023 (indexation removed for most)
TAX_RATE_STCG_DEBT = None      # added to income slab — use investor's marginal rate
LTCG_EQUITY_EXEMPT_LIMIT = 125_000  # INR 1.25 lakh LTCG exempt
HOLDING_PERIOD_EQUITY_DAYS = 365
HOLDING_PERIOD_DEBT_DAYS = 36 * 30  # 3 years (pre-2023 rule; now moot for most)
WASH_SALE_DAYS = 30


@dataclass
class TaxLotSale:
    security_id: str
    asset_class: str
    quantity: float
    cost_basis_inr: float
    current_price_inr: float
    acquisition_date: date
    realized_pnl_inr: float
    is_long_term: bool
    tax_cost_inr: float


@dataclass
class HarvestOpportunity:
    portfolio_id: str
    security_id: str
    asset_class: str
    unrealized_loss_inr: float
    quantity: float
    substitute_security_id: Optional[str]
    estimated_tax_saving_inr: float


class TaxLotManager:
    """Track and manage tax lots at position level."""

    def __init__(self, tax_lots_df: pd.DataFrame):
        self.lots = tax_lots_df.copy()
        if "acquisition_date" in self.lots.columns:
            self.lots["acquisition_date"] = pd.to_datetime(self.lots["acquisition_date"]).dt.date

    def get_portfolio_lots(self, portfolio_id: str) -> pd.DataFrame:
        return self.lots[self.lots["portfolio_id"] == portfolio_id].copy()

    def holding_days(self, row: pd.Series) -> int:
        acq = row["acquisition_date"]
        if isinstance(acq, str):
            acq = date.fromisoformat(acq)
        return (date.today() - acq).days

    def select_lots_for_sale(
        self,
        portfolio_id: str,
        security_id: str,
        target_sell_value_inr: float,
        current_price_inr: float,
        tax_bracket: float = 0.30,
        strategy: str = "tax_optimal",
    ) -> list[TaxLotSale]:
        """
        Select which tax lots to sell to minimise total tax cost.

        Strategies:
        - tax_optimal: prioritise long-term lots with lowest gains
        - fifo: first-in, first-out
        - lifo: last-in, first-out
        - specific_id: select by lot index
        """
        lots = self.get_portfolio_lots(portfolio_id)
        lots = lots[lots["security_id"] == security_id].copy()
        if lots.empty:
            return []

        lots["holding_days"] = lots.apply(self.holding_days, axis=1)
        lots["is_long_term"] = lots["holding_days"] >= HOLDING_PERIOD_EQUITY_DAYS
        lots["unrealized_pnl"] = (current_price_inr - lots["cost_per_unit_inr"]) * lots["quantity"]
        lots["tax_per_unit"] = lots.apply(
            lambda r: self._compute_tax_per_unit(r, current_price_inr, tax_bracket), axis=1
        )

        if strategy == "tax_optimal":
            # Sort: losses first, then lowest tax lots
            lots = lots.sort_values(["is_long_term", "tax_per_unit"], ascending=[False, True])
        elif strategy == "fifo":
            lots = lots.sort_values("acquisition_date")
        elif strategy == "lifo":
            lots = lots.sort_values("acquisition_date", ascending=False)

        selected = []
        remaining_value = target_sell_value_inr
        for _, lot in lots.iterrows():
            if remaining_value <= 0:
                break
            lot_value = lot["quantity"] * current_price_inr
            sell_fraction = min(1.0, remaining_value / lot_value)
            sell_qty = lot["quantity"] * sell_fraction
            sell_value = sell_qty * current_price_inr
            cost = sell_qty * lot["cost_per_unit_inr"]
            pnl = sell_value - cost
            is_lt = bool(lot["is_long_term"])
            tax_cost = self._compute_tax(pnl, is_lt, lot.get("asset_class", "indian_equity"), tax_bracket)

            selected.append(TaxLotSale(
                security_id=security_id,
                asset_class=str(lot.get("asset_class", "indian_equity")),
                quantity=round(sell_qty, 4),
                cost_basis_inr=round(cost, 2),
                current_price_inr=current_price_inr,
                acquisition_date=lot["acquisition_date"],
                realized_pnl_inr=round(pnl, 2),
                is_long_term=is_lt,
                tax_cost_inr=round(tax_cost, 2),
            ))
            remaining_value -= sell_value

        return selected

    def _compute_tax_per_unit(self, lot: pd.Series, current_price: float, bracket: float) -> float:
        pnl = current_price - lot["cost_per_unit_inr"]
        return self._compute_tax(pnl, bool(lot["is_long_term"]), "indian_equity", bracket)

    def _compute_tax(self, pnl: float, is_long_term: bool, asset_class: str, bracket: float) -> float:
        if pnl <= 0:
            return 0.0
        if "equity" in asset_class:
            rate = TAX_RATE_LTCG_EQUITY if is_long_term else TAX_RATE_STCG_EQUITY
        else:
            rate = TAX_RATE_LTCG_DEBT if is_long_term else bracket
        return max(0.0, pnl * rate)


class TaxOptimiser:
    """Identify tax-loss harvesting opportunities and optimise lot selection."""

    SUBSTITUTE_MAP = {
        "IND001": "IND002",  # e.g., Nifty50 ETF -> Nifty Next 50 ETF
        "IND002": "IND003",
        "INT001": "INT002",
    }

    def __init__(self, lot_manager: TaxLotManager):
        self.lot_manager = lot_manager

    def find_harvest_opportunities(
        self,
        portfolio_id: str,
        current_prices: dict[str, float],
        min_loss_inr: float = 5_000,
        tax_bracket: float = 0.30,
    ) -> list[HarvestOpportunity]:
        """Identify positions with harvestable losses."""
        lots = self.lot_manager.get_portfolio_lots(portfolio_id)
        opportunities = []

        for security_id, group in lots.groupby("security_id"):
            security_id = str(security_id)
            price = current_prices.get(security_id, group["current_price_inr"].iloc[0])
            total_qty = group["quantity"].sum()
            total_cost = (group["quantity"] * group["cost_per_unit_inr"]).sum()
            current_value = total_qty * price
            unrealized_pnl = current_value - total_cost

            if unrealized_pnl >= -min_loss_inr:
                continue

            asset_class = str(group["asset_class"].iloc[0])
            rate = TAX_RATE_STCG_EQUITY if "equity" in asset_class else tax_bracket
            tax_saving = abs(unrealized_pnl) * rate
            substitute = self.SUBSTITUTE_MAP.get(security_id)

            # Wash-sale check: don't buy back within 30 days (Indian equivalent)
            if substitute and self._recent_purchase(portfolio_id, substitute, days=WASH_SALE_DAYS):
                substitute = None

            opportunities.append(HarvestOpportunity(
                portfolio_id=portfolio_id,
                security_id=security_id,
                asset_class=asset_class,
                unrealized_loss_inr=round(abs(unrealized_pnl), 2),
                quantity=round(total_qty, 4),
                substitute_security_id=substitute,
                estimated_tax_saving_inr=round(tax_saving, 2),
            ))

        opportunities.sort(key=lambda o: o.estimated_tax_saving_inr, reverse=True)
        return opportunities

    def _recent_purchase(self, portfolio_id: str, security_id: str, days: int = 30) -> bool:
        """Check if this security was purchased within the last N days (wash-sale check)."""
        lots = self.lot_manager.get_portfolio_lots(portfolio_id)
        lots = lots[lots["security_id"] == security_id]
        if lots.empty:
            return False
        cutoff = date.today() - timedelta(days=days)
        recent = lots[lots["acquisition_date"] >= cutoff]
        return not recent.empty
