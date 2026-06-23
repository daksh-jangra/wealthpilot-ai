"""Convert optimiser output into executable security-level trade lists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from src.optimisation.portfolio_optimiser import OptimisationResult
from src.optimisation.cost_estimator import CostEstimator
from src.optimisation.liquidity_scorer import LiquidityScorer

ASSET_CLASSES = [
    "indian_equity", "international_equity",
    "indian_fixed_income", "international_fixed_income",
    "alternatives", "cash",
]


@dataclass
class Trade:
    trade_id: str
    portfolio_id: str
    security_id: str
    asset_class: str
    direction: str          # "BUY" or "SELL"
    quantity: float
    estimated_price_inr: float
    trade_value_inr: float
    estimated_cost_inr: float
    estimated_cost_bps: float
    execution_strategy: str
    days_to_execute: int


class TradeListGenerator:
    """Generate security-level trades from asset-class rebalancing signals."""

    def __init__(
        self,
        cost_estimator: Optional[CostEstimator] = None,
        liquidity_scorer: Optional[LiquidityScorer] = None,
    ):
        self.cost_estimator = cost_estimator or CostEstimator()
        self.liquidity_scorer = liquidity_scorer or LiquidityScorer()
        self._trade_counter = 0

    def generate(
        self,
        portfolio_id: str,
        opt_result: OptimisationResult,
        portfolio_value_inr: float,
        securities_master: pd.DataFrame,
        current_holdings: Optional[dict[str, float]] = None,
    ) -> list[Trade]:
        """
        Convert asset-class trade weights into security-level trades.

        For each asset class with a non-zero trade weight, select one or more
        representative securities from the master table to execute the trade.
        """
        trades = []
        for ac_idx, ac in enumerate(ASSET_CLASSES):
            trade_weight = opt_result.trade_weights[ac_idx]
            if abs(trade_weight) < 1e-4:
                continue

            trade_value = trade_weight * portfolio_value_inr
            ac_securities = securities_master[securities_master["asset_class"] == ac]
            if ac_securities.empty:
                continue

            # Pick most liquid security for the trade
            security = ac_securities.sort_values("avg_daily_volume_inr", ascending=False).iloc[0]
            sec_id = str(security["security_id"])
            price = float(security["current_price_inr"])
            adv = float(security["avg_daily_volume_inr"])

            qty = abs(trade_value) / price if price > 0 else 0
            is_buy = trade_value > 0
            cost_est = self.cost_estimator.estimate_trade(
                trade_value_inr=abs(trade_value),
                asset_class=ac,
                avg_daily_volume_inr=adv,
                is_buy=is_buy,
            )
            schedule = self.liquidity_scorer.schedule_execution(trade_value, sec_id, adv)

            self._trade_counter += 1
            trades.append(Trade(
                trade_id=f"TRD{self._trade_counter:08d}",
                portfolio_id=portfolio_id,
                security_id=sec_id,
                asset_class=ac,
                direction="BUY" if is_buy else "SELL",
                quantity=round(qty, 4),
                estimated_price_inr=round(price, 2),
                trade_value_inr=round(abs(trade_value), 2),
                estimated_cost_inr=cost_est.total_inr,
                estimated_cost_bps=cost_est.total_bps,
                execution_strategy=schedule["execution_strategy"],
                days_to_execute=schedule["days_required"],
            ))

        return trades

    def trades_to_dataframe(self, trades: list[Trade]) -> pd.DataFrame:
        return pd.DataFrame([vars(t) for t in trades])

    def trade_list_summary(self, trades: list[Trade]) -> dict:
        if not trades:
            return {"trade_count": 0, "total_value_inr": 0, "total_cost_inr": 0}
        return {
            "trade_count": len(trades),
            "buy_count": sum(1 for t in trades if t.direction == "BUY"),
            "sell_count": sum(1 for t in trades if t.direction == "SELL"),
            "total_value_inr": round(sum(t.trade_value_inr for t in trades), 2),
            "total_cost_inr": round(sum(t.estimated_cost_inr for t in trades), 2),
            "total_cost_bps": round(sum(t.estimated_cost_bps for t in trades) / len(trades), 1),
            "max_days_to_execute": max(t.days_to_execute for t in trades),
        }
