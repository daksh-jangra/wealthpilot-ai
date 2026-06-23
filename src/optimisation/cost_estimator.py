"""Estimate explicit and implicit transaction costs for Indian market trades."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CostEstimate:
    brokerage_inr: float
    stt_inr: float  # Securities Transaction Tax
    stamp_duty_inr: float
    gst_on_brokerage_inr: float
    market_impact_inr: float
    total_explicit_inr: float
    total_implicit_inr: float
    total_inr: float
    total_bps: float  # basis points of trade value


class CostEstimator:
    """
    Estimate transaction costs for Indian equity and debt markets.
    Rates from SEBI/NSE/BSE as of FY2024-25.
    """

    # STT rates (equity delivery)
    STT_BUY_EQUITY = 0.001  # 0.10% on buy
    STT_SELL_EQUITY = 0.001  # 0.10% on sell

    # Stamp duty on buy
    STAMP_DUTY_EQUITY_BUY = 0.00015  # 0.015%
    STAMP_DUTY_DEBT_BUY = 0.00005

    # Brokerage (discount broker)
    BROKERAGE_FLAT_PER_ORDER = 20.0  # INR 20 flat per order
    MAX_BROKERAGE_PCT = 0.0025  # 0.25% cap

    # GST on brokerage
    GST_RATE = 0.18

    # Market impact: square-root model coefficient
    IMPACT_COEFFICIENT = 0.10

    def estimate_trade(
        self,
        trade_value_inr: float,
        asset_class: str,
        avg_daily_volume_inr: float,
        is_buy: bool = True,
    ) -> CostEstimate:
        """Estimate costs for a single trade."""
        if trade_value_inr <= 0:
            return CostEstimate(0, 0, 0, 0, 0, 0, 0, 0, 0)

        is_equity = "equity" in asset_class
        abs_value = abs(trade_value_inr)

        # Brokerage
        brokerage = min(self.BROKERAGE_FLAT_PER_ORDER, abs_value * self.MAX_BROKERAGE_PCT)
        gst = brokerage * self.GST_RATE

        # STT
        if is_equity:
            stt = abs_value * (self.STT_BUY_EQUITY if is_buy else self.STT_SELL_EQUITY)
        else:
            stt = 0.0

        # Stamp duty (buy side only)
        if is_buy:
            stamp = abs_value * (
                self.STAMP_DUTY_EQUITY_BUY if is_equity else self.STAMP_DUTY_DEBT_BUY
            )
        else:
            stamp = 0.0

        explicit = brokerage + gst + stt + stamp

        # Market impact: square-root model
        if avg_daily_volume_inr > 0:
            participation_rate = abs_value / avg_daily_volume_inr
            vol_pct = 0.25 if is_equity else 0.05
            impact = self.IMPACT_COEFFICIENT * vol_pct * np.sqrt(participation_rate) * abs_value
        else:
            impact = abs_value * 0.005  # 50 bps fallback for illiquid

        total = explicit + impact
        total_bps = (total / abs_value) * 10_000 if abs_value > 0 else 0

        return CostEstimate(
            brokerage_inr=round(brokerage, 2),
            stt_inr=round(stt, 2),
            stamp_duty_inr=round(stamp, 2),
            gst_on_brokerage_inr=round(gst, 2),
            market_impact_inr=round(impact, 2),
            total_explicit_inr=round(explicit, 2),
            total_implicit_inr=round(impact, 2),
            total_inr=round(total, 2),
            total_bps=round(total_bps, 1),
        )

    def estimate_trade_list(
        self,
        trades: list[dict],
        portfolio_value_inr: float,
    ) -> dict:
        """Aggregate cost estimates for a full trade list."""
        total_cost = 0.0
        details = []
        for trade in trades:
            est = self.estimate_trade(
                trade_value_inr=trade.get("trade_value_inr", 0),
                asset_class=trade.get("asset_class", "indian_equity"),
                avg_daily_volume_inr=trade.get("avg_daily_volume_inr", 1e7),
                is_buy=trade.get("is_buy", True),
            )
            total_cost += est.total_inr
            details.append({**trade, "cost_inr": est.total_inr, "cost_bps": est.total_bps})

        return {
            "total_cost_inr": round(total_cost, 2),
            "total_cost_bps": (
                round((total_cost / portfolio_value_inr) * 10_000, 1) if portfolio_value_inr else 0
            ),
            "cost_as_pct_portfolio": (
                round((total_cost / portfolio_value_inr) * 100, 3) if portfolio_value_inr else 0
            ),
            "trade_details": details,
        }
