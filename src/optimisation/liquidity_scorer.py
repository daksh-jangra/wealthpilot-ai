"""Liquidity scoring and multi-day execution scheduling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LiquidityScore:
    security_id: str
    asset_class: str
    score: float  # 0-100 (100 = most liquid)
    avg_daily_volume_inr: float
    bid_ask_spread_bps: float
    market_impact_bps_per_pct: float
    max_single_day_trade_inr: float  # 10% of ADV
    recommended_days: int  # days needed to execute trade


class LiquidityScorer:
    """
    Assign liquidity scores and execution scheduling to securities.
    Follows market microstructure best practices for Indian markets.
    """

    MAX_PARTICIPATION_RATE = 0.10  # max 10% of average daily volume per day

    def score_security(
        self,
        security_id: str,
        asset_class: str,
        avg_daily_volume_inr: float,
        bid_ask_spread_bps: float,
    ) -> LiquidityScore:
        """Compute a composite liquidity score (0-100)."""
        # Volume component (log-normalised against Indian market medians)
        volume_score = min(100, 50 * np.log10(max(avg_daily_volume_inr, 1) / 1e6 + 1))

        # Spread component (tighter = better)
        spread_score = max(0, 50 * (1 - bid_ask_spread_bps / 100))

        composite = 0.6 * volume_score + 0.4 * spread_score

        # Market impact per 1% participation
        vol_pct = 0.25 if "equity" in asset_class else 0.05
        impact_bps = 10 * vol_pct * 100  # simplified

        max_daily = avg_daily_volume_inr * self.MAX_PARTICIPATION_RATE

        return LiquidityScore(
            security_id=security_id,
            asset_class=asset_class,
            score=round(composite, 1),
            avg_daily_volume_inr=avg_daily_volume_inr,
            bid_ask_spread_bps=bid_ask_spread_bps,
            market_impact_bps_per_pct=round(impact_bps, 1),
            max_single_day_trade_inr=round(max_daily, 0),
            recommended_days=1,
        )

    def score_batch(self, securities_master: pd.DataFrame) -> pd.DataFrame:
        """Score all securities in the master table."""
        records = []
        for _, row in securities_master.iterrows():
            score = self.score_security(
                security_id=str(row["security_id"]),
                asset_class=str(row["asset_class"]),
                avg_daily_volume_inr=float(row["avg_daily_volume_inr"]),
                bid_ask_spread_bps=float(row["bid_ask_spread_bps"]),
            )
            records.append(
                {
                    "security_id": score.security_id,
                    "liquidity_score": score.score,
                    "max_daily_trade_inr": score.max_single_day_trade_inr,
                    "impact_bps_per_pct": score.market_impact_bps_per_pct,
                }
            )
        return pd.DataFrame(records)

    def schedule_execution(
        self,
        trade_value_inr: float,
        security_id: str,
        avg_daily_volume_inr: float,
        execution_strategy: str = "auto",
    ) -> dict:
        """Determine VWAP/TWAP execution schedule for a trade."""
        max_daily = avg_daily_volume_inr * self.MAX_PARTICIPATION_RATE
        days_needed = max(1, int(np.ceil(abs(trade_value_inr) / max_daily)))

        if days_needed == 1:
            strategy = "market_order"
        elif days_needed <= 3:
            strategy = "VWAP"
        else:
            strategy = "TWAP"

        if execution_strategy != "auto":
            strategy = execution_strategy

        return {
            "security_id": security_id,
            "total_trade_value_inr": round(abs(trade_value_inr), 2),
            "days_required": days_needed,
            "daily_slice_inr": round(abs(trade_value_inr) / days_needed, 2),
            "execution_strategy": strategy,
        }
