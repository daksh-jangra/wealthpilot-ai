"""Detect systematic biases in agent rebalancing decisions."""

from __future__ import annotations

import pandas as pd
from scipy import stats


class BiasDetector:
    """
    Analyse agent decisions for systematic patterns:
    - Over/under-rebalancing by risk category
    - Security favouritism
    - Momentum or contrarian bias in trigger timing
    """

    def detect_category_bias(self, decision_log: list[dict]) -> dict:
        """Test if rebalancing rates differ significantly across risk categories."""
        df = pd.DataFrame(decision_log)
        if df.empty or "risk_category" not in df.columns:
            return {"error": "Insufficient data"}

        counts = df["risk_category"].value_counts()
        expected_uniform = len(df) / len(counts)
        chi2, p_value = stats.chisquare(counts.values, f_exp=[expected_uniform] * len(counts))

        return {
            "category_counts": counts.to_dict(),
            "chi2_statistic": round(float(chi2), 4),
            "p_value": round(float(p_value), 4),
            "significant_bias": bool(p_value < 0.05),
            "interpretation": (
                "Significant bias in rebalancing frequency across risk categories"
                if p_value < 0.05
                else "No significant bias detected"
            ),
        }

    def detect_security_bias(self, trade_log: list[dict]) -> dict:
        """Check if agent systematically favours certain securities."""
        if not trade_log:
            return {"error": "No trades"}

        df = pd.DataFrame(trade_log)
        if "security_id" not in df.columns:
            return {"error": "No security_id field"}

        freq = df["security_id"].value_counts()
        top_5 = freq.head(5).to_dict()
        total_trades = len(df)
        concentration = freq.head(5).sum() / total_trades

        return {
            "top_5_securities": top_5,
            "top_5_concentration": round(float(concentration), 3),
            "concentration_flag": bool(concentration > 0.20),
            "total_unique_securities": int(df["security_id"].nunique()),
        }

    def detect_momentum_bias(self, decision_log: list[dict]) -> dict:
        """Check if agent triggers correlated with recent market direction (momentum bias)."""
        drifts = [d.get("decision_metadata", {}).get("max_drift_pct", 0) for d in decision_log]
        vix_values = [d.get("decision_metadata", {}).get("vix", 18) for d in decision_log]

        if len(drifts) < 10:
            return {"error": "Insufficient data"}

        correlation, p_value = stats.pearsonr(drifts, vix_values)
        return {
            "drift_vix_correlation": round(float(correlation), 3),
            "p_value": round(float(p_value), 4),
            "momentum_bias_flag": abs(correlation) > 0.3 and p_value < 0.05,
            "interpretation": (
                f"{'Positive' if correlation > 0 else 'Negative'} correlation between drift and VIX "
                f"({'significant' if p_value < 0.05 else 'not significant'})"
            ),
        }

    def full_bias_report(
        self, decision_log: list[dict], trade_log: list[dict] | None = None
    ) -> dict:
        return {
            "category_bias": self.detect_category_bias(decision_log),
            "security_bias": self.detect_security_bias(trade_log or []),
            "momentum_bias": self.detect_momentum_bias(decision_log),
            "report_generated": pd.Timestamp.utcnow().isoformat(),
        }


# Avoid missing import annotation
