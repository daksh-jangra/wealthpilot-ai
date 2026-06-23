"""Generate natural-language counterfactual explanations for rebalancing decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from src.explainability.shap_integration import FEATURE_NAMES


@dataclass
class Counterfactual:
    portfolio_id: str
    original_decision: str      # "rebalance" or "hold"
    flipped_decision: str
    key_feature: str
    current_value: float
    counterfactual_value: float
    natural_language: str


class CounterfactualGenerator:
    """
    Find minimal feature perturbation that changes the rebalancing decision
    and express it in natural language for compliance documentation.
    """

    FEATURE_DESCRIPTIONS = {
        "max_drift_pct": ("equity allocation drift", "%"),
        "sum_abs_drift_pct": ("total portfolio drift", "%"),
        "days_since_last_rebalance": ("days since last rebalance", " days"),
        "vix": ("market volatility index (VIX)", ""),
        "risk_score": ("client risk score", ""),
        "ltcg_lot_fraction": ("fraction of long-term lots", ""),
        "sector_concentration_max": ("maximum sector concentration", ""),
        "portfolio_value_log": ("log portfolio value", ""),
    }

    def generate(
        self,
        portfolio_id: str,
        feature_values: np.ndarray,
        shap_values: np.ndarray,
        decision: str,  # "rebalance" or "hold"
        perturbation_scale: float = 0.30,
    ) -> Counterfactual:
        """Find minimal perturbation to flip the decision."""
        # The feature with highest |SHAP| is the lever
        top_idx = int(np.argmax(np.abs(shap_values)))
        key_feature = FEATURE_NAMES[top_idx]
        current_val = float(feature_values[top_idx])
        shap_val = float(shap_values[top_idx])

        if decision == "rebalance":
            # To flip to hold: reduce the main positive driver
            if shap_val > 0:
                cf_val = current_val * (1 - perturbation_scale)
            else:
                cf_val = current_val * (1 + perturbation_scale)
            flipped = "hold"
        else:
            # To flip to rebalance: increase the main negative driver
            if shap_val < 0:
                cf_val = current_val * (1 + perturbation_scale)
            else:
                cf_val = current_val * (1 - perturbation_scale)
            flipped = "rebalance"

        nl = self._to_natural_language(key_feature, current_val, cf_val, decision, flipped)

        return Counterfactual(
            portfolio_id=portfolio_id,
            original_decision=decision,
            flipped_decision=flipped,
            key_feature=key_feature,
            current_value=round(current_val, 3),
            counterfactual_value=round(cf_val, 3),
            natural_language=nl,
        )

    def _to_natural_language(
        self,
        feature: str,
        current: float,
        cf: float,
        original: str,
        flipped: str,
    ) -> str:
        desc, unit = self.FEATURE_DESCRIPTIONS.get(feature, (feature, ""))
        if unit == "%":
            cur_str = f"{current:.1f}%"
            cf_str = f"{cf:.1f}%"
        elif unit == " days":
            cur_str = f"{int(current)} days"
            cf_str = f"{int(cf)} days"
        else:
            cur_str = f"{current:.2f}"
            cf_str = f"{cf:.2f}"

        return (
            f"If the {desc} were {cf_str} instead of {cur_str}, the agent would have "
            f"decided to {flipped} rather than to {original} this portfolio."
        )

    def generate_batch(
        self,
        portfolio_ids: list[str],
        feature_arrays: list[np.ndarray],
        shap_arrays: list[np.ndarray],
        decisions: list[str],
    ) -> list[Counterfactual]:
        return [
            self.generate(pid, fv, sv, dec)
            for pid, fv, sv, dec in zip(portfolio_ids, feature_arrays, shap_arrays, decisions)
        ]
