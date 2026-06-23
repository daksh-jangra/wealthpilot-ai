"""SHAP integration: surrogate model training and Tree SHAP value computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

try:
    import shap
    import xgboost as xgb

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


FEATURE_NAMES = [
    "max_drift_pct",
    "sum_abs_drift_pct",
    "days_since_last_rebalance",
    "vix",
    "risk_score",
    "ltcg_lot_fraction",
    "sector_concentration_max",
    "portfolio_value_log",
]


@dataclass
class SHAPExplanation:
    portfolio_id: str
    shap_values: np.ndarray
    feature_values: np.ndarray
    feature_names: list[str]
    base_value: float
    prediction: float
    top_features: list[dict]  # sorted by |shap| descending
    counterfactual: Optional[str] = None


class SurrogateModelTrainer:
    """Train an XGBoost surrogate model that predicts rebalancing probability."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model: Optional[xgb.XGBClassifier] = None
        self.explainer: Optional[shap.TreeExplainer] = None

    def generate_training_data(self, n_samples: int = 10000) -> tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic training data based on known rebalancing rules."""
        if not SHAP_AVAILABLE:
            raise ImportError("xgboost and shap are required")
        rng = np.random.default_rng(42)

        X = pd.DataFrame(
            {
                "max_drift_pct": rng.uniform(0, 15, n_samples),
                "sum_abs_drift_pct": rng.uniform(0, 30, n_samples),
                "days_since_last_rebalance": rng.integers(0, 365, n_samples).astype(float),
                "vix": rng.uniform(10, 50, n_samples),
                "risk_score": rng.integers(1, 6, n_samples).astype(float),
                "ltcg_lot_fraction": rng.uniform(0, 1, n_samples),
                "sector_concentration_max": rng.uniform(0, 0.5, n_samples),
                "portfolio_value_log": rng.uniform(10, 20, n_samples),
            }
        )

        # Label: rebalance if drift > threshold or VIX spike
        drift_bands = [0.02, 0.025, 0.03, 0.04, 0.05]
        band = np.array([drift_bands[int(rs) - 1] for rs in X["risk_score"]])
        y = ((X["max_drift_pct"].values / 100 > band) | (X["vix"].values > 35)).astype(int)

        return X, pd.Series(y, name="rebalance")

    def train(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None) -> None:
        if not SHAP_AVAILABLE:
            raise ImportError("xgboost and shap are required")
        if X is None or y is None:
            X, y = self.generate_training_data()

        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            random_state=self.random_state,
            eval_metric="logloss",
            verbosity=0,
        )
        self.model.fit(X, y)
        self.explainer = shap.TreeExplainer(self.model)

    def get_shap_explanation(
        self,
        portfolio_id: str,
        features: dict,
    ) -> SHAPExplanation:
        """Compute SHAP values for a single portfolio decision."""
        if self.model is None or self.explainer is None:
            self.train()

        feature_values = np.array([features.get(f, 0.0) for f in FEATURE_NAMES])
        X_row = pd.DataFrame([feature_values], columns=FEATURE_NAMES)

        shap_values = self.explainer.shap_values(X_row)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # positive class
        else:
            sv = shap_values[0]

        base_value = float(
            self.explainer.expected_value[1]
            if isinstance(self.explainer.expected_value, list)
            else self.explainer.expected_value
        )
        prediction = float(self.model.predict_proba(X_row)[0, 1])

        # Sort features by |shap|
        sorted_idx = np.argsort(np.abs(sv))[::-1]
        top_features = [
            {
                "feature": FEATURE_NAMES[i],
                "value": round(float(feature_values[i]), 3),
                "shap_value": round(float(sv[i]), 4),
                "direction": "increases_rebalancing" if sv[i] > 0 else "decreases_rebalancing",
            }
            for i in sorted_idx[:5]
        ]

        # Counterfactual: find the key feature to flip decision
        counterfactual = self._generate_counterfactual(portfolio_id, feature_values, sv, prediction)

        return SHAPExplanation(
            portfolio_id=portfolio_id,
            shap_values=sv,
            feature_values=feature_values,
            feature_names=FEATURE_NAMES,
            base_value=base_value,
            prediction=prediction,
            top_features=top_features,
            counterfactual=counterfactual,
        )

    def _generate_counterfactual(
        self,
        portfolio_id: str,
        feature_values: np.ndarray,
        shap_values: np.ndarray,
        prediction: float,
    ) -> str:
        """Find the minimal perturbation that would flip the rebalancing decision."""
        top_idx = int(np.argmax(np.abs(shap_values)))
        feature = FEATURE_NAMES[top_idx]
        current_val = feature_values[top_idx]
        shap_val = shap_values[top_idx]

        if prediction >= 0.5:
            # Would not rebalance if feature were lower
            cf_val = current_val * 0.70
            action = "would not have triggered a rebalance"
        else:
            cf_val = current_val * 1.30
            action = "would have triggered a rebalance"

        return (
            f"If '{feature.replace('_', ' ')}' were {cf_val:.2f} instead of "
            f"{current_val:.2f}, the agent {action}."
        )


class SHAPIntegration:
    """Wrapper combining trainer and explanation formatting."""

    def __init__(self):
        self.trainer = SurrogateModelTrainer()
        self._trained = False

    def ensure_trained(self) -> None:
        if not self._trained:
            self.trainer.train()
            self._trained = True

    def explain(self, portfolio_id: str, features: dict) -> SHAPExplanation:
        self.ensure_trained()
        return self.trainer.get_shap_explanation(portfolio_id, features)

    def format_for_compliance(self, explanation: SHAPExplanation) -> str:
        lines = [
            "SHAP FEATURE ATTRIBUTION (Tree SHAP):",
            f"  Baseline rebalancing probability: {explanation.base_value:.3f}",
            f"  Model prediction: {explanation.prediction:.3f}",
            "  Top contributing features:",
        ]
        for feat in explanation.top_features:
            direction_sym = "+" if feat["shap_value"] > 0 else "-"
            lines.append(
                f"    {direction_sym} {feat['feature']}: value={feat['value']}, "
                f"SHAP={feat['shap_value']:+.4f} ({feat['direction']})"
            )
        if explanation.counterfactual:
            lines.append(f"  Counterfactual: {explanation.counterfactual}")
        return "\n".join(lines)
