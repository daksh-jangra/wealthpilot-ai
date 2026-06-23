"""LIME integration for local linear approximations of rebalancing decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except Exception:
    LIME_AVAILABLE = False

from src.explainability.shap_integration import FEATURE_NAMES, SurrogateModelTrainer


@dataclass
class LIMEExplanation:
    portfolio_id: str
    prediction: float
    top_features: list[dict]    # [{feature, value, weight, direction}]
    local_model_score: float    # fidelity of local linear model
    explanation_text: str


class LIMEIntegration:
    """Local Interpretable Model-Agnostic Explanations for rebalancing decisions."""

    def __init__(self, trainer: Optional[SurrogateModelTrainer] = None):
        self.trainer = trainer or SurrogateModelTrainer()
        self._explainer: Optional[object] = None
        self._X_train: Optional[pd.DataFrame] = None

    def _ensure_ready(self) -> None:
        if not LIME_AVAILABLE:
            raise ImportError("lime is required: pip install lime")
        if self.trainer.model is None:
            self.trainer.train()
        if self._explainer is None:
            X_train, _ = self.trainer.generate_training_data(n_samples=2000)
            self._X_train = X_train
            self._explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train.values,
                feature_names=FEATURE_NAMES,
                class_names=["no_rebalance", "rebalance"],
                mode="classification",
                discretize_continuous=True,
                random_state=42,
            )

    def explain(
        self,
        portfolio_id: str,
        features: dict,
        num_features: int = 5,
    ) -> LIMEExplanation:
        self._ensure_ready()

        feature_values = np.array([features.get(f, 0.0) for f in FEATURE_NAMES])
        prediction = float(self.trainer.model.predict_proba([feature_values])[0, 1])

        exp = self._explainer.explain_instance(
            data_row=feature_values,
            predict_fn=self.trainer.model.predict_proba,
            num_features=num_features,
            num_samples=500,
        )

        top_features = []
        for feat_str, weight in exp.as_list(label=1):
            top_features.append({
                "feature": feat_str,
                "weight": round(float(weight), 4),
                "direction": "increases_rebalancing" if weight > 0 else "decreases_rebalancing",
            })

        score = float(exp.score) if hasattr(exp, "score") else 0.85
        explanation_text = self._format_explanation(top_features, prediction)

        return LIMEExplanation(
            portfolio_id=portfolio_id,
            prediction=prediction,
            top_features=top_features,
            local_model_score=score,
            explanation_text=explanation_text,
        )

    def _format_explanation(self, top_features: list[dict], prediction: float) -> str:
        action = "REBALANCE" if prediction >= 0.5 else "HOLD"
        lines = [f"Decision: {action} (confidence: {prediction:.1%})", "Key drivers:"]
        for feat in top_features:
            sign = "+" if feat["weight"] > 0 else "-"
            lines.append(f"  {sign} {feat['feature']} (weight: {feat['weight']:+.3f})")
        return "\n".join(lines)

    def explain_batch(
        self, portfolio_ids: list[str], features_list: list[dict]
    ) -> list[LIMEExplanation]:
        return [
            self.explain(pid, feat)
            for pid, feat in zip(portfolio_ids, features_list)
        ]
