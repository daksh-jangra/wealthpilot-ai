"""Compliance-level explanation builder: full audit record with SEBI references."""

from __future__ import annotations
from datetime import datetime
from typing import Optional

from src.explainability.explanation_generator import ExplanationGenerator, ExplanationOutput
from src.explainability.shap_integration import SHAPIntegration, SHAPExplanation


SEBI_REFERENCES = {
    "sebi_intl_equity": "SEBI Circular SEBI/HO/IMD/IMD-II DOF3/P/CIR/2022/0154 (international equity limits)",
    "threshold_concentration": "SEBI LODR Regulation 25 (concentration risk disclosure)",
    "regulatory_change": "SEBI Circular as referenced in decision metadata",
}


class ComplianceExplainer:
    """Produces compliance-grade audit records for every rebalancing decision."""

    def __init__(
        self,
        generator: ExplanationGenerator,
        shap: Optional[SHAPIntegration] = None,
    ):
        self.generator = generator
        self.shap = shap or SHAPIntegration()

    def explain(self, decision_metadata: dict) -> ExplanationOutput:
        # Enrich metadata with SHAP attribution
        try:
            features = {
                "max_drift_pct": decision_metadata.get("max_drift_pct", 0),
                "sum_abs_drift_pct": decision_metadata.get("sum_abs_drift_pct", 0),
                "days_since_last_rebalance": decision_metadata.get("days_since_last_rebalance", 90),
                "vix": decision_metadata.get("vix", 18),
                "risk_score": decision_metadata.get("risk_score", 3),
                "ltcg_lot_fraction": decision_metadata.get("ltcg_lot_fraction", 0.5),
                "sector_concentration_max": decision_metadata.get("sector_concentration_max", 0.2),
                "portfolio_value_log": decision_metadata.get("portfolio_value_log", 14),
            }
            shap_exp = self.shap.explain(decision_metadata.get("portfolio_id", "UNKNOWN"), features)
            decision_metadata["shap_top_features"] = shap_exp.top_features
            decision_metadata["counterfactual"] = shap_exp.counterfactual
        except Exception:
            pass

        decision_metadata["sebi_references"] = self._get_sebi_refs(decision_metadata)
        decision_metadata["audit_timestamp"] = datetime.utcnow().isoformat()
        return self.generator.generate("compliance", decision_metadata)

    def _get_sebi_refs(self, meta: dict) -> list[str]:
        refs = []
        violations = meta.get("constraint_checks", {}).get("violations", [])
        for v in violations:
            name = v.get("name", "")
            if name in SEBI_REFERENCES:
                refs.append(SEBI_REFERENCES[name])
        trigger = meta.get("trigger_type", "")
        if "regulatory" in trigger:
            refs.append(SEBI_REFERENCES["regulatory_change"])
        return refs

    def generate_full_audit_record(
        self,
        decision_metadata: dict,
        explanation: ExplanationOutput,
        trade_list: Optional[list] = None,
    ) -> dict:
        return {
            "decision_id": decision_metadata.get("decision_id", "UNKNOWN"),
            "audit_timestamp": datetime.utcnow().isoformat(),
            "portfolio_id": decision_metadata.get("portfolio_id"),
            "trigger": decision_metadata.get("trigger_type"),
            "explanation_narrative": explanation.narrative,
            "key_metrics": explanation.key_numbers,
            "sebi_references": decision_metadata.get("sebi_references", []),
            "constraint_checks": decision_metadata.get("constraint_checks", {}),
            "shap_attribution": decision_metadata.get("shap_top_features", []),
            "counterfactual": decision_metadata.get("counterfactual"),
            "trade_list": trade_list or [],
            "model_version": decision_metadata.get("model_version", "1.0.0"),
            "override_history": decision_metadata.get("override_history", []),
        }
