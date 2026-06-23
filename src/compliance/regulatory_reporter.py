"""Generate SEBI-compliant regulatory reports and audit trails."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
import pandas as pd
import json


class RegulatoryReporter:
    """
    Generate SEBI-compliant reports:
    - Complete transaction history with decision rationale
    - Suitability assessment documentation
    - Algorithmic trading audit trail
    - Exception reports
    """

    def generate_transaction_report(
        self,
        decision_log: list[dict],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Complete transaction history with decision rationale."""
        rows = []
        for d in decision_log:
            for trade in d.get("trades", []):
                rows.append({
                    "decision_id": d.get("decision_id"),
                    "portfolio_id": d.get("decision_metadata", {}).get("portfolio_id"),
                    "timestamp": d.get("decision_metadata", {}).get("timestamp"),
                    "trigger_type": d.get("decision_metadata", {}).get("trigger_type"),
                    "risk_category": d.get("decision_metadata", {}).get("risk_category"),
                    "trade_id": trade.get("trade_id"),
                    "security_id": trade.get("security_id"),
                    "asset_class": trade.get("asset_class"),
                    "direction": trade.get("direction"),
                    "quantity": trade.get("quantity"),
                    "trade_value_inr": trade.get("trade_value_inr"),
                    "estimated_cost_inr": trade.get("estimated_cost_inr"),
                    "execution_strategy": trade.get("execution_strategy"),
                    "decision_rationale": d.get("decision_metadata", {}).get("trigger_type"),
                })
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        return df

    def generate_suitability_report(self, decision_log: list[dict]) -> list[dict]:
        """Suitability assessment for each rebalancing recommendation."""
        records = []
        for d in decision_log:
            meta = d.get("decision_metadata", {})
            records.append({
                "decision_id": d.get("decision_id"),
                "portfolio_id": meta.get("portfolio_id"),
                "risk_category": meta.get("risk_category"),
                "risk_score": meta.get("risk_score"),
                "trigger": meta.get("trigger_type"),
                "drift_pct": meta.get("max_drift_pct"),
                "suitability_rationale": (
                    f"Rebalancing restores portfolio to client's stated risk category "
                    f"'{meta.get('risk_category')}' as per onboarding risk questionnaire. "
                    f"Drift of {meta.get('max_drift_pct', 0):.1f}% exceeded the "
                    f"category threshold, creating a risk profile inconsistent with client objectives."
                ),
                "constraint_violations": meta.get("constraint_checks", {}).get("hard", 0),
                "sebi_compliant": meta.get("constraint_checks", {}).get("hard", 0) == 0,
            })
        return records

    def generate_algo_audit_trail(self, decision: dict) -> str:
        """Structured audit trail for a single algorithmic decision."""
        meta = decision.get("decision_metadata", {})
        lines = [
            f"ALGORITHMIC TRADING AUDIT TRAIL",
            f"=" * 60,
            f"Decision ID: {decision.get('decision_id', 'UNKNOWN')}",
            f"Timestamp: {meta.get('timestamp', 'UNKNOWN')}",
            f"Portfolio ID: {meta.get('portfolio_id', 'UNKNOWN')}",
            f"Risk Category: {meta.get('risk_category', 'UNKNOWN')}",
            f"",
            f"TRIGGER ANALYSIS",
            f"-" * 40,
            f"Primary Trigger: {meta.get('trigger_type', 'UNKNOWN')}",
            f"Priority: {meta.get('trigger_priority', 'UNKNOWN')}",
            f"Max Drift: {meta.get('max_drift_pct', 0):.2f}%",
            f"Breaching Asset Classes: {meta.get('breaching_asset_classes', [])}",
            f"",
            f"OPTIMISATION RESULT",
            f"-" * 40,
            f"Status: {decision.get('optimisation_result', {}).status if hasattr(decision.get('optimisation_result', {}), 'status') else 'completed'}",
            f"Tracking Error Before: {meta.get('tracking_error_before', 0):.3f}%",
            f"Tracking Error After: {meta.get('tracking_error_after', 0):.3f}%",
            f"Turnover: {meta.get('turnover', 0):.2%}",
            f"",
            f"CONSTRAINT SATISFACTION",
            f"-" * 40,
            f"Hard Violations: {meta.get('constraint_checks', {}).get('hard', 0)}",
            f"Soft Violations: {meta.get('constraint_checks', {}).get('soft', 0)}",
            f"",
            f"COST & TAX",
            f"-" * 40,
            f"Total Cost: INR {meta.get('total_cost_inr', 0):,.0f}",
            f"Tax Impact: INR {meta.get('tax_impact_inr', 0):,.0f}",
            f"",
            f"MODEL PROVENANCE",
            f"-" * 40,
            f"Model Version: {meta.get('model_version', '1.0.0')}",
            f"Agent Framework: CrewAI + Claude claude-sonnet-4-6",
            f"Optimiser: CVXPY/OSQP",
            f"",
            f"OVERRIDE HISTORY",
            f"-" * 40,
            f"Overrides: {len(meta.get('override_history', []))}",
        ]
        return "\n".join(lines)

    def generate_exception_report(self, decision_log: list[dict]) -> pd.DataFrame:
        """Report all decisions with constraint violations or overrides."""
        exceptions = []
        for d in decision_log:
            meta = d.get("decision_metadata", {})
            hard_violations = meta.get("constraint_checks", {}).get("hard", 0)
            overrides = len(meta.get("override_history", []))
            if hard_violations > 0 or overrides > 0:
                exceptions.append({
                    "decision_id": d.get("decision_id"),
                    "portfolio_id": meta.get("portfolio_id"),
                    "timestamp": meta.get("timestamp"),
                    "hard_violations": hard_violations,
                    "overrides": overrides,
                    "exception_type": "constraint_violation" if hard_violations > 0 else "advisor_override",
                })
        return pd.DataFrame(exceptions) if exceptions else pd.DataFrame()
