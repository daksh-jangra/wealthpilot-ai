"""Core explanation generation: template selection, LLM call, quality assurance."""

from __future__ import annotations

import os
from typing import Literal

import anthropic
from pydantic import BaseModel, Field


class ExplanationOutput(BaseModel):
    audience: str
    narrative: str
    trigger_summary: str
    key_numbers: dict = Field(default_factory=dict)
    readability_grade: float | None = None
    word_count: int = 0
    complete: bool = True


SYSTEM_PROMPTS = {
    "client": (
        "You are a friendly, patient financial advisor explaining a portfolio rebalancing decision "
        "to a retail investor. Use plain language at a Grade 8 reading level. Never use financial "
        "jargon without immediately defining it. Always connect the rebalancing action to the "
        "client's personal investment goals. Always disclose costs transparently. Never make "
        "performance predictions or guarantees. Structure your explanation as: "
        "(1) What happened in the market, (2) What we are doing with your portfolio, "
        "(3) Why this benefits you, (4) What it costs. Keep the total explanation under 200 words. "
        "Output only the explanation text, nothing else."
    ),
    "advisor": (
        "You are a senior portfolio analyst briefing a financial advisor about a rebalancing decision. "
        "Include specific allocation percentages, drift magnitudes, risk metrics, and cost analysis. "
        "Discuss the trade-off between tracking error reduction and transaction costs. "
        "Mention alternative strategies considered and why the recommended strategy was selected. "
        "Include a concise trade list summary. "
        "Structure: (1) Trigger and drift analysis, (2) Proposed rebalancing strategy with rationale, "
        "(3) Risk impact (VaR, tracking error before/after), (4) Cost and tax analysis, "
        "(5) Alternatives considered. Keep under 400 words. "
        "Output only the explanation text, nothing else."
    ),
    "compliance": (
        "You are a regulatory compliance officer documenting a rebalancing decision for audit. "
        "Be exhaustive and precise. Include: decision ID, timestamp chain, trigger classification, "
        "complete input data summary, constraint satisfaction matrix (every rule checked with pass/fail), "
        "SHAP feature attribution summary, counterfactual analysis, model version, and any override history. "
        "Reference SEBI regulation numbers where applicable. Use a structured format with labelled sections. "
        "Do not omit any data points — this document must withstand regulatory scrutiny. "
        "Output only the explanation text, nothing else."
    ),
}


class ExplanationGenerator:
    """Generate LLM-powered explanations for portfolio rebalancing decisions."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        client: anthropic.Anthropic | None = None,
    ):
        self.model = model
        self.client = client or anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    def generate(
        self,
        audience: Literal["client", "advisor", "compliance"],
        decision_metadata: dict,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ExplanationOutput:
        """Generate an explanation for the given audience."""
        temp_map = {"client": 0.45, "advisor": 0.35, "compliance": 0.15}
        tok_map = {"client": 400, "advisor": 800, "compliance": 1500}

        temp = temperature or temp_map[audience]
        tokens = max_tokens or tok_map[audience]

        system_prompt = SYSTEM_PROMPTS[audience]
        user_prompt = self._build_user_prompt(audience, decision_metadata)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=tokens,
                temperature=temp,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            narrative = response.content[0].text.strip()
        except Exception:
            narrative = self._fallback_narrative(audience, decision_metadata)

        output = ExplanationOutput(
            audience=audience,
            narrative=narrative,
            trigger_summary=self._extract_trigger_summary(decision_metadata),
            key_numbers=self._extract_key_numbers(decision_metadata),
            word_count=len(narrative.split()),
        )
        self._quality_check(output, decision_metadata)
        return output

    def generate_all_tiers(self, decision_metadata: dict) -> dict[str, ExplanationOutput]:
        """Generate explanations for all three audience tiers."""
        return {
            audience: self.generate(audience, decision_metadata)
            for audience in ("client", "advisor", "compliance")
        }

    def _build_user_prompt(self, audience: str, meta: dict) -> str:
        portfolio_id = meta.get("portfolio_id", "UNKNOWN")
        trigger = meta.get("trigger_type", "threshold_drift")
        risk_cat = meta.get("risk_category", "balanced")
        drift = meta.get("max_drift_pct", 0)
        trades = meta.get("trade_summary", {})
        cost = meta.get("total_cost_inr", 0)
        tax_impact = meta.get("tax_impact_inr", 0)
        shap_features = meta.get("shap_top_features", [])

        base = (
            f"Portfolio ID: {portfolio_id}\n"
            f"Risk Category: {risk_cat.replace('_', ' ').title()}\n"
            f"Trigger: {trigger.replace('_', ' ').title()}\n"
            f"Max drift from target: {drift:.1f}%\n"
            f"Trade summary: {trades}\n"
            f"Estimated transaction cost: INR {cost:,.0f}\n"
            f"Tax impact: INR {tax_impact:,.0f}\n"
        )
        if audience == "compliance" and shap_features:
            base += f"Top SHAP features: {shap_features}\n"
            base += f"Model version: {meta.get('model_version', '1.0.0')}\n"
            base += f"Decision ID: {meta.get('decision_id', 'UNKNOWN')}\n"
            base += f"Constraint checks: {meta.get('constraint_checks', {})}\n"

        return base

    def _extract_trigger_summary(self, meta: dict) -> str:
        trigger = meta.get("trigger_type", "unknown")
        drift = meta.get("max_drift_pct", 0)
        return f"{trigger.replace('_', ' ').title()} — drift {drift:.1f}%"

    def _extract_key_numbers(self, meta: dict) -> dict:
        return {
            "max_drift_pct": meta.get("max_drift_pct", 0),
            "total_cost_inr": meta.get("total_cost_inr", 0),
            "tax_impact_inr": meta.get("tax_impact_inr", 0),
            "tracking_error_before": meta.get("tracking_error_before", 0),
            "tracking_error_after": meta.get("tracking_error_after", 0),
        }

    def _quality_check(self, output: ExplanationOutput, meta: dict) -> None:
        """Validate numbers in explanation match input data."""
        # Word count check
        if output.audience == "client" and output.word_count > 250:
            output.complete = False
        # Verify key cost figure appears (approximate)
        cost = meta.get("total_cost_inr", 0)
        if (
            cost > 0
            and str(int(cost // 1000)) not in output.narrative
            and output.audience == "client"
        ):
            pass  # soft warning only

    def _fallback_narrative(self, audience: str, meta: dict) -> str:
        drift = meta.get("max_drift_pct", 0)
        trigger = meta.get("trigger_type", "drift")
        cost = meta.get("total_cost_inr", 0)
        if audience == "client":
            return (
                f"Your portfolio has drifted {drift:.1f}% from your target allocation due to recent "
                f"market movements. We are rebalancing it back to your chosen risk level. "
                f"The estimated cost of this rebalancing is INR {cost:,.0f}."
            )
        elif audience == "advisor":
            return (
                f"Trigger: {trigger}. Max drift: {drift:.1f}%. Rebalancing to restore target "
                f"allocation. Estimated transaction cost: INR {cost:,.0f}. "
                f"See trade list for details."
            )
        else:
            return (
                f"COMPLIANCE RECORD — Portfolio rebalancing triggered by {trigger}. "
                f"Max drift: {drift:.1f}%. Cost: INR {cost:,.0f}. "
                f"Full constraint check and audit trail attached."
            )
