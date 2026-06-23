"""Advisor-level explanation builder: quantitative, with strategy alternatives."""

from __future__ import annotations
from src.explainability.explanation_generator import ExplanationGenerator, ExplanationOutput


class AdvisorExplainer:
    """Produces advisor-facing rebalancing explanations with full quant detail."""

    def __init__(self, generator: ExplanationGenerator):
        self.generator = generator

    def explain(self, decision_metadata: dict) -> ExplanationOutput:
        return self.generator.generate("advisor", decision_metadata)

    def format_dashboard_widget(self, explanation: ExplanationOutput) -> dict:
        """Format for the advisor dashboard widget."""
        return {
            "trigger": explanation.trigger_summary,
            "narrative": explanation.narrative,
            "key_metrics": explanation.key_numbers,
            "word_count": explanation.word_count,
        }
