"""Client-level explanation builder: plain language, Grade 8, goal-linked."""

from __future__ import annotations
from src.explainability.explanation_generator import ExplanationGenerator, ExplanationOutput


class ClientExplainer:
    """Produces client-facing rebalancing explanations."""

    ANALOGIES = {
        "threshold": (
            "Think of rebalancing like trimming a garden — we cut back the fast-growing plants "
            "and give the slower ones more room, keeping your garden in the shape you designed."
        ),
        "calendar": "This is your scheduled portfolio health check, like a regular maintenance service.",
        "event": "A significant event requires us to adjust your portfolio to keep you on track.",
    }

    def __init__(self, generator: ExplanationGenerator):
        self.generator = generator

    def explain(self, decision_metadata: dict) -> ExplanationOutput:
        trigger_type = decision_metadata.get("trigger_type", "threshold")
        trigger_class = "threshold" if "threshold" in trigger_type else (
            "calendar" if "calendar" in trigger_type else "event"
        )
        decision_metadata["analogy"] = self.ANALOGIES.get(trigger_class, self.ANALOGIES["threshold"])
        return self.generator.generate("client", decision_metadata)

    def format_email(self, explanation: ExplanationOutput, client_name: str = "Valued Client") -> str:
        return (
            f"Dear {client_name},\n\n"
            f"{explanation.narrative}\n\n"
            f"If you have any questions, please contact your advisor.\n\n"
            f"Warm regards,\nWealthPilot AI"
        )
