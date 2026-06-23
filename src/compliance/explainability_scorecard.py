"""Rate each explanation on accuracy, completeness, readability, actionability, and regulatory sufficiency."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass
class ExplainabilityScore:
    explanation_id: str
    audience: str
    accuracy_score: float        # 0-1: numbers in narrative match metadata
    completeness_score: float    # 0-1: all required sections present
    readability_score: float     # 0-1: grade level appropriate
    actionability_score: float   # 0-1: client knows what to do
    regulatory_score: float      # 0-1: meets compliance requirements
    overall_score: float         # weighted average
    passed: bool

    WEIGHTS = {
        "accuracy": 0.25,
        "completeness": 0.25,
        "readability": 0.20,
        "actionability": 0.15,
        "regulatory": 0.15,
    }


class ExplainabilityScorecard:
    """
    Evaluate explanation quality across all five dimensions.
    Minimum overall score to pass: 0.75.
    """

    PASS_THRESHOLD = 0.75

    def score(
        self,
        explanation_id: str,
        audience: str,
        narrative: str,
        metadata: dict,
    ) -> ExplainabilityScore:
        accuracy = self._accuracy(narrative, metadata)
        completeness = self._completeness(narrative, audience)
        readability = self._readability(narrative, audience)
        actionability = self._actionability(narrative, audience)
        regulatory = self._regulatory_sufficiency(narrative, audience)

        weights = ExplainabilityScore.WEIGHTS
        overall = (
            weights["accuracy"] * accuracy
            + weights["completeness"] * completeness
            + weights["readability"] * readability
            + weights["actionability"] * actionability
            + weights["regulatory"] * regulatory
        )

        return ExplainabilityScore(
            explanation_id=explanation_id,
            audience=audience,
            accuracy_score=round(accuracy, 2),
            completeness_score=round(completeness, 2),
            readability_score=round(readability, 2),
            actionability_score=round(actionability, 2),
            regulatory_score=round(regulatory, 2),
            overall_score=round(overall, 2),
            passed=overall >= self.PASS_THRESHOLD,
        )

    def _accuracy(self, narrative: str, metadata: dict) -> float:
        """Check if key numbers appear in the explanation."""
        score = 1.0
        cost = metadata.get("total_cost_inr", 0)
        drift = metadata.get("max_drift_pct", 0)

        if cost > 0:
            cost_str = str(int(cost // 1000))
            if cost_str not in narrative:
                score -= 0.3

        if drift > 0:
            drift_str = f"{drift:.0f}"
            drift_str2 = f"{drift:.1f}"
            if drift_str not in narrative and drift_str2 not in narrative:
                score -= 0.2

        return max(0.0, score)

    def _completeness(self, narrative: str, audience: str) -> float:
        lower = narrative.lower()
        required = {
            "client": ["market", "portfolio", "cost", "rebalanc"],
            "advisor": ["drift", "tracking", "trade", "cost"],
            "compliance": ["decision", "constraint", "trigger", "audit"],
        }.get(audience, [])

        if not required:
            return 1.0
        found = sum(1 for kw in required if kw in lower)
        return found / len(required)

    def _readability(self, narrative: str, audience: str) -> float:
        words = narrative.split()
        sentences = [s for s in re.split(r"[.!?]", narrative) if s.strip()]
        if not sentences:
            return 0.5
        avg_words = len(words) / len(sentences)
        fk_grade = 0.39 * avg_words + 11.8 * 1.5 - 15.59

        targets = {"client": 9, "advisor": 14, "compliance": 20}
        target = targets.get(audience, 14)
        if fk_grade <= target:
            return 1.0
        elif fk_grade <= target + 3:
            return 0.75
        else:
            return 0.50

    def _actionability(self, narrative: str, audience: str) -> float:
        if audience != "client":
            return 1.0  # Actionability only evaluated for client explanations
        lower = narrative.lower()
        action_words = ["will", "are", "rebalance", "restore", "cost"]
        found = sum(1 for w in action_words if w in lower)
        return min(1.0, found / len(action_words))

    def _regulatory_sufficiency(self, narrative: str, audience: str) -> float:
        if audience != "compliance":
            return 1.0
        lower = narrative.lower()
        required = ["decision", "audit", "constraint", "sebi", "model"]
        found = sum(1 for kw in required if kw in lower)
        return found / len(required)

    def batch_score(
        self,
        explanations: list[dict],
    ) -> list[ExplainabilityScore]:
        """Score a batch of explanations."""
        scores = []
        for i, item in enumerate(explanations):
            for audience in ("client", "advisor", "compliance"):
                expl = item.get("explanations", {}).get(audience, {})
                narrative = expl.get("narrative", "") if isinstance(expl, dict) else ""
                score = self.score(
                    explanation_id=f"{item.get('decision_id', i)}_{audience}",
                    audience=audience,
                    narrative=narrative,
                    metadata=item.get("decision_metadata", {}),
                )
                scores.append(score)
        return scores
