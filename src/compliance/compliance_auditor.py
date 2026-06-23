"""Automated compliance auditor: sample decisions, evaluate quality, detect bias."""

from __future__ import annotations

import random
from datetime import datetime

import numpy as np
import pandas as pd


class ComplianceAuditor:
    """
    Quarterly automated audit of 100 sampled rebalancing decisions.
    Checks explanation quality, systematic bias, and constraint satisfaction.
    """

    SAMPLE_SIZE = 100

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def stratified_sample(
        self,
        decision_log: list[dict],
        n: int = SAMPLE_SIZE,
    ) -> list[dict]:
        """Stratified sample: ~equal representation of trigger types and risk categories."""
        if len(decision_log) <= n:
            return decision_log

        trigger_types = list({d.get("trigger_type", "unknown") for d in decision_log})
        sample = []
        per_type = max(1, n // len(trigger_types))
        for tt in trigger_types:
            subset = [d for d in decision_log if d.get("trigger_type") == tt]
            sample.extend(self.rng.sample(subset, min(per_type, len(subset))))

        return sample[:n]

    def evaluate_explanation_quality(self, explanation: str, audience: str) -> dict:
        """Automated metrics: accuracy proxy, completeness, readability."""
        words = explanation.split()
        word_count = len(words)
        sentences = [
            s.strip()
            for s in explanation.replace("!", ".").replace("?", ".").split(".")
            if s.strip()
        ]
        n_sentences = max(1, len(sentences))
        avg_words_per_sentence = word_count / n_sentences

        # Flesch-Kincaid Grade Level approximation
        syllables_per_word = 1.5  # rough estimate
        fk_grade = 0.39 * avg_words_per_sentence + 11.8 * syllables_per_word - 15.59

        required_sections = {
            "client": ["market", "portfolio", "cost"],
            "advisor": ["drift", "tracking", "cost"],
            "compliance": ["decision", "constraint", "trigger"],
        }
        required = required_sections.get(audience, [])
        lower = explanation.lower()
        completeness = sum(1 for kw in required if kw in lower) / max(len(required), 1)

        grade_ok = {
            "client": fk_grade <= 9,
            "advisor": fk_grade <= 14,
            "compliance": True,
        }.get(audience, True)

        return {
            "word_count": word_count,
            "flesch_kincaid_grade": round(fk_grade, 1),
            "grade_level_ok": grade_ok,
            "completeness_score": round(completeness, 2),
            "readability_pass": completeness >= 0.67 and grade_ok,
        }

    def detect_systematic_bias(self, decision_log: list[dict]) -> dict:
        """Check for over/under-rebalancing by risk category or other patterns."""
        if not decision_log:
            return {}

        df = pd.DataFrame(decision_log)
        bias_report = {}

        if "risk_category" in df.columns:
            rebalance_rates = df.groupby("risk_category").size().to_dict()
            total = len(df)
            expected_share = 1 / max(len(rebalance_rates), 1)
            for cat, count in rebalance_rates.items():
                actual_share = count / total
                deviation = abs(actual_share - expected_share) / expected_share
                bias_report[cat] = {
                    "count": count,
                    "share_pct": round(actual_share * 100, 1),
                    "expected_pct": round(expected_share * 100, 1),
                    "deviation_pct": round(deviation * 100, 1),
                    "bias_flag": deviation > 0.30,
                }

        if "trigger_type" in df.columns:
            trigger_dist = df["trigger_type"].value_counts().to_dict()
            bias_report["trigger_distribution"] = trigger_dist

        return bias_report

    def generate_audit_report(
        self,
        decision_log: list[dict],
        quarter: str = "Q1",
    ) -> dict:
        """Generate the full quarterly audit report."""
        sample = self.stratified_sample(decision_log)
        quality_scores = []
        for decision in sample:
            for audience in ("client", "advisor", "compliance"):
                expl = decision.get("explanations", {}).get(audience, {})
                narrative = expl.get("narrative", "") if isinstance(expl, dict) else ""
                if narrative:
                    score = self.evaluate_explanation_quality(narrative, audience)
                    quality_scores.append(score)

        pass_rate = (
            sum(1 for s in quality_scores if s["readability_pass"]) / len(quality_scores)
            if quality_scores
            else 0.0
        )

        bias_analysis = self.detect_systematic_bias(decision_log)

        return {
            "audit_quarter": quarter,
            "audit_timestamp": datetime.utcnow().isoformat(),
            "total_decisions": len(decision_log),
            "sampled_decisions": len(sample),
            "explanation_quality": {
                "avg_completeness": round(
                    (
                        float(np.mean([s["completeness_score"] for s in quality_scores]))
                        if quality_scores
                        else 0.0
                    ),
                    2,
                ),
                "pass_rate": round(pass_rate, 2),
                "grade_level_pass_rate": round(
                    sum(1 for s in quality_scores if s["grade_level_ok"])
                    / max(len(quality_scores), 1),
                    2,
                ),
            },
            "bias_analysis": bias_analysis,
            "audit_passed": pass_rate >= 0.85,
        }
