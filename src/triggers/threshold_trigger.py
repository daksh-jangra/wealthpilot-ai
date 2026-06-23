"""Threshold-based trigger evaluators: asset-class drift and concentration breaches."""

from __future__ import annotations

from src.monitoring.drift_calculator import DriftResult, DriftSeverity
from src.triggers.trigger_evaluator import (
    TriggerEvaluator,
    TriggerEvent,
    TriggerType,
)


class ThresholdTrigger(TriggerEvaluator):
    """Fires when asset-class drift exceeds the portfolio's drift band."""

    def evaluate(self, portfolio_id: str, context: dict) -> TriggerEvent | None:
        drift_result: DriftResult | None = context.get("drift_result")
        if drift_result is None or drift_result.severity == DriftSeverity.NONE:
            return None

        severity_map = {
            DriftSeverity.CRITICAL: TriggerType.THRESHOLD_CONCENTRATION,
            DriftSeverity.HIGH: TriggerType.THRESHOLD_ASSET_CLASS,
            DriftSeverity.MEDIUM: TriggerType.THRESHOLD_ASSET_CLASS,
            DriftSeverity.LOW: TriggerType.THRESHOLD_ASSET_CLASS,
        }
        trigger_type = severity_map[drift_result.severity]

        return self._make_event(
            portfolio_id=portfolio_id,
            trigger_type=trigger_type,
            details={
                "drift_severity": drift_result.severity.value,
                "max_drift": round(drift_result.max_drift, 4),
                "sum_abs_drift": round(drift_result.sum_abs_drift, 4),
                "rmsd": round(drift_result.rmsd, 4),
                "drift_band": drift_result.drift_band,
                "breaching_asset_classes": drift_result.breaching_asset_classes,
            },
        )


class ConcentrationTrigger(TriggerEvaluator):
    """Fires when a single issuer or sector breaches hard concentration limits."""

    SECTOR_LIMIT = 0.35
    ISSUER_LIMIT = 0.10

    def evaluate(self, portfolio_id: str, context: dict) -> TriggerEvent | None:
        sector_weights: dict[str, float] = context.get("sector_weights", {})
        issuer_weights: dict[str, float] = context.get("issuer_weights", {})

        violations = {}
        for sector, w in sector_weights.items():
            if w > self.SECTOR_LIMIT:
                violations[f"sector:{sector}"] = round(w, 4)
        for issuer, w in issuer_weights.items():
            if w > self.ISSUER_LIMIT:
                violations[f"issuer:{issuer}"] = round(w, 4)

        if not violations:
            return None

        return self._make_event(
            portfolio_id=portfolio_id,
            trigger_type=TriggerType.THRESHOLD_CONCENTRATION,
            details={
                "violations": violations,
                "sector_limit": self.SECTOR_LIMIT,
                "issuer_limit": self.ISSUER_LIMIT,
            },
        )


class FactorExposureTrigger(TriggerEvaluator):
    """Fires when factor exposure (beta, value, momentum) drifts beyond 1.5 std devs."""

    STD_THRESHOLD = 1.5

    def evaluate(self, portfolio_id: str, context: dict) -> TriggerEvent | None:
        factor_tilts: dict[str, float] = context.get("factor_tilts", {})
        if not factor_tilts:
            return None

        breaching = {k: round(v, 3) for k, v in factor_tilts.items() if abs(v) > self.STD_THRESHOLD}
        if not breaching:
            return None

        return self._make_event(
            portfolio_id=portfolio_id,
            trigger_type=TriggerType.THRESHOLD_FACTOR,
            details={
                "breaching_factors": breaching,
                "threshold_std": self.STD_THRESHOLD,
            },
        )
