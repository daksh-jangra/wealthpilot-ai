"""Vectorised drift calculation for 50,000 portfolios in under 30 seconds."""

from __future__ import annotations

import heapq
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Optional

ASSET_CLASSES = [
    "indian_equity",
    "international_equity",
    "indian_fixed_income",
    "international_fixed_income",
    "alternatives",
    "cash",
]

TARGET_ALLOCATIONS = {
    "ultra_conservative": np.array([0.10, 0.05, 0.45, 0.15, 0.10, 0.15]),
    "conservative": np.array([0.20, 0.10, 0.35, 0.10, 0.12, 0.13]),
    "balanced": np.array([0.35, 0.15, 0.20, 0.10, 0.12, 0.08]),
    "aggressive": np.array([0.50, 0.20, 0.10, 0.05, 0.10, 0.05]),
    "ultra_aggressive": np.array([0.60, 0.25, 0.05, 0.00, 0.07, 0.03]),
}


class DriftSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftResult:
    portfolio_id: str
    risk_category: str
    current_weights: np.ndarray
    target_weights: np.ndarray
    abs_drift: np.ndarray  # per asset-class absolute drift
    max_drift: float  # largest single asset-class drift
    sum_abs_drift: float  # L1 norm
    rmsd: float  # root-mean-square drift
    drift_band: float  # category-specific threshold
    severity: DriftSeverity
    breaching_asset_classes: list[str]

    def to_dict(self) -> dict:
        return {
            "portfolio_id": self.portfolio_id,
            "risk_category": self.risk_category,
            "max_drift": round(self.max_drift, 4),
            "sum_abs_drift": round(self.sum_abs_drift, 4),
            "rmsd": round(self.rmsd, 4),
            "drift_band": self.drift_band,
            "severity": self.severity.value,
            "breaching_asset_classes": self.breaching_asset_classes,
        }


DRIFT_BANDS = {
    "ultra_conservative": 0.020,
    "conservative": 0.025,
    "balanced": 0.030,
    "aggressive": 0.040,
    "ultra_aggressive": 0.050,
}

SEVERITY_MULTIPLIERS = {
    DriftSeverity.CRITICAL: 2.0,
    DriftSeverity.HIGH: 1.5,
    DriftSeverity.MEDIUM: 1.0,
    DriftSeverity.LOW: 0.5,
    DriftSeverity.NONE: 0.0,
}


class DriftCalculator:
    """
    Compute portfolio drift metrics using vectorised NumPy operations.
    Designed to process 50,000 portfolios in a single batch efficiently.
    """

    def compute_batch(
        self,
        current_weights: pd.DataFrame,
        risk_categories: pd.Series,
        drift_bands: Optional[dict[str, float]] = None,
    ) -> list[DriftResult]:
        """
        Vectorised batch drift calculation.

        Args:
            current_weights: DataFrame (n_portfolios, 6) of current asset weights
            risk_categories: Series mapping portfolio_id -> risk_category
            drift_bands: override per-category drift bands

        Returns:
            List of DriftResult, sorted by max_drift descending.
        """
        bands = drift_bands or DRIFT_BANDS

        # Build target matrix (n_portfolios, 6) vectorised
        cats = risk_categories.values
        target_matrix = np.vstack([TARGET_ALLOCATIONS[c] for c in cats])
        current_matrix = current_weights.values

        # Core drift metrics (all vectorised)
        abs_drift_matrix = np.abs(current_matrix - target_matrix)
        max_drift_vec = abs_drift_matrix.max(axis=1)
        sum_abs_drift_vec = abs_drift_matrix.sum(axis=1)
        rmsd_vec = np.sqrt((abs_drift_matrix**2).mean(axis=1))

        # Per-portfolio drift bands
        band_vec = np.array([bands[c] for c in cats])

        # Severity classification
        ratio = max_drift_vec / np.where(band_vec == 0, 1e-9, band_vec)
        severities = np.where(
            ratio >= SEVERITY_MULTIPLIERS[DriftSeverity.CRITICAL],
            DriftSeverity.CRITICAL.value,
            np.where(
                ratio >= SEVERITY_MULTIPLIERS[DriftSeverity.HIGH],
                DriftSeverity.HIGH.value,
                np.where(
                    ratio >= SEVERITY_MULTIPLIERS[DriftSeverity.MEDIUM],
                    DriftSeverity.MEDIUM.value,
                    np.where(
                        ratio >= SEVERITY_MULTIPLIERS[DriftSeverity.LOW],
                        DriftSeverity.LOW.value,
                        DriftSeverity.NONE.value,
                    ),
                ),
            ),
        )

        portfolio_ids = current_weights.index.tolist()
        results = []
        for i, pid in enumerate(portfolio_ids):
            cat = cats[i]
            band = band_vec[i]
            ad = abs_drift_matrix[i]
            breaching = [ASSET_CLASSES[j] for j in range(6) if ad[j] > band]

            results.append(
                DriftResult(
                    portfolio_id=pid,
                    risk_category=cat,
                    current_weights=current_matrix[i].copy(),
                    target_weights=target_matrix[i].copy(),
                    abs_drift=ad.copy(),
                    max_drift=float(max_drift_vec[i]),
                    sum_abs_drift=float(sum_abs_drift_vec[i]),
                    rmsd=float(rmsd_vec[i]),
                    drift_band=float(band),
                    severity=DriftSeverity(severities[i]),
                    breaching_asset_classes=breaching,
                )
            )

        # Return sorted by severity (critical first)
        severity_order = {
            DriftSeverity.CRITICAL.value: 0,
            DriftSeverity.HIGH.value: 1,
            DriftSeverity.MEDIUM.value: 2,
            DriftSeverity.LOW.value: 3,
            DriftSeverity.NONE.value: 4,
        }
        results.sort(key=lambda r: (severity_order[r.severity.value], -r.max_drift))
        return results

    def compute_single(
        self,
        portfolio_id: str,
        current_weights: np.ndarray,
        risk_category: str,
        drift_band: Optional[float] = None,
    ) -> DriftResult:
        """Compute drift for a single portfolio."""
        target = TARGET_ALLOCATIONS[risk_category]
        band = drift_band or DRIFT_BANDS[risk_category]
        ad = np.abs(current_weights - target)
        max_drift = float(ad.max())
        ratio = max_drift / band if band > 0 else 0
        if ratio >= 2.0:
            severity = DriftSeverity.CRITICAL
        elif ratio >= 1.5:
            severity = DriftSeverity.HIGH
        elif ratio >= 1.0:
            severity = DriftSeverity.MEDIUM
        elif ratio >= 0.5:
            severity = DriftSeverity.LOW
        else:
            severity = DriftSeverity.NONE

        breaching = [ASSET_CLASSES[j] for j in range(6) if ad[j] > band]
        return DriftResult(
            portfolio_id=portfolio_id,
            risk_category=risk_category,
            current_weights=current_weights.copy(),
            target_weights=target.copy(),
            abs_drift=ad,
            max_drift=max_drift,
            sum_abs_drift=float(ad.sum()),
            rmsd=float(np.sqrt((ad**2).mean())),
            drift_band=band,
            severity=severity,
            breaching_asset_classes=breaching,
        )

    def build_priority_queue(self, results: list[DriftResult]) -> list[tuple]:
        """Build a max-heap of (max_drift, portfolio_id) for O(log n) retrieval."""
        heap = [(-r.max_drift, r.portfolio_id) for r in results if r.severity != DriftSeverity.NONE]
        heapq.heapify(heap)
        return heap

    def results_to_dataframe(self, results: list[DriftResult]) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in results])
