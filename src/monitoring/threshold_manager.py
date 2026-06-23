"""Maintain risk-category and client-overlay drift thresholds."""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Optional

BASE_DRIFT_BANDS = {
    "ultra_conservative": 0.020,
    "conservative": 0.025,
    "balanced": 0.030,
    "aggressive": 0.040,
    "ultra_aggressive": 0.050,
}

FACTOR_DRIFT_STD_DEFAULT = 1.5


class ThresholdManager:
    """
    Load and resolve per-portfolio drift thresholds, combining category-level
    bands with client-specific overlays (ESG screens, restrictions, etc.).
    """

    def __init__(self, config_path: Optional[Path] = None):
        self._base_bands = dict(BASE_DRIFT_BANDS)
        self._client_overrides: dict[str, float] = {}
        if config_path and config_path.exists():
            self._load_config(config_path)

    def _load_config(self, path: Path) -> None:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        bands = cfg.get("drift_thresholds", {})
        for k, v in bands.items():
            if k in self._base_bands:
                self._base_bands[k] = float(v)

    def get_drift_band(self, risk_category: str, client_id: Optional[str] = None) -> float:
        """Return the effective drift band for a client."""
        base = self._base_bands.get(risk_category, 0.030)
        if client_id and client_id in self._client_overrides:
            return self._client_overrides[client_id]
        return base

    def set_client_override(self, client_id: str, band: float) -> None:
        """Set a custom drift band for a specific client (e.g., restricted portfolios)."""
        if band <= 0 or band > 0.20:
            raise ValueError(f"Drift band {band} out of valid range (0, 0.20]")
        self._client_overrides[client_id] = band

    def remove_client_override(self, client_id: str) -> None:
        self._client_overrides.pop(client_id, None)

    def get_all_bands(self, risk_categories: dict[str, str]) -> dict[str, float]:
        """Bulk-resolve drift bands for {portfolio_id: risk_category} mapping."""
        return {pid: self.get_drift_band(cat, pid) for pid, cat in risk_categories.items()}

    def tighten_bands(self, factor: float = 0.80) -> None:
        """Tighten all bands (e.g., during stressed market regime)."""
        for k in self._base_bands:
            self._base_bands[k] *= factor

    def restore_bands(self) -> None:
        self._base_bands = dict(BASE_DRIFT_BANDS)
