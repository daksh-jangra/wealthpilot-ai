"""Synthetic market data generator using multivariate geometric Brownian motion."""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MarketParameters:
    """Calibrated parameters for Indian market simulation."""

    asset_classes: list[str] = field(
        default_factory=lambda: [
            "indian_equity",
            "international_equity",
            "indian_fixed_income",
            "international_fixed_income",
            "alternatives",
            "cash",
        ]
    )

    # Annualised expected returns
    expected_returns: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                0.14,  # indian_equity
                0.10,  # international_equity
                0.07,  # indian_fixed_income
                0.05,  # international_fixed_income
                0.09,  # alternatives (gold, REITs)
                0.065,  # cash/liquid
            ]
        )
    )

    # Annualised volatilities
    volatilities: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                0.20,  # indian_equity
                0.15,  # international_equity
                0.04,  # indian_fixed_income
                0.05,  # international_fixed_income
                0.12,  # alternatives
                0.005,  # cash
            ]
        )
    )

    # Correlation matrix (calibrated to NSE/BSE/global data)
    correlation_matrix: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [1.00, 0.60, -0.10, -0.05, 0.30, 0.00],
                [0.60, 1.00, -0.05, -0.02, 0.25, 0.00],
                [-0.10, -0.05, 1.00, 0.70, -0.15, 0.10],
                [-0.05, -0.02, 0.70, 1.00, -0.10, 0.05],
                [0.30, 0.25, -0.15, -0.10, 1.00, 0.00],
                [0.00, 0.00, 0.10, 0.05, 0.00, 1.00],
            ]
        )
    )

    trading_days: int = 252
    degrees_of_freedom: int = 6  # Student-t for fat tails


class MarketDataSimulator:
    """Generate synthetic market return time series."""

    def __init__(self, params: Optional[MarketParameters] = None, seed: int = 42):
        self.params = params or MarketParameters()
        self.rng = np.random.default_rng(seed)
        self._validate_params()

    def _validate_params(self) -> None:
        corr = self.params.correlation_matrix
        assert corr.shape == (6, 6), "Correlation matrix must be 6x6"
        eigenvalues = np.linalg.eigvalsh(corr)
        if eigenvalues.min() < -1e-8:
            raise ValueError("Correlation matrix is not positive semi-definite")

    def simulate_returns(self, num_days: Optional[int] = None) -> pd.DataFrame:
        """Generate daily log-returns for all asset classes."""
        days = num_days or self.params.trading_days
        n = len(self.params.asset_classes)

        # Daily drift and vol
        mu_daily = self.params.expected_returns / self.params.trading_days
        sigma_daily = self.params.volatilities / np.sqrt(self.params.trading_days)

        # Cholesky decomposition for correlated draws
        cov_matrix = np.outer(sigma_daily, sigma_daily) * self.params.correlation_matrix
        L = np.linalg.cholesky(cov_matrix)

        # Student-t innovations for fat tails
        chi2 = self.rng.chisquare(self.params.degrees_of_freedom, size=days)
        scale = np.sqrt(self.params.degrees_of_freedom / chi2)
        z = self.rng.standard_normal(size=(days, n))
        innovations = (z * scale[:, None]) @ L.T

        # GBM daily log-returns
        returns = mu_daily - 0.5 * sigma_daily**2 + innovations

        # Simulate market crash at day ~60 (scenario 2)
        crash_day = 60
        if days > crash_day + 5:
            crash_returns = np.array([-0.05, -0.04, 0.01, 0.01, -0.02, 0.0])
            for d in range(crash_day, crash_day + 5):
                returns[d] = crash_returns

        dates = pd.bdate_range(start="2024-04-01", periods=days, freq="B")
        return pd.DataFrame(returns, index=dates, columns=self.params.asset_classes)

    def simulate_price_levels(self, base_values: Optional[dict] = None) -> pd.DataFrame:
        """Cumulative price index from simulated returns."""
        returns = self.simulate_returns()
        if base_values is None:
            base_values = {ac: 100.0 for ac in self.params.asset_classes}
        base = pd.Series(base_values)
        prices = (1 + returns).cumprod() * base
        return prices

    def get_vix_series(self, num_days: Optional[int] = None) -> pd.Series:
        """Simulate a VIX-like volatility index."""
        days = num_days or self.params.trading_days
        # Mean-reverting VIX: Ornstein-Uhlenbeck process
        mean_vix = 18.0
        speed = 0.15
        vol_vix = 5.0
        dt = 1 / self.params.trading_days
        vix = np.zeros(days)
        vix[0] = mean_vix
        for t in range(1, days):
            dW = self.rng.standard_normal()
            vix[t] = vix[t - 1] + speed * (mean_vix - vix[t - 1]) * dt + vol_vix * np.sqrt(dt) * dW
            vix[t] = max(vix[t], 5.0)

        # Spike during crash
        crash_day = 60
        if days > crash_day + 5:
            vix[crash_day : crash_day + 5] *= 2.5

        dates = pd.bdate_range(start="2024-04-01", periods=days, freq="B")
        return pd.Series(vix, index=dates, name="VIX")

    def get_nifty_returns(self, num_days: Optional[int] = None) -> pd.Series:
        """Return the Indian equity component as the Nifty 50 proxy."""
        returns = self.simulate_returns(num_days)
        return returns["indian_equity"].rename("NIFTY_50")
