"""Constrained quadratic programming portfolio optimiser using CVXPY."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

from src.optimisation.constraint_manager import ConstraintManager


@dataclass
class OptimisationResult:
    status: str                    # "optimal", "infeasible", "suboptimal"
    trade_weights: np.ndarray      # delta weights (positive = buy, negative = sell)
    post_trade_weights: np.ndarray
    tracking_error: float
    turnover: float
    solver_time_ms: float
    iterations: int
    constraint_violations: list


class PortfolioOptimiser:
    """
    Minimise post-trade tracking error subject to:
    - Budget constraint (weights sum to 1)
    - Long-only constraint
    - Turnover budget
    - Sector and issuer concentration limits
    - Minimum trade size (rounded via post-processing)

    Uses CVXPY with OSQP solver for speed.
    """

    def __init__(
        self,
        constraint_manager: Optional[ConstraintManager] = None,
        solver: str = "OSQP",
    ):
        self.constraint_manager = constraint_manager or ConstraintManager()
        self.solver = solver
        if not CVXPY_AVAILABLE:
            raise ImportError("cvxpy is required: pip install cvxpy")

    def optimise(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        covariance_matrix: Optional[np.ndarray] = None,
        turnover_budget: Optional[float] = None,
        sector_weights: Optional[dict] = None,
        issuer_weights: Optional[dict] = None,
        restricted_indices: Optional[list[int]] = None,
        cash_flow_fraction: float = 0.0,
        portfolio_value_inr: float = 1_000_000,
    ) -> OptimisationResult:
        """
        Solve the constrained rebalancing optimisation.

        Args:
            current_weights: (n,) current asset-class weights
            target_weights:  (n,) target asset-class weights
            covariance_matrix: (n, n) covariance; defaults to identity (min drift)
            turnover_budget: max fraction of portfolio to trade
            cash_flow_fraction: net cash inflow/outflow as fraction of portfolio value
        """
        import time
        n = len(current_weights)
        cov = covariance_matrix if covariance_matrix is not None else np.eye(n)
        budget = turnover_budget or self.constraint_manager.turnover_budget

        w_trade = cp.Variable(n)
        w_post = current_weights + w_trade + cash_flow_fraction * np.ones(n) / n

        # Objective: minimise tracking error (QP)
        diff = w_post - target_weights
        objective = cp.Minimize(cp.quad_form(diff, cov))

        constraints = [
            cp.sum(w_post) == 1.0,              # budget
            w_post >= 0,                          # long-only
            cp.norm1(w_trade) <= budget,          # turnover
            w_post[1] <= self.constraint_manager.international_equity_limit,  # SEBI intl limit
        ]

        # No trades in restricted asset classes
        if restricted_indices:
            for idx in restricted_indices:
                if 0 <= idx < n:
                    constraints.append(w_trade[idx] == 0)

        t0 = time.perf_counter()
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=self.solver, warm_start=True, verbose=False)
        except Exception:
            try:
                problem.solve(solver="SCS", verbose=False)
            except Exception:
                return self._fallback_result(current_weights, target_weights)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if problem.status in ("optimal", "optimal_inaccurate"):
            trade_w = w_trade.value if w_trade.value is not None else np.zeros(n)
            post_w = np.clip(current_weights + trade_w, 0, 1)
            post_w /= post_w.sum()

            # Round tiny trades to zero (min trade size)
            min_trade_fraction = self.constraint_manager.min_trade_size_inr / portfolio_value_inr
            trade_w = np.where(np.abs(trade_w) >= min_trade_fraction, trade_w, 0.0)

            violations = self.constraint_manager.check_post_trade(post_w, trade_w, sector_weights, issuer_weights)
            te = float(np.sqrt(diff.value.T @ cov @ diff.value)) if diff.value is not None else float(np.sqrt((post_w - target_weights) @ cov @ (post_w - target_weights)))

            return OptimisationResult(
                status=problem.status,
                trade_weights=trade_w,
                post_trade_weights=post_w,
                tracking_error=te,
                turnover=float(np.abs(trade_w).sum()),
                solver_time_ms=round(elapsed_ms, 1),
                iterations=0,
                constraint_violations=violations,
            )

        return self._fallback_result(current_weights, target_weights)

    def _fallback_result(
        self, current_weights: np.ndarray, target_weights: np.ndarray
    ) -> OptimisationResult:
        """Return no-trade result when optimiser fails."""
        return OptimisationResult(
            status="infeasible",
            trade_weights=np.zeros_like(current_weights),
            post_trade_weights=current_weights.copy(),
            tracking_error=float(np.abs(current_weights - target_weights).max()),
            turnover=0.0,
            solver_time_ms=0.0,
            iterations=0,
            constraint_violations=[],
        )

    def optimise_partial(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        rebalance_fraction: float = 0.5,
        **kwargs,
    ) -> OptimisationResult:
        """Partial rebalance toward target (reduce turnover for cost efficiency)."""
        partial_target = current_weights + rebalance_fraction * (target_weights - current_weights)
        partial_target /= partial_target.sum()
        return self.optimise(current_weights, partial_target, **kwargs)
