# ADR 002: CVXPY for Portfolio Optimisation

**Status:** Accepted  
**Date:** 2025-03-01

## Decision

Use **CVXPY** with the OSQP solver for constrained quadratic programming.

## Rationale

- Domain-specific language for convex optimisation — constraints expressed naturally
- OSQP solver: fast (< 2s for 50-asset portfolios), warm-start capable
- SCS fallback for large-scale problems
- Handles QP objective (tracking error minimisation) + linear constraints directly
- `PyPortfolioOpt` used for prototyping; CVXPY for production with custom SEBI constraints

## Consequences

- Binary decision variables (min trade size) require post-processing rounding
- Must handle infeasibility gracefully when constraints conflict
