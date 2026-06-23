# ADR 003: SHAP over LIME as Primary Explainability Method

**Status:** Accepted  
**Date:** 2025-03-01

## Decision

Use **SHAP (Tree SHAP)** as the primary explainability method, with **LIME** as a complementary local explainer.

## Rationale

- Tree SHAP provides **exact** Shapley values (not approximations) for tree-based surrogate models
- SHAP values satisfy game-theoretic axioms: efficiency, symmetry, dummy, linearity
- Compliance documentation requires exact attribution, not approximations
- LIME complements SHAP for client/advisor explanations (simpler, feature-based)
- XGBoost surrogate model trained on drift/trigger features is a natural fit for TreeExplainer

## Consequences

- Requires training a surrogate model on historical decisions
- SHAP plots (waterfall, summary) add visual explainability to compliance records
- LIME adds diversity of explanation style for different audiences
