# System Architecture

## Overview

The Autonomous Portfolio Rebalancing Agent is a modular, event-driven system with six core layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Dashboard                          │
│     Portfolio Overview | Activity | Analytics | XAI | Health    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│              Multi-Agent Orchestration (CrewAI)                 │
│  Orchestrator → Portfolio Analyst → Risk Manager →              │
│  Tax Specialist → Compliance Officer → Explanation Writer        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼────────────────────────┐
        │                  │                         │
┌───────▼──────┐  ┌────────▼────────┐  ┌────────────▼──────────┐
│  Monitoring  │  │   Optimisation  │  │   Explainability       │
│  Drift Calc  │  │   CVXPY/QP      │  │   SHAP + LIME          │
│  Thresholds  │  │   Tax Lots      │  │   Counterfactuals      │
│  DriftMonitor│  │   Liquidity     │  │   3-Tier Explanations  │
└───────┬──────┘  └────────┬────────┘  └────────────┬──────────┘
        │                  │                         │
┌───────▼──────────────────▼─────────────────────────▼──────────┐
│                    Trigger System                               │
│   Threshold | Calendar | Event-Driven | Consolidator            │
└──────────────────────────┬─────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                     Data Layer                                   │
│   50,000 Portfolios | Market Data | Client Profiles | Tax Lots  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Market data arrives** → `MarketDataSimulator` generates returns
2. **Drift scan** → `DriftCalculator.compute_batch()` vectorised across 50k portfolios
3. **Trigger evaluation** → `TriggerConsolidator` merges multiple simultaneous triggers
4. **Agent crew** → `RebalancingOrchestrator` dispatches to specialist agents
5. **Optimisation** → `PortfolioOptimiser` (CVXPY) solves constrained QP
6. **Tax optimisation** → `TaxOptimiser` selects optimal lots
7. **Explanations** → `ExplanationGenerator` produces 3-tier LLM output
8. **Override check** → `InterventionClassifier` assigns intervention level
9. **Audit trail** → `ComplianceExplainer` produces full regulatory record
10. **Dashboard** → Streamlit displays live state

## Performance Targets

| Operation | Target | Implementation |
|-----------|--------|----------------|
| Full 50k portfolio drift scan | < 30s | Vectorised NumPy |
| Single portfolio optimisation | < 2s | CVXPY/OSQP |
| Explanation generation | < 5s | Claude claude-sonnet-4-6 |
| Batch rebalancing throughput | 1,000/hr | CrewAI parallel agents |
