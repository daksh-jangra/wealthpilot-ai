---
ZETHETA_INTERN_ID: [your-intern-id]
ZETHETA_PROJECT_CODE: Project 1D
ZETHETA_PROJECT_TITLE: Autonomous Portfolio Rebalancing Agent with Explainable Decisions
ZETHETA_ROLE: Agentic AI Engineer
ZETHETA_SUBMISSION_TYPE: github
ZETHETA_SUBMISSION_DATE: 2025-03-28
ZETHETA_TECH_STACK: Python, LangChain, CrewAI, CVXPY, SHAP, LIME, Streamlit
---

# Autonomous Portfolio Rebalancing Agent — WealthPilot AI

A fully autonomous, multi-agent portfolio rebalancing system with explainable decisions, built for WealthPilot AI which manages INR 25,000 crores across 50,000 client portfolios.

## Architecture

```
Data Layer → Drift Monitoring → Trigger System → Multi-Agent Crew
    ↓                                                    ↓
Market Sim    DriftCalculator      ThresholdTrigger   Portfolio Analyst
Portfolio Gen DriftMonitor         CalendarTrigger    Risk Manager
Client Profiles ThresholdManager  EventTrigger       Tax Specialist
                                  TriggerConsolidator Compliance Officer
                                                      Explanation Writer
                                                           ↓
                                              Trade List + 3-Tier Explanations
                                              Override System + Audit Trail
                                                           ↓
                                              Backtesting Framework
                                              Compliance Auditor
                                              Streamlit Dashboard
```

## Quickstart

```bash
# 1. Install dependencies
pip install -e .

# 2. Set API key
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY

# 3. Run tests
pytest

# 4. Try the CLI
python main.py demo          # full rebalancing cycle (no API key needed)
python main.py scan          # drift scan across all 50k portfolios
python main.py backtest      # strategy comparison
python main.py rebalance --id WP000001  # rebalance a single portfolio

# 5. Launch dashboard
streamlit run src/dashboard/app.py

# 6. Open the demo notebook
jupyter notebook notebooks/demo_rebalancing_cycle.ipynb
```

## Project Structure

```
src/
├── data/           # Portfolio, market data, client profile generation
├── monitoring/     # Vectorised drift calculation (50k portfolios < 30s)
├── triggers/       # Three-tier trigger taxonomy + consolidation
├── optimisation/   # CVXPY constrained QP + tax-aware execution
├── agents/         # CrewAI multi-agent orchestration
├── explainability/ # SHAP/LIME + 3-tier LLM explanations (Claude)
├── override/       # Graduated intervention model + kill switch
├── backtesting/    # Strategy comparison + 5 scenario tests
├── compliance/     # Audit, regulatory reporting, bias detection
└── dashboard/      # Streamlit 5-view monitoring app
tests/              # 85%+ coverage
docs/               # Architecture + ADRs
```

## Key Capabilities

| Capability | Implementation |
|-----------|----------------|
| Drift monitoring (50k portfolios) | Vectorised NumPy, < 30s |
| Trigger taxonomy | Threshold + Calendar + Event, priority queue |
| Portfolio optimisation | CVXPY/OSQP, constrained QP |
| Tax-aware execution | Tax-lot level, LTCG/STCG, wash-sale avoidance |
| Multi-agent orchestration | CrewAI (6 agents) + Claude claude-sonnet-4-6 |
| Explainability | SHAP (Tree) + LIME + counterfactuals |
| 3-tier explanations | Client (Grade 8) / Advisor / Compliance |
| Human override | Graduated: Informational / Advisory / Approval / Escalation |
| Backtesting | Agent vs Legacy vs Threshold vs Buy-and-Hold |
| Market scenarios | 5 scenarios: Drift / Crash / Rotation / Regulatory / Tax |
| Dashboard | Streamlit: 5 views with drill-down |
| Compliance | SEBI audit trail, bias detection, explainability scorecard |

## Performance Benchmarks

- Full 50k portfolio drift scan: **< 30 seconds**
- Single portfolio optimisation: **< 2 seconds**
- Explanation generation: **< 5 seconds** (includes Claude API call)
- Target: agent outperforms legacy on ≥ 4 of 6 metrics

## Tech Stack

- **Python 3.10+**
- **LangChain + langchain-anthropic**: LLM integration
- **CrewAI**: Multi-agent orchestration
- **CVXPY + OSQP**: Constrained portfolio optimisation
- **SHAP + XGBoost**: Explainability
- **LIME**: Local linear explanations
- **Streamlit + Plotly**: Dashboard
- **pytest + pytest-cov**: Testing (85%+ coverage)
- **Anthropic Claude claude-sonnet-4-6**: Explanation generation

## Configuration

All parameters are in `config/`:
- `default.yaml` — app settings, data sizes, agent model
- `risk_categories.yaml` — target allocations and drift bands for 5 risk profiles
- `thresholds.yaml` — trigger thresholds, optimisation params, LLM temperatures

## Running Scenarios

```python
from src.backtesting.scenario_runner import ScenarioRunner
from src.data.market_data_simulator import MarketDataSimulator

market_returns = MarketDataSimulator().simulate_returns()
runner = ScenarioRunner(market_returns)

# Run all 5 scenarios
result1 = runner.run_scenario_1_normal_drift(portfolios)
result2 = runner.run_scenario_2_market_crash(portfolios)
result3 = runner.run_scenario_3_sector_rotation(portfolios)
result4 = runner.run_scenario_4_regulatory_event(portfolios)
result5 = runner.run_scenario_5_tax_harvesting(portfolios)
```

## Regulatory Compliance

- SEBI algorithmic trading audit trail per decision
- SEBI international equity limit (25%) enforced at optimisation layer
- Suitability documentation for every recommendation
- Override audit trail with advisor identity and reason classification
- Quarterly automated audit via `ComplianceAuditor`
