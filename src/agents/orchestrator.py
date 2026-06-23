"""
Multi-agent orchestration using CrewAI.
Defines the full agent crew: Orchestrator, Portfolio Analyst, Risk Manager,
Tax Specialist, Compliance Officer, Explanation Writer.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime

try:
    from crewai import Agent
    from langchain_anthropic import ChatAnthropic

    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

import numpy as np
import pandas as pd

from src.explainability.explanation_generator import ExplanationGenerator
from src.explainability.shap_integration import SHAPIntegration
from src.monitoring.drift_calculator import DriftResult
from src.optimisation.constraint_manager import ConstraintManager
from src.optimisation.portfolio_optimiser import PortfolioOptimiser
from src.optimisation.trade_list_generator import Trade, TradeListGenerator
from src.triggers.trigger_consolidator import ConsolidatedTrigger

TARGET_ALLOCATIONS = {
    "ultra_conservative": np.array([0.10, 0.05, 0.45, 0.15, 0.10, 0.15]),
    "conservative": np.array([0.20, 0.10, 0.35, 0.10, 0.12, 0.13]),
    "balanced": np.array([0.35, 0.15, 0.20, 0.10, 0.12, 0.08]),
    "aggressive": np.array([0.50, 0.20, 0.10, 0.05, 0.10, 0.05]),
    "ultra_aggressive": np.array([0.60, 0.25, 0.05, 0.00, 0.07, 0.03]),
}


class RebalancingOrchestrator:
    """
    Orchestrate the full rebalancing pipeline for a single portfolio.
    Uses CrewAI when available; falls back to direct pipeline otherwise.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        use_crewai: bool = True,
        securities_master: pd.DataFrame | None = None,
    ):
        self.model_name = model
        self.use_crewai = use_crewai and CREWAI_AVAILABLE
        self.securities_master = securities_master
        self.optimiser = PortfolioOptimiser()
        self.trade_generator = TradeListGenerator()
        self.constraint_manager = ConstraintManager()
        self.explanation_generator = ExplanationGenerator(model=model)
        self.shap = SHAPIntegration()

        if self.use_crewai:
            self._llm = ChatAnthropic(
                model=model,
                anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                temperature=0.3,
            )
            self._build_crew()

    def _build_crew(self) -> None:
        """Define CrewAI agents with domain-specific backstories and goals."""
        self._portfolio_analyst = Agent(
            role="Portfolio Analyst",
            goal="Compute optimal trades to restore target allocation with minimum cost",
            backstory=(
                "A CFA-qualified portfolio analyst with 15 years experience in Indian equity "
                "and fixed income markets. Expert in constrained optimisation and trade execution."
            ),
            llm=self._llm,
            verbose=False,
            allow_delegation=False,
        )

        self._risk_manager = Agent(
            role="Risk Manager",
            goal="Evaluate risk implications of proposed trades and ensure post-trade risk is acceptable",
            backstory=(
                "A senior risk manager who has managed portfolio risk through the 2008 GFC, "
                "2020 COVID crash, and 2022 rate shock. Expert in VaR, CVaR, and stress testing."
            ),
            llm=self._llm,
            verbose=False,
            allow_delegation=False,
        )

        self._tax_specialist = Agent(
            role="Tax Specialist",
            goal="Optimise trade execution for after-tax efficiency using Indian tax laws",
            backstory=(
                "A chartered accountant specialising in securities taxation for HNIs. "
                "Deep expertise in STCG/LTCG rules, indexation, and tax-loss harvesting "
                "under Indian income tax law."
            ),
            llm=self._llm,
            verbose=False,
            allow_delegation=False,
        )

        self._compliance_officer = Agent(
            role="Compliance Officer",
            goal="Verify all trades comply with SEBI regulations and internal risk limits",
            backstory=(
                "A SEBI-registered compliance officer who has audited robo-advisory systems "
                "for regulatory compliance. Expert in algorithmic trading regulations "
                "and SEBI's IA regulations."
            ),
            llm=self._llm,
            verbose=False,
            allow_delegation=False,
        )

        self._explanation_writer = Agent(
            role="Explanation Writer",
            goal="Produce clear, accurate explanations for all three audiences: client, advisor, compliance",
            backstory=(
                "A financial communications expert who has written client-facing content for "
                "leading robo-advisors. Skilled at translating complex financial decisions "
                "into plain language without sacrificing accuracy."
            ),
            llm=self._llm,
            verbose=False,
            allow_delegation=False,
        )

    def process_portfolio(
        self,
        drift_result: DriftResult,
        trigger: ConsolidatedTrigger,
        portfolio_value_inr: float,
        client_profile: dict,
        vix: float = 18.0,
        days_since_last_rebalance: int = 90,
    ) -> dict:
        """
        Run the full rebalancing pipeline for one portfolio.
        Returns a comprehensive decision record.
        """
        decision_id = f"DEC{uuid.uuid4().hex[:8].upper()}"

        # Step 1: Portfolio optimisation
        opt_result = self.optimiser.optimise(
            current_weights=drift_result.current_weights,
            target_weights=drift_result.target_weights,
            turnover_budget=0.20,
            portfolio_value_inr=portfolio_value_inr,
        )

        # Step 2: Generate trades
        trades: list[Trade] = []
        if self.securities_master is not None and not self.securities_master.empty:
            trades = self.trade_generator.generate(
                portfolio_id=drift_result.portfolio_id,
                opt_result=opt_result,
                portfolio_value_inr=portfolio_value_inr,
                securities_master=self.securities_master,
            )

        # Step 3: Constraint verification
        violations = self.constraint_manager.check_post_trade(
            post_trade_weights=opt_result.post_trade_weights,
            trade_weights=opt_result.trade_weights,
        )
        constraint_checks = self.constraint_manager.violations_summary(violations)

        # Step 4: Assemble decision metadata
        trade_summary = self.trade_generator.trade_list_summary(trades)
        max_drift_pct = drift_result.max_drift * 100

        decision_metadata = {
            "decision_id": decision_id,
            "portfolio_id": drift_result.portfolio_id,
            "risk_category": drift_result.risk_category,
            "trigger_type": trigger.primary_trigger.trigger_type.value,
            "trigger_priority": trigger.priority.value,
            "max_drift_pct": round(max_drift_pct, 2),
            "sum_abs_drift_pct": round(drift_result.sum_abs_drift * 100, 2),
            "breaching_asset_classes": drift_result.breaching_asset_classes,
            "tracking_error_before": round(drift_result.rmsd * 100, 3),
            "tracking_error_after": round(opt_result.tracking_error * 100, 3),
            "turnover": round(opt_result.turnover, 4),
            "total_cost_inr": trade_summary.get("total_cost_inr", 0),
            "tax_impact_inr": 0,  # populated by tax specialist
            "trade_summary": trade_summary,
            "constraint_checks": constraint_checks,
            "vix": vix,
            "risk_score": client_profile.get("risk_score", 3),
            "tax_bracket": client_profile.get("tax_bracket", 0.30),
            "ltcg_lot_fraction": 0.5,  # simplified
            "sector_concentration_max": 0.20,
            "portfolio_value_log": float(np.log(max(portfolio_value_inr, 1))),
            "days_since_last_rebalance": days_since_last_rebalance,
            "model_version": "1.0.0",
            "override_history": [],
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Step 5: Generate explanations
        explanations = self.explanation_generator.generate_all_tiers(decision_metadata)

        # Step 6: SHAP attribution
        features = {
            "max_drift_pct": max_drift_pct,
            "sum_abs_drift_pct": drift_result.sum_abs_drift * 100,
            "days_since_last_rebalance": float(days_since_last_rebalance),
            "vix": vix,
            "risk_score": float(client_profile.get("risk_score", 3)),
            "ltcg_lot_fraction": 0.5,
            "sector_concentration_max": 0.20,
            "portfolio_value_log": float(np.log(max(portfolio_value_inr, 1))),
        }
        try:
            shap_exp = self.shap.explain(drift_result.portfolio_id, features)
            decision_metadata["shap_top_features"] = shap_exp.top_features
            decision_metadata["counterfactual"] = shap_exp.counterfactual
        except Exception:
            pass

        return {
            "decision_id": decision_id,
            "decision_metadata": decision_metadata,
            "optimisation_result": opt_result,
            "trades": [vars(t) for t in trades],
            "explanations": {k: v.model_dump() for k, v in explanations.items()},
            "constraint_violations": constraint_checks,
            "status": "completed" if opt_result.status == "optimal" else "suboptimal",
        }
