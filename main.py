"""
WealthPilot AI — Autonomous Portfolio Rebalancing Agent
CLI entry point for the full pipeline.

Usage:
    python main.py demo                      # Run a demo rebalancing cycle
    python main.py scan                      # Scan all 50k portfolios for drift
    python main.py rebalance --id WP000001  # Rebalance a specific portfolio
    python main.py backtest                  # Run strategy comparison backtest
    python main.py dashboard                 # Launch Streamlit dashboard
"""

from __future__ import annotations

import argparse
import sys
import time
import os

import numpy as np
import pandas as pd

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import track
    RICH = True
except ImportError:
    RICH = False

console = Console() if RICH else None


def _print(msg: str, style: str = "") -> None:
    if RICH and console:
        console.print(msg, style=style)
    else:
        print(msg)


def _header(title: str) -> None:
    if RICH and console:
        console.print(Panel(f"[bold]{title}[/bold]", style="blue"))
    else:
        print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ── Demo ─────────────────────────────────────────────────────────────────────

def run_demo() -> None:
    _header("WealthPilot AI — Demo Rebalancing Cycle")

    _print("\n[1/7] Generating synthetic market data (252 trading days)...", "cyan")
    from src.data.market_data_simulator import MarketDataSimulator
    sim = MarketDataSimulator(seed=42)
    returns = sim.simulate_returns()
    vix = sim.get_vix_series()
    _print(f"  Market data: {returns.shape[0]} days × {returns.shape[1]} asset classes", "green")
    _print(f"  VIX range: {vix.min():.1f} – {vix.max():.1f}", "green")

    _print("\n[2/7] Generating 50,000 client portfolios...", "cyan")
    t0 = time.time()
    from src.data.client_profile_generator import ClientProfileGenerator
    from src.data.portfolio_generator import PortfolioGenerator
    clients = ClientProfileGenerator(seed=42).generate_all()
    gen = PortfolioGenerator(seed=42)
    weights = gen.generate_current_allocations(clients)
    values = gen.generate_portfolio_values(clients)
    _print(f"  Generated {len(clients):,} portfolios in {time.time()-t0:.1f}s", "green")
    _print(f"  AUM: INR {values.sum()/1e7:.0f} crores", "green")

    _print("\n[3/7] Running drift scan across all 50,000 portfolios...", "cyan")
    from src.monitoring.drift_monitor import DriftMonitor
    monitor = DriftMonitor()
    t0 = time.time()
    summary = monitor.run_scan(weights, clients["risk_category"])
    elapsed = time.time() - t0
    _print(f"  Scan completed in {elapsed:.2f}s (target: <30s)", "green")
    _print(f"  Portfolios requiring action: {summary['actionable_count']:,}", "green")
    _print(f"  Critical:  {summary.get('critical_count', 0):>5,}", "yellow")
    _print(f"  High:      {summary.get('high_count', 0):>5,}", "yellow")
    _print(f"  Medium:    {summary.get('medium_count', 0):>5,}", "dim")

    _print("\n[4/7] Selecting top-priority portfolio and evaluating triggers...", "cyan")
    top = monitor.get_top_n(n=1)
    if not top:
        _print("  No portfolios need rebalancing today.", "green")
        return
    drift_result = top[0]
    _print(f"  Portfolio: {drift_result.portfolio_id}", "bold")
    _print(f"  Max drift: {drift_result.max_drift*100:.1f}%  |  Severity: {drift_result.severity.name}")

    from src.triggers.trigger_consolidator import TriggerConsolidator
    from src.monitoring.drift_calculator import DriftResult as DriftRes
    consolidator = TriggerConsolidator()
    trigger = consolidator.evaluate_portfolio(drift_result)
    _print(f"  Primary trigger: {trigger.primary_trigger.trigger_type.value}", "yellow")
    _print(f"  Priority: {trigger.priority.value}")

    _print("\n[5/7] Optimising portfolio with CVXPY / OSQP...", "cyan")
    from src.optimisation.portfolio_optimiser import PortfolioOptimiser
    optimiser = PortfolioOptimiser()
    t0 = time.time()
    opt = optimiser.optimise(
        current_weights=drift_result.current_weights,
        target_weights=drift_result.target_weights,
        portfolio_value_inr=float(values.iloc[0]),
    )
    _print(f"  Status: {opt.status}  ({time.time()-t0:.2f}s)", "green")
    _print(f"  Tracking error before: {drift_result.rmsd*100:.2f}%  →  after: {opt.tracking_error*100:.2f}%")
    _print(f"  Turnover: {opt.turnover*100:.1f}%")

    _print("\n[6/7] Generating 3-tier explanations (no API key = fallback mode)...", "cyan")
    from src.explainability.explanation_generator import ExplanationGenerator
    explainer = ExplanationGenerator(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
    meta = {
        "portfolio_id": drift_result.portfolio_id,
        "risk_category": drift_result.risk_category,
        "trigger_type": trigger.primary_trigger.trigger_type.value,
        "max_drift_pct": round(drift_result.max_drift * 100, 2),
        "sum_abs_drift_pct": round(drift_result.sum_abs_drift * 100, 2),
        "total_cost_inr": 3500,
        "vix": float(vix.iloc[-1]),
        "risk_score": 3,
        "tax_bracket": 0.30,
        "days_since_last_rebalance": 90,
    }
    explanations = explainer.generate_all_tiers(meta)
    for audience, exp in explanations.items():
        _print(f"\n  [{audience.upper()}] ({exp.word_count} words)", "bold")
        _print(f"  {exp.narrative[:200]}{'...' if len(exp.narrative) > 200 else ''}", "dim")

    _print("\n[7/7] Running quick backtest (agent vs legacy quarterly)...", "cyan")
    from src.backtesting.backtest_engine import BacktestEngine
    from src.backtesting.performance_analyser import PerformanceAnalyser
    market = sim.simulate_returns()
    market.index = pd.date_range("2024-01-02", periods=len(market), freq="B")
    engine = BacktestEngine(market)
    agent_r = engine.run(drift_result.portfolio_id, drift_result.risk_category,
                         drift_result.current_weights, 1_000_000, "agent")
    legacy_r = engine.run(drift_result.portfolio_id, drift_result.risk_category,
                          drift_result.current_weights, 1_000_000, "legacy_quarterly")
    analyser = PerformanceAnalyser()
    sc = analyser.improvement_scorecard(analyser.summarise(agent_r), analyser.summarise(legacy_r))
    _print(f"  Agent wins on {sc['agent_wins']}/6 metrics  |  Target met: {'YES' if sc['target_met'] else 'NO'}", "green")

    _print("\nDemo complete.", "bold green")


# ── Scan ─────────────────────────────────────────────────────────────────────

def run_scan() -> None:
    _header("WealthPilot AI — Full Portfolio Drift Scan")
    from src.data.client_profile_generator import ClientProfileGenerator
    from src.data.portfolio_generator import PortfolioGenerator
    from src.monitoring.drift_monitor import DriftMonitor

    _print("Generating portfolios...", "cyan")
    clients = ClientProfileGenerator(seed=42).generate_all()
    weights = PortfolioGenerator(seed=42).generate_current_allocations(clients)

    _print("Scanning for drift...", "cyan")
    t0 = time.time()
    monitor = DriftMonitor()
    summary = monitor.run_scan(weights, clients["risk_category"])
    elapsed = time.time() - t0

    if RICH and console:
        table = Table(title="Drift Scan Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Total portfolios", f"{summary['total_portfolios']:,}")
        table.add_row("Actionable", f"{summary['actionable_count']:,}")
        table.add_row("Critical", str(summary.get("critical_count", 0)))
        table.add_row("High", str(summary.get("high_count", 0)))
        table.add_row("Scan time", f"{elapsed:.2f}s")
        console.print(table)
    else:
        for k, v in summary.items():
            print(f"  {k}: {v}")


# ── Rebalance ────────────────────────────────────────────────────────────────

def run_rebalance(portfolio_id: str) -> None:
    _header(f"Rebalancing Portfolio: {portfolio_id}")
    from src.data.client_profile_generator import ClientProfileGenerator
    from src.data.portfolio_generator import PortfolioGenerator
    from src.monitoring.drift_calculator import DriftCalculator
    from src.triggers.trigger_consolidator import TriggerConsolidator
    from src.agents.orchestrator import RebalancingOrchestrator, TARGET_ALLOCATIONS

    clients = ClientProfileGenerator(seed=42).generate_all()
    weights = PortfolioGenerator(seed=42).generate_current_allocations(clients)
    values = PortfolioGenerator(seed=42).generate_portfolio_values(clients)

    if portfolio_id not in weights.index:
        _print(f"Portfolio {portfolio_id!r} not found.", "red")
        sys.exit(1)

    client = clients[clients["client_id"] == portfolio_id].iloc[0].to_dict()
    current_w = weights.loc[portfolio_id].values
    target_w = TARGET_ALLOCATIONS[client["risk_category"]]
    pv = float(values[portfolio_id])

    calc = DriftCalculator()
    drift = calc.compute_single(portfolio_id, current_w, target_w, client["risk_category"])
    consolidator = TriggerConsolidator()
    trigger = consolidator.evaluate_portfolio(drift)

    orchestrator = RebalancingOrchestrator(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        use_crewai=False,
    )
    decision = orchestrator.process_portfolio(drift, trigger, pv, client)

    _print(f"\n  Decision ID:   {decision['decision_id']}", "bold")
    _print(f"  Status:        {decision['status']}")
    _print(f"  Max drift:     {decision['decision_metadata']['max_drift_pct']:.1f}%")
    _print(f"  Trades:        {len(decision['trades'])}")
    _print(f"\n  Client explanation:")
    _print(f"  {decision['explanations']['client']['narrative'][:300]}", "dim")


# ── Backtest ─────────────────────────────────────────────────────────────────

def run_backtest() -> None:
    _header("WealthPilot AI — Strategy Comparison Backtest")
    from src.data.market_data_simulator import MarketDataSimulator
    from src.backtesting.strategy_comparator import StrategyComparator
    from src.backtesting.backtest_engine import BacktestEngine

    sim = MarketDataSimulator(seed=42)
    returns = sim.simulate_returns()
    returns.index = pd.date_range("2024-01-02", periods=len(returns), freq="B")
    engine = BacktestEngine(returns)
    comparator = StrategyComparator(engine)

    initial_weights = np.array([0.35, 0.15, 0.20, 0.10, 0.12, 0.08])
    _print("Running 4 strategies on balanced portfolio...", "cyan")
    result = comparator.compare_all("WP000001", "balanced", initial_weights)

    if RICH and console:
        table = Table(title="Strategy Comparison")
        table.add_column("Strategy", style="cyan")
        table.add_column("Sharpe", justify="right")
        table.add_column("Max DD", justify="right")
        table.add_column("Tracking Err", justify="right")
        table.add_column("Rebalances", justify="right")
        for strategy, summary in result["strategy_summaries"].items():
            table.add_row(
                strategy,
                f"{summary['sharpe_ratio']:.3f}",
                f"{summary['max_drawdown_pct']:.1f}%",
                f"{summary['tracking_error_pct']:.2f}%",
                str(summary["rebalance_count"]),
            )
        console.print(table)
    sc = result["improvement_scorecard"]
    _print(f"\nAgent beats legacy on {sc['agent_wins']}/6 metrics. Target met: {'YES' if sc['target_met'] else 'NO'}",
           "green" if sc["target_met"] else "yellow")


# ── Dashboard ────────────────────────────────────────────────────────────────

def run_dashboard() -> None:
    _print("Launching Streamlit dashboard...", "cyan")
    os.execvp("streamlit", ["streamlit", "run", "src/dashboard/app.py"])


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="wealthpilot",
        description="WealthPilot AI — Autonomous Portfolio Rebalancing Agent",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("demo", help="Run a complete demo rebalancing cycle")
    sub.add_parser("scan", help="Scan all 50,000 portfolios for drift")

    p_reb = sub.add_parser("rebalance", help="Rebalance a specific portfolio")
    p_reb.add_argument("--id", required=True, dest="portfolio_id", help="Portfolio ID (e.g. WP000001)")

    sub.add_parser("backtest", help="Run agent vs legacy strategy comparison")
    sub.add_parser("dashboard", help="Launch the Streamlit monitoring dashboard")

    args = parser.parse_args()

    if args.command == "demo":
        run_demo()
    elif args.command == "scan":
        run_scan()
    elif args.command == "rebalance":
        run_rebalance(args.portfolio_id)
    elif args.command == "backtest":
        run_backtest()
    elif args.command == "dashboard":
        run_dashboard()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
