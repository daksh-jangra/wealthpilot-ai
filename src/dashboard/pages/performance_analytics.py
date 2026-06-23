"""Performance Analytics page: backtest results and strategy comparison."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd


def render(data: dict) -> None:
    st.title("Performance Analytics")
    st.markdown("Backtest results, rolling metrics, and strategy comparison")

    market_returns = data.get("market_returns")
    if market_returns is None:
        st.warning("Market data not available")
        return

    # Run a quick backtest on a sample portfolio
    st.subheader("Strategy Comparison: Agent vs Legacy Quarterly")

    with st.spinner("Running backtest simulation..."):
        comparison_data = _run_sample_backtest(market_returns)

    # Performance table
    st.dataframe(
        pd.DataFrame(comparison_data).T.round(3),
        use_container_width=True,
    )

    st.divider()

    # Equity curve comparison
    st.subheader("Portfolio Value Over Time")
    eq_curves = _simulate_equity_curves(market_returns)

    fig = go.Figure()
    for strategy, values in eq_curves.items():
        fig.add_trace(
            go.Scatter(
                x=list(range(len(values))),
                y=values,
                name=strategy,
                mode="lines",
            )
        )
    fig.update_layout(
        height=400,
        xaxis_title="Trading Days",
        yaxis_title="Portfolio Value (INR)",
        legend_title="Strategy",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Improvement scorecard
    st.subheader("Improvement Scorecard: Agent vs Legacy")
    scorecard = _build_scorecard(comparison_data)
    cols = st.columns(len(scorecard))
    for col, (metric, result) in zip(cols, scorecard.items()):
        icon = "✅" if result["agent_wins"] else "❌"
        col.metric(
            label=metric,
            value=f"{result['agent']:.3f}",
            delta=f"{result['agent'] - result['legacy']:.3f} vs legacy",
        )

    wins = sum(1 for r in scorecard.values() if r["agent_wins"])
    if wins >= 4:
        st.success(f"Agent outperforms legacy on {wins}/6 key metrics — target MET")
    else:
        st.warning(f"Agent outperforms legacy on {wins}/6 key metrics — target: 4+")

    st.divider()

    # Rolling Sharpe
    st.subheader("Rolling 63-Day Sharpe Ratio")
    rolling_data = _rolling_sharpe(market_returns)
    fig2 = px.line(
        rolling_data,
        y=list(rolling_data.keys()),
        labels={"value": "Sharpe Ratio", "index": "Day"},
    )
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)


def _run_sample_backtest(market_returns) -> dict:
    from src.backtesting.backtest_engine import BacktestEngine

    target = np.array([0.35, 0.15, 0.20, 0.10, 0.12, 0.08])
    # Slight initial drift
    initial = target + np.array([0.05, 0.02, -0.04, -0.02, 0.0, -0.01])
    initial = np.clip(initial, 0, 1)
    initial /= initial.sum()

    engine = BacktestEngine(market_returns)
    strategies = ["agent", "legacy_quarterly", "threshold_only", "buy_and_hold"]
    comparison = {}
    for s in strategies:
        r = engine.run("SAMPLE001", "balanced", initial, 1_000_000, s)
        comparison[s] = {
            "Annualised Return": round(r.annualised_return, 4),
            "Volatility": round(r.annualised_volatility, 4),
            "Sharpe": round(r.sharpe_ratio, 3),
            "Max Drawdown": round(r.max_drawdown, 3),
            "Tracking Error": round(r.tracking_error, 4),
            "Total Cost (INR)": round(r.total_cost_inr, 0),
        }
    return comparison


def _simulate_equity_curves(market_returns) -> dict:
    from src.backtesting.backtest_engine import BacktestEngine

    target = np.array([0.35, 0.15, 0.20, 0.10, 0.12, 0.08])
    initial = target + np.array([0.05, 0.02, -0.04, -0.02, 0.0, -0.01])
    initial = np.clip(initial, 0, 1)
    initial /= initial.sum()
    engine = BacktestEngine(market_returns)
    curves = {}
    for s in ["agent", "legacy_quarterly", "buy_and_hold"]:
        r = engine.run("SAMPLE001", "balanced", initial, 1_000_000, s)
        curves[s] = [state.value_inr for state in r.daily_states]
    return curves


def _build_scorecard(comparison: dict) -> dict:
    agent = comparison.get("agent", {})
    legacy = comparison.get("legacy_quarterly", {})
    metrics = {
        "Sharpe": {
            "agent": agent.get("Sharpe", 0),
            "legacy": legacy.get("Sharpe", 0),
            "higher_better": True,
        },
        "Max DD": {
            "agent": agent.get("Max Drawdown", 0),
            "legacy": legacy.get("Max Drawdown", 0),
            "higher_better": False,
        },
        "Track.Err": {
            "agent": agent.get("Tracking Error", 0),
            "legacy": legacy.get("Tracking Error", 0),
            "higher_better": False,
        },
        "Cost": {
            "agent": agent.get("Total Cost (INR)", 0),
            "legacy": legacy.get("Total Cost (INR)", 0),
            "higher_better": False,
        },
        "Ann.Ret": {
            "agent": agent.get("Annualised Return", 0),
            "legacy": legacy.get("Annualised Return", 0),
            "higher_better": True,
        },
        "Volatility": {
            "agent": agent.get("Volatility", 0),
            "legacy": legacy.get("Volatility", 0),
            "higher_better": False,
        },
    }
    for m in metrics.values():
        if m["higher_better"]:
            m["agent_wins"] = m["agent"] > m["legacy"]
        else:
            m["agent_wins"] = m["agent"] < m["legacy"]
    return metrics


def _rolling_sharpe(market_returns, window: int = 63) -> pd.DataFrame:
    returns = market_returns["indian_equity"]
    rolling = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
    return pd.DataFrame({"indian_equity_sharpe": rolling.dropna()})
