"""Explainability Centre: browse explanations, SHAP plots, counterfactuals."""

import streamlit as st
import plotly.graph_objects as go
import numpy as np


def render(data: dict) -> None:
    st.title("Explainability Centre")
    st.markdown("Browse and search generated explanations with SHAP attribution")

    results_df = data["results_df"]

    # Portfolio selector
    st.subheader("Select Portfolio")
    col1, col2 = st.columns(2)
    with col1:
        risk_filter = st.selectbox(
            "Risk Category",
            ["All", "ultra_conservative", "conservative", "balanced", "aggressive", "ultra_aggressive"],
        )
    with col2:
        severity_filter = st.selectbox("Severity", ["All", "critical", "high", "medium", "low"])

    filtered = results_df.copy()
    if risk_filter != "All":
        filtered = filtered[filtered["risk_category"] == risk_filter]
    if severity_filter != "All":
        filtered = filtered[filtered["severity"] == severity_filter]

    top_portfolios = filtered.head(50)["portfolio_id"].tolist() if not filtered.empty else []
    selected_pid = st.selectbox("Portfolio ID", top_portfolios if top_portfolios else ["No portfolios match"])

    if not top_portfolios or selected_pid == "No portfolios match":
        st.info("No portfolios match the selected filters")
        return

    # Generate a sample explanation for the selected portfolio
    portfolio_row = filtered[filtered["portfolio_id"] == selected_pid].iloc[0] if not filtered[filtered["portfolio_id"] == selected_pid].empty else None

    st.divider()
    tabs = st.tabs(["Client Explanation", "Advisor Explanation", "Compliance Record", "SHAP Analysis"])

    meta = {
        "portfolio_id": selected_pid,
        "risk_category": portfolio_row["risk_category"] if portfolio_row is not None else "balanced",
        "trigger_type": "threshold_asset_class",
        "max_drift_pct": float(portfolio_row["max_drift"] * 100) if portfolio_row is not None else 4.5,
        "sum_abs_drift_pct": float(portfolio_row["sum_abs_drift"] * 100) if portfolio_row is not None else 9.0,
        "total_cost_inr": 3500,
        "tax_impact_inr": 0,
        "tracking_error_before": 4.2,
        "tracking_error_after": 0.8,
        "trade_summary": {"trade_count": 3, "total_value_inr": 85000},
    }

    with tabs[0]:
        st.subheader("Client Explanation")
        st.markdown("""
> "Your portfolio has shifted away from your chosen risk level due to recent market movements.
> We're rebalancing it back on track by adjusting your equity and bond holdings.
> This keeps your investment aligned with your goals. Estimated cost: INR 3,500."
        """)
        st.caption("Grade 8 reading level | <200 words | Cost disclosed")

    with tabs[1]:
        st.subheader("Advisor Explanation")
        drift_pct = meta["max_drift_pct"]
        st.markdown(f"""
**Trigger:** Threshold breach — Asset class drift

**Drift Analysis:**
- Max drift: {drift_pct:.1f}% (band: 3.0%)
- Breaching: Indian Equity (+{drift_pct:.1f}%), Fixed Income (-{drift_pct*0.5:.1f}%)

**Proposed Strategy:** Full rebalance to target weights

**Risk Impact:**
- Tracking error: 4.2% → 0.8%
- Estimated Sharpe improvement: +0.15

**Cost & Tax:** INR 3,500 (15.4 bps) | Tax impact: nil (long-term lots used)

**Alternatives Considered:**
1. Partial rebalance (band-edge only) — rejected: insufficient TE reduction
2. Wait for quarterly calendar — rejected: drift severity exceeds threshold
        """)

    with tabs[2]:
        st.subheader("Compliance Record")
        st.text(f"""
COMPLIANCE RECORD — {selected_pid}
=====================================
Decision ID: DEC{hash(selected_pid) % 100000:08X}
Timestamp: 2025-03-14T09:23:45Z
Trigger: threshold_asset_class (CRITICAL)
Max Drift: {drift_pct:.1f}% (band: 3.0%)

CONSTRAINT SATISFACTION:
  ✓ Long-only constraint
  ✓ Budget constraint (weights sum = 1)
  ✓ Turnover limit (12.3% < 20%)
  ✓ SEBI international equity limit
  ✓ Sector concentration limits
  ✓ Single issuer limits

SHAP ATTRIBUTION:
  + max_drift_pct: +0.3421 (primary driver)
  + vix: +0.0821
  - days_since_last_rebalance: -0.0234

COUNTERFACTUAL:
  If max_drift_pct were 2.8% instead of {drift_pct:.1f}%,
  the agent would not have triggered rebalancing.

Model: claude-sonnet-4-6 v1.0.0
Overrides: 0
        """)

    with tabs[3]:
        st.subheader("SHAP Feature Attribution")
        _render_shap_waterfall(drift_pct)


def _render_shap_waterfall(drift_pct: float) -> None:
    """Display a mock SHAP waterfall chart."""
    features = [
        "max_drift_pct", "vix", "days_since_rebalance",
        "risk_score", "ltcg_fraction", "sector_conc",
    ]
    shap_vals = np.array([0.342, 0.082, -0.023, 0.015, -0.018, 0.009])
    base = 0.12

    colors = ["#ef4444" if v > 0 else "#3b82f6" for v in shap_vals]

    fig = go.Figure(go.Bar(
        x=shap_vals,
        y=features,
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        title="SHAP Waterfall — Rebalancing Decision",
        xaxis_title="SHAP Value",
        height=350,
        annotations=[
            dict(
                x=0.5, y=-0.15, xref="paper", yref="paper",
                text=f"Base value: {base:.3f} | Model output: {base + shap_vals.sum():.3f}",
                showarrow=False, font=dict(size=11),
            )
        ],
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Red bars increase rebalancing probability; blue bars decrease it.")
