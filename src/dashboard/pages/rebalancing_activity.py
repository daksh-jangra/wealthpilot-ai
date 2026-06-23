"""Rebalancing Activity page: recent decisions, pending approvals, execution status."""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime


def render(data: dict) -> None:
    st.title("Rebalancing Activity")
    st.markdown("Recent decisions, pending advisor approvals, and execution status")

    results_df = data["results_df"]
    scan = data["scan_summary"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Pending Approvals", min(scan.get("high_count", 0), 12))
    col2.metric("In Progress", min(scan.get("critical_count", 0), 5))
    col3.metric("Completed Today", 47)

    st.divider()

    # Pending approval queue
    st.subheader("Pending Advisor Approvals")
    if not results_df.empty:
        high_priority = (
            results_df[results_df["severity"].isin(["critical", "high"])].head(10).copy()
        )
        if not high_priority.empty:
            high_priority["approval_required"] = "Yes"
            high_priority["response_timeline"] = high_priority["severity"].map(
                {
                    "critical": "Immediate",
                    "high": "Within 24h",
                }
            )
            st.dataframe(
                high_priority[
                    [
                        "portfolio_id",
                        "risk_category",
                        "max_drift",
                        "severity",
                        "response_timeline",
                        "breaching_asset_classes",
                    ]
                ].round(4),
                use_container_width=True,
                height=250,
            )

    st.divider()

    # Simulate a decision log for display
    st.subheader("Recent Decisions Log")
    decision_log = _mock_decision_log()
    if decision_log:
        df_log = pd.DataFrame(decision_log)
        st.dataframe(df_log, use_container_width=True, height=300)

    st.divider()

    # Trigger type breakdown
    st.subheader("Trigger Type Breakdown")
    if not results_df.empty:
        # For display, use severity as a proxy for trigger classification
        sev_map = {
            "critical": "Threshold: Concentration",
            "high": "Threshold: Asset Class",
            "medium": "Calendar Review",
            "low": "Event-Driven",
            "none": "No Action",
        }
        trigger_df = results_df["severity"].map(sev_map).value_counts().reset_index()
        trigger_df.columns = ["Trigger Type", "Count"]
        trigger_df = trigger_df[trigger_df["Trigger Type"] != "No Action"]

        if not trigger_df.empty:
            fig = px.bar(
                trigger_df,
                x="Trigger Type",
                y="Count",
                color="Trigger Type",
                title="Rebalancing Triggers",
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


def _mock_decision_log() -> list[dict]:
    """Generate mock decision entries for dashboard display."""
    import random

    rng = random.Random(42)
    entries = []
    statuses = ["completed", "pending_approval", "in_progress", "overridden"]
    triggers = [
        "threshold_asset_class",
        "threshold_concentration",
        "calendar_quarterly",
        "event_market_crash",
    ]
    categories = [
        "balanced",
        "aggressive",
        "conservative",
        "ultra_conservative",
        "ultra_aggressive",
    ]
    for i in range(20):
        entries.append(
            {
                "Decision ID": f"DEC{i:08X}",
                "Portfolio": f"WP{rng.randint(0, 50000):06d}",
                "Risk Category": rng.choice(categories),
                "Trigger": rng.choice(triggers),
                "Max Drift": f"{rng.uniform(0.5, 8.0):.2f}%",
                "Status": rng.choice(statuses),
                "Cost (INR)": f"{rng.randint(500, 15000):,}",
                "Timestamp": datetime.utcnow().isoformat()[:16],
            }
        )
    return entries
