"""Portfolio Overview page: drift heatmap across 50,000 portfolios."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def render(data: dict) -> None:
    st.title("Portfolio Overview")
    st.markdown("Aggregate drift heatmap across all managed portfolios")

    results_df = data["results_df"]
    heatmap_data = data["heatmap_data"]
    scan = data["scan_summary"]

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Portfolios", f"{scan['total_portfolios']:,}")
    col2.metric("Actionable", scan["actionable_count"], delta="requires rebalancing")
    col3.metric("Critical", scan["critical_count"])
    col4.metric("Scan Time", f"{scan['elapsed_seconds']}s")

    st.divider()

    # Drift heatmap by category
    st.subheader("Drift Severity by Risk Category")
    if not heatmap_data.empty:
        fig = px.bar(
            heatmap_data,
            x="risk_category",
            y=["mean_max_drift", "max_max_drift"],
            barmode="group",
            labels={"value": "Drift (%)", "variable": "Metric", "risk_category": "Risk Category"},
            color_discrete_sequence=["#3b82f6", "#ef4444"],
        )
        fig.update_layout(height=350, xaxis_tickangle=0)
        st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        # Severity distribution
        st.subheader("Severity Distribution")
        if not results_df.empty and "severity" in results_df.columns:
            sev_counts = results_df["severity"].value_counts().reset_index()
            sev_counts.columns = ["Severity", "Count"]
            fig2 = px.pie(
                sev_counts,
                names="Severity",
                values="Count",
                color="Severity",
                color_discrete_map={
                    "critical": "#ef4444",
                    "high": "#f97316",
                    "medium": "#eab308",
                    "low": "#22c55e",
                    "none": "#94a3b8",
                },
            )
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        # Drift distribution histogram
        st.subheader("Max Drift Distribution")
        if not results_df.empty and "max_drift" in results_df.columns:
            fig3 = px.histogram(
                results_df,
                x="max_drift",
                nbins=50,
                labels={"max_drift": "Max Drift (decimal)"},
                color_discrete_sequence=["#3b82f6"],
            )
            fig3.update_layout(height=300)
            st.plotly_chart(fig3, use_container_width=True)

    # Individual portfolio drill-down
    st.subheader("Portfolio Drill-Down")
    if not results_df.empty:
        critical_df = results_df[results_df["severity"].isin(["critical", "high"])].head(20)
        if not critical_df.empty:
            st.dataframe(
                critical_df[
                    [
                        "portfolio_id",
                        "risk_category",
                        "max_drift",
                        "severity",
                        "breaching_asset_classes",
                    ]
                ].round(4),
                use_container_width=True,
                height=300,
            )

    # Market returns chart
    st.subheader("Market Returns (Simulated Year)")
    market_returns = data.get("market_returns")
    if market_returns is not None:
        cumret = (1 + market_returns).cumprod() - 1
        fig4 = px.line(
            cumret,
            labels={"value": "Cumulative Return", "index": "Date"},
            title="Asset Class Cumulative Returns",
        )
        fig4.update_layout(height=350)
        st.plotly_chart(fig4, use_container_width=True)
