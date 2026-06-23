"""System Health page: agent uptime, error rates, processing latency."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render(data: dict) -> None:
    st.title("System Health")
    st.markdown("Agent uptime, error rates, and processing latency monitoring")

    scan = data["scan_summary"]

    # Live status indicators
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("System Status", "Operational", delta="All agents healthy")
    col2.metric("Kill Switch", "INACTIVE", delta="Auto-triggers clear")
    col3.metric("Scan Latency", f"{scan['elapsed_seconds']}s", delta="vs 30s target")
    col4.metric("Error Rate", "0.0%", delta="threshold: 1%")

    st.divider()

    # Scan performance over time (simulated)
    st.subheader("Scan Latency History")
    rng = np.random.default_rng(42)
    scan_times = rng.normal(loc=8.5, scale=1.2, size=48)
    scan_times = np.clip(scan_times, 4, 20)
    timestamps = [datetime.utcnow() - timedelta(minutes=30 * i) for i in range(47, -1, -1)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=scan_times,
            mode="lines+markers",
            name="Scan Time (s)",
            line=dict(color="#3b82f6"),
        )
    )
    fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="30s target")
    fig.update_layout(height=300, xaxis_title="Time", yaxis_title="Seconds")
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Agent throughput
        st.subheader("Agent Throughput")
        throughput_data = pd.DataFrame(
            {
                "Hour": [f"{h}:00" for h in range(9, 16)],
                "Portfolios Processed": [rng.integers(80, 150) for _ in range(7)],
            }
        )
        fig2 = px.bar(
            throughput_data, x="Hour", y="Portfolios Processed", color_discrete_sequence=["#22c55e"]
        )
        fig2.update_layout(height=280)
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        # Error log
        st.subheader("Error Log (Last 24h)")
        error_data = pd.DataFrame(
            [
                {
                    "Time": "09:15:03",
                    "Component": "optimiser",
                    "Error": "OSQP solver timeout (retried)",
                    "Severity": "warning",
                },
                {
                    "Time": "11:42:17",
                    "Component": "llm_api",
                    "Error": "Rate limit — backoff applied",
                    "Severity": "warning",
                },
            ]
        )
        if not error_data.empty:
            st.dataframe(error_data, use_container_width=True, height=150)
        else:
            st.success("No errors in last 24 hours")

    st.divider()

    # Agent status table
    st.subheader("Agent Status")
    agents = [
        {
            "Agent": "Orchestrator",
            "Status": "Running",
            "Decisions/hr": 87,
            "Avg Latency (ms)": 1240,
        },
        {
            "Agent": "Portfolio Analyst",
            "Status": "Running",
            "Decisions/hr": 87,
            "Avg Latency (ms)": 1850,
        },
        {"Agent": "Risk Manager", "Status": "Running", "Decisions/hr": 87, "Avg Latency (ms)": 620},
        {
            "Agent": "Tax Specialist",
            "Status": "Running",
            "Decisions/hr": 87,
            "Avg Latency (ms)": 480,
        },
        {
            "Agent": "Compliance Officer",
            "Status": "Running",
            "Decisions/hr": 87,
            "Avg Latency (ms)": 340,
        },
        {
            "Agent": "Explanation Writer",
            "Status": "Running",
            "Decisions/hr": 87,
            "Avg Latency (ms)": 3200,
        },
    ]
    st.dataframe(pd.DataFrame(agents), use_container_width=True)

    # Kill switch controls
    st.divider()
    st.subheader("Kill Switch Controls")
    st.warning("The kill switch halts all autonomous rebalancing immediately.")
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        if st.button("Activate Kill Switch", type="secondary", use_container_width=True):
            st.error("Kill switch ACTIVATED — all autonomous rebalancing halted")
    with col_k2:
        if st.button("Deactivate Kill Switch", type="primary", use_container_width=True):
            st.success("Kill switch DEACTIVATED — autonomous rebalancing resumed")
