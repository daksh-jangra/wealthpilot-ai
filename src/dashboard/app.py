"""WealthPilot AI — Autonomous Portfolio Rebalancing Agent Dashboard."""

import streamlit as st

st.set_page_config(
    page_title="WealthPilot AI — Rebalancing Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Lazy-load heavy modules
@st.cache_resource(show_spinner=False)
def load_data():
    import numpy as np
    import pandas as pd
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from src.data.portfolio_generator import PortfolioGenerator
    from src.data.client_profile_generator import ClientProfileGenerator
    from src.data.market_data_simulator import MarketDataSimulator
    from src.monitoring.drift_calculator import DriftCalculator
    from src.monitoring.drift_monitor import DriftMonitor

    # Generate synthetic data (cached on first run)
    gen = PortfolioGenerator(seed=42)
    client_gen = ClientProfileGenerator(seed=42)
    market_sim = MarketDataSimulator(seed=42)

    client_profiles = client_gen.generate_all()
    current_weights = gen.generate_current_allocations(client_profiles)
    portfolio_values = gen.generate_portfolio_values(client_profiles)
    market_returns = market_sim.simulate_returns()
    vix_series = market_sim.get_vix_series()

    # Run drift scan
    monitor = DriftMonitor()
    risk_categories = client_profiles.set_index("client_id")["risk_category"]
    scan_summary = monitor.run_scan(current_weights, risk_categories)
    results_df = monitor.get_results_dataframe()
    heatmap_data = monitor.drift_heatmap_data()

    return {
        "client_profiles": client_profiles,
        "current_weights": current_weights,
        "portfolio_values": portfolio_values,
        "market_returns": market_returns,
        "vix_series": vix_series,
        "scan_summary": scan_summary,
        "results_df": results_df,
        "heatmap_data": heatmap_data,
        "monitor": monitor,
    }


def main():
    st.sidebar.title("WealthPilot AI")
    st.sidebar.markdown("**Autonomous Portfolio Rebalancing Agent**")
    st.sidebar.divider()

    pages = {
        "Portfolio Overview": "portfolio_overview",
        "Rebalancing Activity": "rebalancing_activity",
        "Performance Analytics": "performance_analytics",
        "Explainability Centre": "explainability_centre",
        "System Health": "system_health",
    }

    selected = st.sidebar.radio("Navigate to", list(pages.keys()))

    with st.spinner("Loading data..."):
        data = load_data()

    st.sidebar.divider()
    scan = data["scan_summary"]
    st.sidebar.metric("Portfolios Scanned", f"{scan['total_portfolios']:,}")
    st.sidebar.metric("Action Required", scan["actionable_count"])
    st.sidebar.metric("Critical", scan["critical_count"])
    if scan.get("performance_ok"):
        st.sidebar.success(f"Scan: {scan['elapsed_seconds']}s")
    else:
        st.sidebar.warning(f"Scan slow: {scan['elapsed_seconds']}s")

    # Route to selected page
    page_module = pages[selected]
    if page_module == "portfolio_overview":
        from src.dashboard.pages.portfolio_overview import render
    elif page_module == "rebalancing_activity":
        from src.dashboard.pages.rebalancing_activity import render
    elif page_module == "performance_analytics":
        from src.dashboard.pages.performance_analytics import render
    elif page_module == "explainability_centre":
        from src.dashboard.pages.explainability_centre import render
    elif page_module == "system_health":
        from src.dashboard.pages.system_health import render

    render(data)


if __name__ == "__main__":
    main()
