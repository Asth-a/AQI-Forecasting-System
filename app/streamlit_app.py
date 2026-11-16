"""
Interactive Streamlit Dashboard for Project 44:
Air Quality Forecasting & Personalised Health Advisory System
Run with: streamlit run app/streamlit_app.py
"""
import os
import pickle
import sys
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.explainability import explain_single_prediction
from src.health_advisory import generate_health_advisory, classify_aqi_band

st.set_page_config(
    page_title="AQI Forecasting & Health Advisory System",
    page_icon="🌬️",
    layout="wide",
)

st.title("🌬️ Air Quality Forecasting & Personalised Health Advisory System")
st.caption("CSE276 Course Project | 24–72h Multi-Horizon Forecasting with Quantile Uncertainty & Rule-Based Health Guidance")

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "web_cache")


@st.cache_resource(show_spinner=False)
def load_precomputed(city: str, horizon: int):
    model_df = pd.read_parquet(os.path.join(CACHE_DIR, f"{city}_model_df.parquet"))
    with open(os.path.join(CACHE_DIR, f"{city}_{horizon}h.pkl"), "rb") as f:
        bundle = pickle.load(f)
    return (
        model_df,
        bundle["metrics_df"],
        bundle["preds_df"],
        bundle["unc_stats"],
        bundle["artifacts"],
        bundle["imp_df"],
    )


# Sidebar Controls
st.sidebar.header("⚙️ Forecast & Profile Settings")
selected_city = st.sidebar.selectbox("Select City", ["Delhi", "Chandigarh"], index=0)
selected_horizon = st.sidebar.radio("Forecast Horizon (Hours Ahead)", [24, 48, 72], index=0, horizontal=True)
selected_group = st.sidebar.selectbox(
    "Population / Vulnerability Group",
    [
        "Asthmatics & Respiratory Patients",
        "Children & School-Age Youth",
        "Elderly (65+ Years)",
        "Outdoor Workers & Athletes",
        "General Public",
    ],
)

model_df, metrics_df, preds_df, unc_stats, artifacts, imp_df = load_precomputed(selected_city, selected_horizon)

# Select forecast timestamp index from test window
max_idx = len(preds_df) - 1
sample_idx = st.sidebar.slider("Select Test Forecast Timestamp Index", min_value=0, max_value=max_idx, value=max_idx - 24)

row_pred = preds_df.iloc[sample_idx]
row_feat = model_df.iloc[-len(preds_df) + sample_idx : -len(preds_df) + sample_idx + 1]

med_val = float(row_pred["median_50"])
low_val = float(row_pred["lower_80"])
high_val = float(row_pred["upper_80"])
actual_val = float(row_pred["actual"])
wind_val = float(row_feat["WindSpeed_ms"].iloc[0])

advisory = generate_health_advisory(
    group=selected_group,
    forecast_median=med_val,
    lower_80=low_val,
    upper_80=high_val,
    dominant_pollutant="PM2.5",
    wind_speed=wind_val,
)

# Top KPI Cards
band_info = classify_aqi_band(med_val)
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Forecasted AQI (+{selected_horizon}h)", f"{med_val:.0f} ({band_info['band']})", f"Actual: {actual_val:.0f}")
c2.metric("80% Prediction Interval", f"{low_val:.0f} – {high_val:.0f}", f"Width: ±{(high_val - low_val)/2:.0f} AQI")
c3.metric("Health Risk Level", advisory.risk_level, selected_group.split("&")[0].strip())
c4.metric("Best Model RMSE Skill", f"+{metrics_df['Skill vs Persistence (%)'].max():.1f}%", "vs Persistence Baseline")

st.divider()

# Tabs for structured exploration
tab1, tab2, tab3 = st.tabs(["🩺 Personalised Health Advisory", "📈 Forecast & Uncertainty Bounds", "🧠 Model Benchmark & SHAP Drivers"])

with tab1:
    st.subheader(f"Personalised Guidance for: {selected_group}")
    st.info(f"**Uncertainty Assessment:** {advisory.uncertainty_note}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### ✅ Recommended Protective Actions")
        for act in advisory.recommended_actions:
            st.markdown(f"- {act}")
        st.markdown(f"**😷 Recommended Gear:** {advisory.protective_gear}")
        st.markdown(f"**⏱️ Outdoor Activity Limit:** {advisory.outdoor_activity_limit}")

    with col_b:
        st.markdown("### 🔍 Auditable Rule Engine Trace")
        for r in advisory.triggered_rules:
            st.code(r, language="text")

with tab2:
    st.subheader(f"{selected_city} — {selected_horizon}h Ahead AQI Forecast with 80% Prediction Interval")
    plot_df = preds_df.iloc[max(0, sample_idx - 120) : min(len(preds_df), sample_idx + 48)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["timestamp"], y=plot_df["upper_80"],
        mode="lines", line=dict(width=0), showlegend=False, name="Upper 90th"
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["timestamp"], y=plot_df["lower_80"],
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(33, 150, 243, 0.22)", name="80% Uncertainty Band (P10–P90)"
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["timestamp"], y=plot_df["actual"],
        mode="lines", line=dict(color="#111111", width=2), name="Observed AQI"
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["timestamp"], y=plot_df["Gradient Boosting"],
        mode="lines", line=dict(color="#1976D2", width=2), name="Gradient Boosting Forecast"
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["timestamp"], y=plot_df["Persistence Baseline"],
        mode="lines", line=dict(color="#E65100", width=1.5, dash="dot"), name="Persistence Baseline"
    ))
    fig.update_layout(height=430, xaxis_title="Timestamp", yaxis_title="CPCB AQI", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Calibration Check on Holdout Test Set: Target Coverage = {unc_stats['Target Coverage (%)']}% | "
        f"Empirical Coverage = {unc_stats['Empirical Coverage (%)']}% | "
        f"Mean Interval Width = {unc_stats['Mean Interval Width (AQI pts)']} AQI"
    )

with tab3:
    col_m1, col_m2 = st.columns([1.15, 1])
    with col_m1:
        st.subheader(f"Model Comparison (+{selected_horizon}h Horizon)")
        st.dataframe(metrics_df, use_container_width=True)

    with col_m2:
        st.subheader("Top Global Feature Drivers")
        top_imp = imp_df.head(12).iloc[::-1]
        val_col = top_imp.columns[1]
        fig_imp = go.Figure(go.Bar(
            x=top_imp[val_col], y=top_imp["Feature"], orientation="h", marker_color="#00796B"
        ))
        fig_imp.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=val_col)
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("### 🔎 Local Explanation for Selected Forecast Timestamp")
    local_drivers = explain_single_prediction(artifacts["rf_model"], row_feat, artifacts["feature_cols"], top_k=5)
    st.table(pd.DataFrame(local_drivers))