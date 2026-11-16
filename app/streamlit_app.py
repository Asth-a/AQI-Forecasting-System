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
from src.live_inference import predict_live_horizons

st.set_page_config(
    page_title="AQI Forecasting & Health Advisory System",
    page_icon="🌬️",
    layout="wide",
)

st.title("🌬️ Air Quality Forecasting & Personalised Health Advisory System")
st.caption("CSE276 Course Project | 24–72h Multi-Horizon Forecasting with Quantile Uncertainty & Rule-Based Health Guidance")

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "web_cache")


@st.cache_resource(show_spinner=False)
def load_city_bundles(city: str):
    model_df = pd.read_parquet(os.path.join(CACHE_DIR, f"{city}_model_df.parquet"))
    bundles = {}
    for h in [24, 48, 72]:
        with open(os.path.join(CACHE_DIR, f"{city}_{h}h.pkl"), "rb") as f:
            bundles[h] = pickle.load(f)
    return model_df, bundles


@st.cache_data(ttl=1800, show_spinner="Fetching live air quality & weather observations...")
def get_live_forecast_cached(city: str):
    _, bundles = load_city_bundles(city)
    return predict_live_horizons(city, bundles)


# Sidebar Controls
st.sidebar.header("⚙️ Forecast & Profile Settings")
forecast_mode = st.sidebar.radio(
    "Forecast Mode",
    ["Live Real-Time Forecast", "Historical Backtest Evaluation"],
    index=0,
)
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

model_df, bundles = load_city_bundles(selected_city)
active_bundle = bundles[selected_horizon]
metrics_df = active_bundle["metrics_df"]
preds_df = active_bundle["preds_df"]
unc_stats = active_bundle["unc_stats"]
artifacts = active_bundle["artifacts"]
imp_df = active_bundle["imp_df"]

live_available = False
if forecast_mode == "Live Real-Time Forecast":
    try:
        raw_live_df, live_fc_df, latest_row = get_live_forecast_cached(selected_city)
        live_available = True
        fc_row = live_fc_df[live_fc_df["horizon_h"] == selected_horizon].iloc[0]
        now_row = live_fc_df[live_fc_df["horizon_h"] == 0].iloc[0]

        med_val = float(fc_row["median_50"])
        low_val = float(fc_row["lower_80"])
        high_val = float(fc_row["upper_80"])
        current_aqi = float(now_row["median_50"])
        target_ts_str = pd.to_datetime(fc_row["target_timestamp"]).strftime("%d %b %Y, %I:%M %p")
        now_ts_str = pd.to_datetime(now_row["target_timestamp"]).strftime("%d %b %Y, %I:%M %p")
        row_feat = latest_row
        wind_val = float(row_feat["WindSpeed_ms"].iloc[0])

        st.success(f"🟢 Live Feed Active ({selected_city}) — Latest Observation: **{now_ts_str}** (Current AQI: **{current_aqi:.0f}**) | Target Forecast: **{target_ts_str}**")
    except Exception as e:
        st.warning(f"Live API unreachable ({e}). Falling back to Historical Backtest mode.")

if not live_available:
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
if live_available:
    c1.metric(f"Forecasted AQI (+{selected_horizon}h)", f"{med_val:.0f} ({band_info['band']})", f"{med_val - current_aqi:+.0f} vs Current ({current_aqi:.0f})")
else:
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
    if live_available:
        st.subheader(f"{selected_city} — Live Observed Trail (Last 72h) & Next 24–72h Forecast")
        hist_tail = raw_live_df.tail(72)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_tail["timestamp"], y=hist_tail["AQI"],
            mode="lines", line=dict(color="#111111", width=2.2), name="Live Observed AQI (Past 72h)"
        ))
        fig.add_trace(go.Scatter(
            x=live_fc_df["target_timestamp"], y=live_fc_df["upper_80"],
            mode="lines", line=dict(width=0), showlegend=False, name="Upper 90th"
        ))
        fig.add_trace(go.Scatter(
            x=live_fc_df["target_timestamp"], y=live_fc_df["lower_80"],
            mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(33, 150, 243, 0.25)", name="80% Prediction Interval (P10–P90)"
        ))
        fig.add_trace(go.Scatter(
            x=live_fc_df["target_timestamp"], y=live_fc_df["median_50"],
            mode="lines+markers", line=dict(color="#1976D2", width=2.5, dash="dash"),
            marker=dict(size=9), name="Quantile Median Forecast (+24h/+48h/+72h)"
        ))
        fig.add_trace(go.Scatter(
            x=live_fc_df["target_timestamp"], y=live_fc_df["point_gb"],
            mode="markers", marker=dict(color="#00796B", size=8, symbol="diamond"),
            name="Gradient Boosting Point Forecast"
        ))
        fig.update_layout(height=430, xaxis_title="Timestamp (IST)", yaxis_title="CPCB AQI", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        display_fc = live_fc_df[live_fc_df["horizon_h"] > 0][
            ["label", "target_timestamp", "point_gb", "median_50", "lower_80", "upper_80"]
        ].rename(columns={
            "label": "Horizon",
            "target_timestamp": "Target Time (IST)",
            "point_gb": "GB Point AQI",
            "median_50": "Median Forecast (P50)",
            "lower_80": "Lower 80% (P10)",
            "upper_80": "Upper 80% (P90)",
        })
        st.dataframe(display_fc, use_container_width=True, hide_index=True)
    else:
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

    st.markdown("### 🔎 Local Explanation for Current / Selected Forecast Timestamp")
    local_drivers = explain_single_prediction(artifacts["rf_model"], row_feat, artifacts["feature_cols"], top_k=5)
    st.table(pd.DataFrame(local_drivers))