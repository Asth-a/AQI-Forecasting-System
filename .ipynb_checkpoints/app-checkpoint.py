"""Streamlit dashboard: AQI forecast (24/48/72 h) + rule-based health advisory.

Run with:   streamlit run app.py
"""
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (CITY, FEATURE_PATH, MODEL_DIR, RESULTS_DIR, FIG_DIR, HORIZONS,
                   GROUPS, AQI_BINS, AQI_LABELS, AQI_COLORS, aqi_category_index, advisory)

st.set_page_config(page_title="AQI Forecast & Health Advisory", page_icon="🌫️", layout="wide")


@st.cache_data
def load_data():
    feature_cols = json.load(open(MODEL_DIR / "feature_cols.json"))
    data = pd.read_csv(FEATURE_PATH, index_col=0, parse_dates=True).dropna(subset=feature_cols)
    return data, feature_cols


@st.cache_resource
def load_models():
    info = json.load(open(MODEL_DIR / "intervals.json"))
    models = {int(h): joblib.load(MODEL_DIR / f"{v['model']}_h{h}.joblib") for h, v in info.items()}
    return models, {int(h): v for h, v in info.items()}


try:
    data, feature_cols = load_data()
    models, info = load_models()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Run notebooks 01-04 first (see README).")
    st.stop()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("Settings")
st.sidebar.write(f"**City:** {CITY}")
min_d, max_d = data.index.min().date(), data.index.max().date()
issue_date = st.sidebar.date_input("Forecast issued at end of day", value=max_d,
                                   min_value=min_d, max_value=max_d)
group = st.sidebar.selectbox("Who is the advice for?", GROUPS)
st.sidebar.caption("The public CPCB dataset ends in 2020, so 'today' is the last day in the data. "
                   "Pick an earlier date to replay a past forecast and compare it with what happened.")

issue_ts = pd.Timestamp(issue_date)
if issue_ts not in data.index:
    st.warning("No data for that date - choose another one.")
    st.stop()
row = data.loc[[issue_ts]]

# ---------------------------------------------------------------- forecasts
forecasts = []
for h in HORIZONS:
    pred = float(max(models[h].predict(row[feature_cols])[0], 1))
    lo = pred * np.exp(info[h]["log_q_low"])
    hi = pred * np.exp(info[h]["log_q_high"])
    actual = row[f"y_h{h}"].iloc[0] if f"y_h{h}" in row and pd.notna(row[f"y_h{h}"].iloc[0]) else None
    forecasts.append({"h": h, "date": issue_ts + pd.Timedelta(days=h), "pred": pred,
                      "lo": lo, "hi": hi, "actual": actual})

st.title(f"🌫️ Air Quality Forecast & Health Advisory - {CITY}")
st.caption(f"Forecast issued at the end of {issue_ts:%A, %d %B %Y}. "
           f"Today's observed AQI: **{row['aqi_lag0'].iloc[0]:.0f}** ({AQI_LABELS[aqi_category_index(row['aqi_lag0'].iloc[0])]})")

# ------------------------------------------------------------------ metrics
cols = st.columns(len(HORIZONS))
for c, fc in zip(cols, forecasts):
    cat_i = aqi_category_index(fc["pred"])
    with c:
        st.markdown(f"#### Day +{fc['h']} &nbsp; <small>{fc['date']:%a %d %b}</small>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:2.4rem;font-weight:700;color:{AQI_COLORS[cat_i]}'>"
                    f"{fc['pred']:.0f}</div>", unsafe_allow_html=True)
        st.markdown(f"**{AQI_LABELS[cat_i]}**")
        st.caption(f"80% range: {fc['lo']:.0f} - {fc['hi']:.0f}")
        if fc["actual"] is not None:
            st.caption(f"Actual (replay): {fc['actual']:.0f}")

# -------------------------------------------------------------------- chart
hist = data["aqi_lag0"].loc[issue_ts - pd.Timedelta(days=30): issue_ts]
fig = go.Figure()
bounds = [0] + AQI_BINS + [500]
ymax = max(500 if max(f["hi"] for f in forecasts) > 400 else 420, 0)
for i in range(6):
    fig.add_hrect(y0=bounds[i], y1=bounds[i + 1], fillcolor=AQI_COLORS[i], opacity=0.10, line_width=0)
fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines+markers", name="Observed",
                         line=dict(color="black")))
fx = [fc["date"] for fc in forecasts]
fig.add_trace(go.Scatter(x=[issue_ts] + fx, y=[hist.iloc[-1]] + [f["pred"] for f in forecasts],
                         mode="lines+markers", name="Forecast", line=dict(color="#c0392b", dash="dash")))
fig.add_trace(go.Scatter(x=fx + fx[::-1], y=[f["hi"] for f in forecasts] + [f["lo"] for f in forecasts][::-1],
                         fill="toself", fillcolor="rgba(192,57,43,0.2)", line=dict(width=0),
                         name="80% interval", hoverinfo="skip"))
known = [(f["date"], f["actual"]) for f in forecasts if f["actual"] is not None]
if known:
    fig.add_trace(go.Scatter(x=[k[0] for k in known], y=[k[1] for k in known], mode="markers",
                             name="Actual (replay)", marker=dict(color="black", symbol="x", size=11)))
fig.update_layout(height=420, yaxis_title="AQI", yaxis_range=[0, ymax], margin=dict(t=20),
                  legend=dict(orientation="h", y=-0.15))
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------- advisory
st.header(f"Health advisory - {group}")
worst = max(forecasts, key=lambda f: advisory(group, f["lo"], f["pred"], f["hi"])["level"])
w = advisory(group, worst["lo"], worst["pred"], worst["hi"])
box = st.error if w["level"] >= 4 else (st.warning if w["level"] >= 2 else st.success)
box(f"**Highest planning level in the next 72 h: {w['planning_category']}** "
    f"(expected on {worst['date']:%a %d %b}). {w['advice']}")

for fc in forecasts:
    a = advisory(group, fc["lo"], fc["pred"], fc["hi"])
    with st.expander(f"Day +{fc['h']} ({fc['date']:%a %d %b}): forecast {a['category']} "
                     f"- planning for {a['planning_category']} - confidence {a['confidence']}",
                     expanded=(fc["h"] == 1)):
        st.write(a["advice"])
        for n in a["notes"]:
            st.markdown(f"- {n}")
        if not a["notes"]:
            st.markdown("- The whole 80% range falls in the same AQI category.")

st.info("Advice is generated by transparent if-then rules on the CPCB AQI categories. "
        "The upper end of the uncertainty range is used for planning, so advice errs on the safe side. "
        "This is an educational tool, not medical advice.")

# ------------------------------------------------------------- transparency
with st.expander("How reliable is this forecast?"):
    st.write("Each horizon uses the model that did best on validation data. The 80% interval is built "
             "from the model's past errors; the table shows how often the real AQI actually fell inside "
             "it on the unseen test period (ideal: about 80%).")
    st.dataframe(pd.DataFrame([{"Horizon": f"Day +{h}", "Model": info[h]["model"],
                                "Test coverage of 80% interval": f"{info[h]['test_coverage']:.0%}"}
                               for h in HORIZONS]), hide_index=True)
    comp = RESULTS_DIR / "model_comparison.csv"
    if comp.exists():
        st.write("**Test MAE by model (lower is better). Persistence = 'tomorrow equals today'.**")
        c = pd.read_csv(comp).pivot(index="model", columns="horizon", values="MAE")
        c.columns = [f"Day +{x}" for x in c.columns]
        st.dataframe(c)

shap_img = FIG_DIR / "shap_group_importance.png"
if shap_img.exists():
    with st.expander("What drives the forecast? (SHAP)"):
        st.image(str(shap_img))
