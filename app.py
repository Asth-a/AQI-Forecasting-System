"""Streamlit dashboard: AQI forecast (24/48/72 h) + rule-based health advisory.

Run with:   python -m streamlit run app.py
"""
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (CITY, FEATURE_PATH, MODEL_DIR, RESULTS_DIR, FIG_DIR, HORIZONS,
                   GROUPS, AQI_LABELS, AQI_COLORS, aqi_category_index, advisory)

st.set_page_config(page_title=f"AQI Forecast - {CITY}", page_icon="🌫️", layout="wide")

TEXT = ["#ffffff", "#11330a", "#3a3300", "#ffffff", "#ffffff", "#ffffff"]   # readable text per category
GROUP_ICONS = {"General public": "👥", "Asthma / respiratory patients": "🫁", "Children": "🧒",
               "Elderly": "🧓", "Outdoor workers": "👷"}

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 1200px;}
.hero {background: linear-gradient(120deg,#0f2027 0%,#203a43 50%,#2c5364 100%); color:#fff;
       padding: 1.5rem 1.8rem; border-radius: 18px; margin-bottom: 1.1rem;}
.hero h1 {margin:0; font-size:2rem; color:#fff; line-height:1.2;}
.hero p {margin:.35rem 0 0; opacity:.85; font-size:.98rem;}
.card {border-radius:18px; padding:1.1rem 1.2rem; box-shadow:0 6px 18px rgba(0,0,0,.18); min-height:210px;}
.card .ct {font-weight:700; font-size:1.05rem;}
.card .cs {opacity:.85; font-size:.85rem; margin-bottom:.2rem;}
.card .cv {font-size:3.2rem; font-weight:800; line-height:1.05;}
.card .cc {font-weight:700; font-size:1.05rem; margin-bottom:.35rem;}
.card .cl {font-size:.85rem; opacity:.92;}
.scale {display:flex; height:30px; border-radius:8px; overflow:hidden; margin-top:2px;}
.scale div {display:flex; align-items:center; justify-content:center; font-size:.72rem; font-weight:600;}
.scalewrap {position:relative; padding-top:30px; margin:.4rem 0 1.2rem;}
.mark {position:absolute; top:0; transform:translateX(-50%); text-align:center; font-size:.75rem; font-weight:700;}
.mark:after {content:"▼"; display:block; line-height:.8; font-size:.8rem;}
.adv {border-radius:14px; padding:.9rem 1.1rem; margin-bottom:.7rem; background:rgba(128,128,128,.10);}
.adv .at {font-weight:700; margin-bottom:.15rem;}
.pill {display:inline-block; padding:.12rem .6rem; border-radius:999px; font-size:.78rem; font-weight:700; margin-left:.4rem;}
.banner {border-radius:16px; padding:1.1rem 1.3rem; margin-bottom:1rem;}
.banner .bt {font-size:1.15rem; font-weight:800;}
.small {font-size:.85rem; opacity:.8;}
</style>
""", unsafe_allow_html=True)


def html(s):
    st.markdown(" ".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def show_plot(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:                       # older Streamlit versions
        st.plotly_chart(fig, use_container_width=True)


def card(title, sub, value, cat_i, line1="", line2=""):
    return (f'<div class="card" style="background:{AQI_COLORS[cat_i]};color:{TEXT[cat_i]}">'
            f'<div class="ct">{title}</div><div class="cs">{sub}</div>'
            f'<div class="cv">{value}</div><div class="cc">{AQI_LABELS[cat_i]}</div>'
            f'<div class="cl">{line1}</div><div class="cl">{line2}</div></div>')


def scale_bar(markers):
    widths = [50, 50, 100, 100, 100, 100]
    names = ["Good", "Satisfactory", "Moderate", "Poor", "Very poor", "Severe"]
    segs = "".join(f'<div style="flex:{w};background:{c};color:{t}">{n}</div>'
                   for w, c, t, n in zip(widths, AQI_COLORS, TEXT, names))
    marks = "".join(f'<div class="mark" style="left:{min(v, 500) / 5:.1f}%">{l}</div>' for l, v in markers)
    html(f'<div class="scalewrap">{marks}<div class="scale">{segs}</div></div>')


# --------------------------------------------------------------------- load
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
st.sidebar.markdown("## ⚙️ Settings")
st.sidebar.write(f"**Station / city:** {CITY}")
min_d, max_d = data.index.min().date(), data.index.max().date()
issue_date = st.sidebar.date_input("Forecast issued at end of day", value=max_d,
                                   min_value=min_d, max_value=max_d)
group = st.sidebar.selectbox("Advice for", GROUPS, format_func=lambda g: f"{GROUP_ICONS[g]} {g}")
st.sidebar.caption("Pick an earlier date to replay a past forecast and compare it with what actually happened.")

issue_ts = pd.Timestamp(issue_date)
if issue_ts not in data.index:
    st.warning("No data for that date - choose another one.")
    st.stop()
row = data.loc[[issue_ts]]
today_aqi = float(row["aqi_lag0"].iloc[0])

# ---------------------------------------------------------------- forecasts
forecasts = []
for h in HORIZONS:
    pred = float(max(models[h].predict(row[feature_cols])[0], 1))
    lo = pred * float(np.exp(info[h]["log_q_low"]))
    hi = pred * float(np.exp(info[h]["log_q_high"]))
    col = f"y_h{h}"
    actual = float(row[col].iloc[0]) if col in row and pd.notna(row[col].iloc[0]) else None
    forecasts.append({"h": h, "date": issue_ts + pd.Timedelta(days=h), "pred": pred,
                      "lo": lo, "hi": hi, "actual": actual})

# --------------------------------------------------------------------- hero
html(f'<div class="hero"><h1>🌫️ Air Quality Forecast & Health Advisory</h1>'
     f'<p>{CITY} · next 72 hours · forecast issued at the end of {issue_ts:%A, %d %B %Y}</p></div>')

age = (pd.Timestamp.today().normalize() - data.index.max()).days
if issue_ts == data.index.max() and age > 7:
    st.info(f"The latest data in the system is from **{data.index.max():%d %b %Y}** ({age} days ago), "
            "so these forecasts are for the days right after that date. Add newer data to update them.")

tab1, tab2, tab3 = st.tabs(["📈 Forecast", "🩺 Health advisory", "🔬 Model & insights"])

# ===================================================================== TAB 1
with tab1:
    cols = st.columns(len(HORIZONS) + 1)
    ci = aqi_category_index(today_aqi)
    with cols[0]:
        html(card("Observed", f"{issue_ts:%a %d %b}", f"{today_aqi:.0f}", ci, "Measured at the station"))
    for c, fc in zip(cols[1:], forecasts):
        i = aqi_category_index(fc["pred"])
        diff = fc["pred"] - today_aqi
        arrow = "▲" if diff > 0 else "▼"
        l2 = f"Actual (replay): {fc['actual']:.0f}" if fc["actual"] is not None else ""
        with c:
            html(card(f"Day +{fc['h']}", f"{fc['date']:%a %d %b}", f"{fc['pred']:.0f}", i,
                      f"80% range: {fc['lo']:.0f} – {fc['hi']:.0f}",
                      f"{arrow} {abs(diff):.0f} vs observed" if not l2 else f"{arrow} {abs(diff):.0f} vs observed · {l2}"))

    st.markdown("#### Where the forecasts fall on the AQI scale")
    scale_bar([("Now", today_aqi)] + [(f"D+{f['h']}", f["pred"]) for f in forecasts])

    hist = data["aqi_lag0"].loc[issue_ts - pd.Timedelta(days=30): issue_ts]
    top = max(max(f["hi"] for f in forecasts), float(hist.max())) * 1.12
    fig = go.Figure()
    edges = [0, 50, 100, 200, 300, 400, 10000]
    for i in range(6):
        fig.add_hrect(y0=edges[i], y1=min(edges[i + 1], top * 2), fillcolor=AQI_COLORS[i], opacity=0.12, line_width=0)
    fx = [fc["date"] for fc in forecasts]
    fig.add_trace(go.Scatter(x=fx + fx[::-1], y=[f["hi"] for f in forecasts] + [f["lo"] for f in forecasts][::-1],
                             fill="toself", fillcolor="rgba(192,57,43,0.22)", line=dict(width=0),
                             name="80% interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines+markers", name="Observed",
                             line=dict(color="#1f2d3d", width=2.5), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=[issue_ts] + fx, y=[today_aqi] + [f["pred"] for f in forecasts],
                             mode="lines+markers", name="Forecast",
                             line=dict(color="#c0392b", width=3, dash="dash"), marker=dict(size=9)))
    known = [(f["date"], f["actual"]) for f in forecasts if f["actual"] is not None]
    if known:
        fig.add_trace(go.Scatter(x=[k[0] for k in known], y=[k[1] for k in known], mode="markers",
                                 name="Actual (replay)", marker=dict(color="black", symbol="x", size=12)))
    fig.update_layout(height=430, template="plotly_white", yaxis_title="AQI", yaxis_range=[0, max(top, 150)],
                      margin=dict(t=30, b=10), legend=dict(orientation="h", y=-0.15),
                      title="Last 30 days and the 72-hour forecast")
    show_plot(fig)

# ===================================================================== TAB 2
with tab2:
    plans = [advisory(group, f["lo"], f["pred"], f["hi"]) for f in forecasts]
    wi = max(range(len(plans)), key=lambda k: plans[k]["level"])
    w, wf = plans[wi], forecasts[wi]
    wc = AQI_COLORS[w["level"]]
    html(f'<div class="banner" style="background:{wc};color:{TEXT[w["level"]]}">'
         f'<div class="bt">{GROUP_ICONS[group]} {group}: highest planning level in the next 72 h is '
         f'{w["planning_category"]}</div>'
         f'<div>Expected on {wf["date"]:%A %d %b} · {w["advice"]}</div></div>')

    st.markdown("#### Day by day")
    dcols = st.columns(len(forecasts))
    for c, fc, a in zip(dcols, forecasts, plans):
        with c:
            col = AQI_COLORS[a["level"]]
            notes = "".join(f"<div class='small'>• {n.replace('**', '')}</div>" for n in a["notes"]) \
                or "<div class='small'>• The whole 80% range sits in one AQI category.</div>"
            html(f'<div class="adv" style="border-left:7px solid {col}">'
                 f'<div class="at">Day +{fc["h"]} · {fc["date"]:%a %d %b}'
                 f'<span class="pill" style="background:{col};color:{TEXT[a["level"]]}">{a["planning_category"]}</span></div>'
                 f'<div class="small">Forecast: {a["category"]} · confidence: {a["confidence"]}</div>'
                 f'<div style="margin:.5rem 0">{a["advice"]}</div>{notes}</div>')

    st.markdown("#### Advice for every group")
    pick = st.radio("Show the advice for", [f"Day +{f['h']}" for f in forecasts], horizontal=True)
    k = int(pick.split("+")[1]) - 1
    fk = forecasts[k]
    for g in GROUPS:
        a = advisory(g, fk["lo"], fk["pred"], fk["hi"])
        col = AQI_COLORS[a["level"]]
        html(f'<div class="adv" style="border-left:7px solid {col}"><div class="at">{GROUP_ICONS[g]} {g}</div>'
             f'<div>{a["advice"]}</div></div>')

    st.caption("Advice comes from transparent if-then rules on the CPCB AQI categories. The upper end of the "
               "uncertainty range is used for planning, so the advice errs on the safe side. "
               "This is an educational tool, not medical advice.")

# ===================================================================== TAB 3
with tab3:
    st.markdown("#### How reliable is this forecast?")
    st.write("Each horizon uses the model that did best on validation data. The 80% interval is built from "
             "the model's past errors; coverage shows how often the real AQI fell inside it on the unseen "
             "test period (ideal: about 80%).")
    m1, m2, m3 = st.columns(3)
    for c, h in zip((m1, m2, m3), HORIZONS):
        c.metric(f"Day +{h} · {info[h]['model']}", f"{info[h]['test_coverage']:.0%}", "interval coverage on test data",
                 delta_color="off")

    comp = RESULTS_DIR / "model_comparison.csv"
    if comp.exists():
        st.markdown("#### Models vs the persistence baseline (test MAE, lower is better)")
        c = pd.read_csv(comp)
        piv = c.pivot(index="model", columns="horizon", values="MAE")
        fig = go.Figure()
        for h in piv.columns:
            fig.add_trace(go.Bar(x=piv.index, y=piv[h], name=f"Day +{h}"))
        fig.update_layout(barmode="group", template="plotly_white", height=380, yaxis_title="MAE (AQI points)",
                          margin=dict(t=20), legend=dict(orientation="h", y=-0.2))
        show_plot(fig)

    tp = RESULTS_DIR / "test_predictions.csv"
    if tp.exists():
        st.markdown("#### 24-hour forecasts on the unseen test period")
        d = pd.read_csv(tp, parse_dates=["date"])
        d = d[d["horizon"] == 1].set_index("date")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d.index.tolist() + d.index.tolist()[::-1],
                                 y=d["hi"].tolist() + d["lo"].tolist()[::-1], fill="toself",
                                 fillcolor="rgba(192,57,43,0.2)", line=dict(width=0), name="80% interval",
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=d.index, y=d["y_true"], name="Actual", line=dict(color="black", width=1.5)))
        fig.add_trace(go.Scatter(x=d.index, y=d["best_pred"], name="Forecast", line=dict(color="#c0392b", width=1.5)))
        fig.update_layout(template="plotly_white", height=380, yaxis_title="AQI", margin=dict(t=20),
                          legend=dict(orientation="h", y=-0.2))
        show_plot(fig)

    shap_csv = RESULTS_DIR / "shap_importance.csv"
    if shap_csv.exists():
        st.markdown("#### What drives the forecast? (SHAP, top 10 features for Day +1)")
        s = pd.read_csv(shap_csv, index_col=0)["Day +1"].sort_values().tail(10)
        fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color="#2c5364"))
        fig.update_layout(template="plotly_white", height=380, xaxis_title="mean |SHAP value|", margin=dict(t=20))
        show_plot(fig)
    g_img = FIG_DIR / "shap_group_importance.png"
    if g_img.exists():
        with st.expander("Importance by feature group"):
            st.image(str(g_img))
