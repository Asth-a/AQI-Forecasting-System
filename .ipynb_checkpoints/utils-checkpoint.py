"""Shared helpers for the AQI Forecasting & Health Advisory System.

Every notebook and the Streamlit app import from here, so the feature
engineering and advisory rules are written once and stay consistent.
"""
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "docs" / "figures"
for _d in (RAW_DIR, PROC_DIR, MODEL_DIR, RESULTS_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- config
CITY = "Delhi"          # change this (must exist in city_day.csv) and re-run all notebooks
CITY_COORDS = {         # lat, lon used to download weather
    "Delhi": (28.61, 77.21), "Mumbai": (19.07, 72.88), "Kolkata": (22.57, 88.36),
    "Chennai": (13.08, 80.27), "Bengaluru": (12.97, 77.59), "Hyderabad": (17.38, 78.48),
    "Lucknow": (26.85, 80.95), "Ahmedabad": (23.02, 72.57), "Patna": (25.59, 85.14),
    "Jaipur": (26.91, 75.79),
}
SLUG = CITY.lower()
HORIZONS = [1, 2, 3]                 # days ahead (24h, 48h, 72h)
TRAIN_END = "2018-12-31"             # chronological split - never shuffle time series
VAL_END = "2019-06-30"               # test = everything after this date
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
WEATHER = ["temp", "humidity", "wind", "precip", "pressure"]
DIWALI = ["2014-10-23", "2015-11-11", "2016-10-30", "2017-10-19",
          "2018-11-07", "2019-10-27", "2020-11-14"]

CLEAN_PATH = PROC_DIR / f"{SLUG}_clean.csv"
FEATURE_PATH = PROC_DIR / f"{SLUG}_features.csv"


# ---------------------------------------------------------- AQI categories
AQI_BINS = [50, 100, 200, 300, 400]   # CPCB (India) upper limits
AQI_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
AQI_COLORS = ["#00a651", "#8dc63f", "#f7e000", "#f58220", "#ed1c24", "#7e0023"]


def aqi_category_index(value):
    """0..5 index of the CPCB category for an AQI value."""
    return int(np.searchsorted(AQI_BINS, max(float(value), 0.0), side="left"))


def aqi_category(value):
    return AQI_LABELS[aqi_category_index(value)]


# -------------------------------------------------------- feature engineering
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build predictors from a *daily, gap-free-index* frame.

    Expected columns: AQI, PM2.5, PM10, NO2, SO2, CO, O3, temp, humidity,
    wind, precip, pressure.  Every feature on row t only uses information
    available at the end of day t (no look-ahead leakage).
    """
    f = pd.DataFrame(index=df.index)
    aqi = df["AQI"]

    # 1) AQI memory
    for lag in (0, 1, 2, 3, 7, 14):
        f[f"aqi_lag{lag}"] = aqi.shift(lag)
    for w in (3, 7, 14):
        f[f"aqi_mean{w}"] = aqi.rolling(w).mean()
    f["aqi_std7"] = aqi.rolling(7).std()
    f["aqi_trend3"] = aqi - aqi.shift(3)

    # 2) Pollutants today (+ short smoothing for PM2.5)
    for p in POLLUTANTS:
        f[p.lower().replace(".", "")] = df[p]
    f["pm25_mean3"] = df["PM2.5"].rolling(3).mean()
    f["pm_ratio"] = df["PM2.5"] / df["PM10"].replace(0, np.nan)

    # 3) Traffic proxies (no live traffic data: use NO2/CO behaviour + weekday)
    f["traffic_proxy"] = df["NO2"].rolling(3).mean() / df["NO2"].rolling(28, min_periods=7).mean()
    f["co_mean3"] = df["CO"].rolling(3).mean()

    # 4) Meteorology
    for c in WEATHER:
        f[c] = df[c]
    f["wind_mean3"] = df["wind"].rolling(3).mean()
    f["precip_sum3"] = df["precip"].rolling(3).sum()
    f["temp_change1"] = df["temp"].diff()
    f["pressure_change1"] = df["pressure"].diff()
    f["humidity_mean3"] = df["humidity"].rolling(3).mean()

    # 5) Seasonality / calendar
    idx = df.index
    f["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    f["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    f["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    f["dayofweek"] = idx.dayofweek
    f["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    # 6) Pollution event history
    n = np.arange(len(aqi))
    last_severe = pd.Series(np.where(aqi > 300, n, np.nan), index=idx).ffill()
    f["days_since_severe"] = (n - last_severe).fillna(365).clip(upper=365)
    f["poor_days_14"] = (aqi > 200).astype(float).rolling(14).sum()
    diwali = pd.to_datetime(DIWALI)
    dist = np.array([min(((d - x).days for x in diwali), key=abs) for d in idx], dtype=float)
    f["diwali_window"] = ((dist >= -3) & (dist <= 7)).astype(int)
    md = idx.month * 100 + idx.day
    f["stubble_season"] = ((md >= 1015) & (md <= 1130)).astype(int)   # Oct 15 - Nov 30
    return f


def add_targets(features: pd.DataFrame, aqi: pd.Series) -> pd.DataFrame:
    out = features.copy()
    for h in HORIZONS:
        out[f"y_h{h}"] = aqi.shift(-h)
    return out


# ----------------------------------------------------- rule-based advisory
GROUPS = ["General public", "Asthma / respiratory patients", "Children",
          "Elderly", "Outdoor workers"]

ADVICE = {
    "General public": [
        "Air is clean. Enjoy normal outdoor activity.",
        "Air is acceptable. Very sensitive people may notice mild effects; otherwise carry on as usual.",
        "Possible mild breathing discomfort on long exposure. Prefer lighter exercise if you feel any irritation.",
        "Breathing discomfort is likely on prolonged exposure. Shorten outdoor exercise and avoid busy roads.",
        "Respiratory illness on prolonged exposure. Avoid outdoor exercise; wear a fitted N95 mask if you must go out.",
        "Health emergency conditions. Stay indoors with windows closed, use an air purifier if available, avoid all outdoor exertion.",
    ],
    "Asthma / respiratory patients": [
        "Good conditions. Keep your reliever inhaler with you as usual.",
        "Generally fine. Keep your inhaler handy and watch for symptoms.",
        "Symptoms may appear. Limit long outdoor exertion and keep your reliever inhaler within reach.",
        "High risk of attacks. Stay mostly indoors, follow your asthma action plan, and wear an N95 mask outdoors.",
        "Very high risk. Stay indoors, keep windows shut, take preventer medication as prescribed and contact your doctor if symptoms worsen.",
        "Severe risk. Do not go outdoors. Keep medication ready and seek medical help early if breathing worsens.",
    ],
    "Children": [
        "Safe for all outdoor play.",
        "Safe for play. Watch any child with known sensitivities.",
        "Fine for school and play; reduce very strenuous outdoor sport if a child coughs or wheezes.",
        "Move sports and PE indoors. Limit outdoor play to short periods.",
        "Keep children indoors. Cancel outdoor activities and avoid travel in open vehicles or on foot near traffic.",
        "Keep children indoors with windows closed. Consider school closure guidance from local authorities.",
    ],
    "Elderly": [
        "Safe for walks and normal activity.",
        "Safe for normal activity; those with heart or lung disease should stay alert to symptoms.",
        "Prefer short, gentle walks. Take rest breaks and stay hydrated.",
        "Walk indoors or at midday when air is typically better. Avoid busy roads.",
        "Stay indoors. Avoid exertion; contact a doctor for chest tightness, palpitations or breathlessness.",
        "Remain indoors in a purified or sealed room. Seek care promptly for any chest or breathing symptoms.",
    ],
    "Outdoor workers": [
        "No restrictions.",
        "No restrictions. Stay hydrated.",
        "Take regular breaks in cleaner air where possible; consider a mask for dusty tasks.",
        "Wear an N95/FFP2 mask. Take frequent breaks away from the pollution source and reduce heavy exertion.",
        "Use an N95/FFP2 mask throughout, cut heavy physical work to a minimum, and ask for shift changes if possible.",
        "Avoid outdoor work where at all possible. If unavoidable: N95 mask, short shifts, long breaks indoors.",
    ],
}


def advisory(group: str, aqi_low: float, aqi_pred: float, aqi_high: float) -> dict:
    """Interpretable rule engine.  Uses the *upper* interval bound to pick the
    advice (precautionary) and states the uncertainty explicitly."""
    i_low, i_pred, i_high = (aqi_category_index(v) for v in (aqi_low, aqi_pred, aqi_high))
    plan = max(i_pred, i_high)
    notes = []
    if i_high > i_pred:
        notes.append(f"Uncertainty: AQI could plausibly reach **{AQI_LABELS[i_high]}** "
                     f"(upper estimate {aqi_high:.0f}). The advice below is written for that case.")
    if i_low < i_pred:
        notes.append(f"It could also turn out better, down to **{AQI_LABELS[i_low]}** "
                     f"(lower estimate {aqi_low:.0f}).")
    width = aqi_high - aqi_low
    confidence = "high" if width < 0.4 * max(aqi_pred, 1) else ("medium" if width < 0.8 * max(aqi_pred, 1) else "low")
    return {"category": AQI_LABELS[i_pred], "planning_category": AQI_LABELS[plan],
            "level": plan, "advice": ADVICE[group][plan], "notes": notes,
            "confidence": confidence}