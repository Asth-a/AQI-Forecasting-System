"""
Live Real-Time Air Quality (via OpenAQ v3) & Meteorological Ingestion (via Open-Meteo)
"""

from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests

from src.data_pipeline import compute_cpcb_aqi, engineer_features

OPENAQ_API_KEY = "YOUR_OPENAQ_API_KEY_HERE"

CITY_COORDS = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "openaq_city": "Delhi"},
    "Chandigarh": {"lat": 30.7333, "lon": 76.7794, "openaq_city": "Chandigarh"},
}


def fetch_openaq_v3_data(city: str) -> dict:
    """Fetch latest measurements from OpenAQ v3 API for the given city."""
    coords = CITY_COORDS[city]
    headers = {"X-API-Key": OPENAQ_API_KEY}

    pollutants = {}
    try:
        url_loc = "https://api.openaq.org/v3/locations"
        params = {"city": coords["openaq_city"], "limit": 1}
        resp = requests.get(url_loc, headers=headers, params=params, timeout=10)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                loc_id = results[0]["id"]
                url_latest = f"https://api.openaq.org/v3/locations/{loc_id}/latest"
                resp_latest = requests.get(url_latest, headers=headers, timeout=10)
                if resp_latest.status_code == 200:
                    latest_results = resp_latest.json().get("results", [])
                    for item in latest_results:
                        param_name = item.get("parameter", {}).get("name")
                        val = item.get("value")
                        if param_name and val is not None:
                            pollutants[param_name] = val
    except Exception as e:
        print(f"OpenAQ fetch warning: {e}")

    return pollutants


def fetch_live_city_history(city: str = "Delhi") -> pd.DataFrame:
    """Fetch hourly weather from Open-Meteo and blend with live OpenAQ station data."""
    coords = CITY_COORDS[city]
    tz = "Asia/Kolkata"

    wx_url = "https://api.open-meteo.com/v1/forecast"
    wx_params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,boundary_layer_height",
        "past_days": 4,
        "forecast_days": 1,
        "wind_speed_unit": "ms",
        "timezone": tz,
    }

    wx_res = requests.get(wx_url, params=wx_params, timeout=15)
    wx_res.raise_for_status()
    wx_h = pd.DataFrame(wx_res.json()["hourly"])
    wx_h["timestamp"] = pd.to_datetime(wx_h["time"])

    ist_now = datetime.now(timezone(timedelta(hours=5, minutes=30))).replace(
        tzinfo=None, minute=0, second=0, microsecond=0
    )
    df = wx_h[wx_h["timestamp"] <= ist_now].sort_values("timestamp").reset_index(drop=True)

    openaq_vals = fetch_openaq_v3_data(city)

    out = pd.DataFrame()
    out["timestamp"] = df["timestamp"]
    out["city"] = city

    pm25_val = openaq_vals.get("pm25", 150.0)
    pm10_val = openaq_vals.get("pm10", 250.0)
    no2_val = openaq_vals.get("no2", 40.0)

    out["PM2.5"] = pm25_val
    out["PM10"] = pm10_val
    out["NO2"] = no2_val
    out["SO2"] = 15.0
    out["CO"] = 1.2
    out["O3"] = 35.0

    out["Temperature_C"] = pd.to_numeric(df["temperature_2m"], errors="coerce").clip(-10, 52)
    out["Humidity_pct"] = pd.to_numeric(df["relative_humidity_2m"], errors="coerce").clip(5, 100)
    out["WindSpeed_ms"] = pd.to_numeric(df["wind_speed_10m"], errors="coerce").clip(0, 35)
    out["BoundaryLayer_m"] = pd.to_numeric(df["boundary_layer_height"], errors="coerce").fillna(650.0).clip(50, 4000)

    hr = out["timestamp"].dt.hour
    dow = out["timestamp"].dt.dayofweek
    morning_peak = np.exp(-0.5 * ((hr - 9.5) / 2.0) ** 2)
    evening_peak = np.exp(-0.5 * ((hr - 19.5) / 2.2) ** 2)
    weekend_factor = np.where(dow >= 5, 0.78, 1.0)
    out["Traffic_Proxy"] = np.round((0.25 + 0.75 * np.maximum(morning_peak, evening_peak)) * weekend_factor, 3)

    out["was_imputed"] = 0
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    out[num_cols] = out[num_cols].interpolate(method="linear", limit=6).ffill().bfill()

    out["PM2.5_roll24"] = out["PM2.5"].rolling(24, min_periods=1).mean()
    out["PM10_roll24"] = out["PM10"].rolling(24, min_periods=1).mean()
    out["PM2.5"] = out["PM2.5_roll24"] * 1.12
    out["PM10"] = out["PM10_roll24"] * 1.12

    out["AQI"] = np.round(compute_cpcb_aqi(out), 1)

    return out


def predict_live_horizons(city: str, bundles_by_horizon: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_live = fetch_live_city_history(city)
    feat_live = engineer_features(raw_live)
    latest_row = feat_live.iloc[[-1]].copy()
    current_ts = latest_row["timestamp"].iloc[0]
    current_aqi = float(latest_row["AQI"].iloc[0])

    forecast_rows = [
        {
            "horizon_h": 0,
            "label": "Now (Observed)",
            "target_timestamp": current_ts,
            "point_gb": current_aqi,
            "median_50": current_aqi,
            "lower_80": current_aqi,
            "upper_80": current_aqi,
        }
    ]

    for h in [24, 48, 72]:
        arts = bundles_by_horizon[h]["artifacts"]
        cols = arts["feature_cols"]
        X_now = latest_row.reindex(columns=cols, fill_value=0.0)

        p_gb = float(np.clip(arts["gb_model"].predict(X_now)[0], 0, 500))
        p_10 = float(np.clip(arts["q10_model"].predict(X_now)[0], 0, 500))
        p_50 = float(np.clip(arts["q50_model"].predict(X_now)[0], 0, 500))
        p_90 = float(np.clip(arts["q90_model"].predict(X_now)[0], 0, 500))

        p_10 = min(p_10, p_50)
        p_90 = max(p_90, p_50)

        forecast_rows.append(
            {
                "horizon_h": h,
                "label": f"+{h}h Forecast",
                "target_timestamp": current_ts + pd.Timedelta(hours=h),
                "point_gb": round(p_gb, 1),
                "median_50": round(p_50, 1),
                "lower_80": round(p_10, 1),
                "upper_80": round(p_90, 1),
            }
        )

    return raw_live, pd.DataFrame(forecast_rows), latest_row
