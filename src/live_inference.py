"""
Live Real-Time Air Quality & Meteorological Ingestion + Multi-Horizon Inference
Uses Open-Meteo Free Air Quality & Weather APIs (no API key required) to fetch the
trailing 96 hours of observations and run live +24h / +48h / +72h forecasts.
"""

from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests

from src.data_pipeline import compute_cpcb_aqi, engineer_features

CITY_COORDS = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Chandigarh": {"lat": 30.7333, "lon": 76.7794},
}


def fetch_live_city_history(city: str = "Delhi") -> pd.DataFrame:
    """Fetch last 96 hours of hourly air quality & weather data up to the current IST hour."""
    coords = CITY_COORDS[city]
    tz = "Asia/Kolkata"

    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone",
        "past_days": 4,
        "forecast_days": 1,
        "timezone": tz,
    }

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

    aq_res = requests.get(aq_url, params=aq_params, timeout=15)
    aq_res.raise_for_status()
    wx_res = requests.get(wx_url, params=wx_params, timeout=15)
    wx_res.raise_for_status()

    aq_h = pd.DataFrame(aq_res.json()["hourly"])
    wx_h = pd.DataFrame(wx_res.json()["hourly"])

    df = pd.merge(aq_h, wx_h, on="time", how="inner")
    df["timestamp"] = pd.to_datetime(df["time"])

    # Keep strictly up to the current IST hour
    ist_now = datetime.now(timezone(timedelta(hours=5, minutes=30))).replace(
        tzinfo=None, minute=0, second=0, microsecond=0
    )
    df = df[df["timestamp"] <= ist_now].sort_values("timestamp").reset_index(drop=True)

    # Map API fields to exact training schema
    out = pd.DataFrame()
    out["timestamp"] = df["timestamp"]
    out["city"] = city
    out["PM2.5"] = pd.to_numeric(df["pm2_5"], errors="coerce").clip(0, 999)
    out["PM10"] = pd.to_numeric(df["pm10"], errors="coerce").clip(0, 1200)
    out["NO2"] = pd.to_numeric(df["nitrogen_dioxide"], errors="coerce").clip(0, 500)
    out["SO2"] = pd.to_numeric(df["sulphur_dioxide"], errors="coerce").clip(0, 300)
    # Open-Meteo CO is in ug/m3; convert to mg/m3 (CPCB standard unit)
    out["CO"] = (pd.to_numeric(df["carbon_monoxide"], errors="coerce") / 1000.0).clip(0, 20)
    out["O3"] = pd.to_numeric(df["ozone"], errors="coerce").clip(0, 300)
    out["Temperature_C"] = pd.to_numeric(df["temperature_2m"], errors="coerce").clip(-10, 52)
    out["Humidity_pct"] = pd.to_numeric(df["relative_humidity_2m"], errors="coerce").clip(5, 100)
    out["WindSpeed_ms"] = pd.to_numeric(df["wind_speed_10m"], errors="coerce").clip(0, 35)
    out["BoundaryLayer_m"] = pd.to_numeric(df["boundary_layer_height"], errors="coerce").fillna(650.0).clip(50, 4000)

    # Diurnal traffic proxy matching urban Indian rush-hour profile
    hr = out["timestamp"].dt.hour
    dow = out["timestamp"].dt.dayofweek
    morning_peak = np.exp(-0.5 * ((hr - 9.5) / 2.0) ** 2)
    evening_peak = np.exp(-0.5 * ((hr - 19.5) / 2.2) ** 2)
    weekend_factor = np.where(dow >= 5, 0.78, 1.0)
    out["Traffic_Proxy"] = np.round((0.25 + 0.75 * np.maximum(morning_peak, evening_peak)) * weekend_factor, 3)

    out["was_imputed"] = out[["PM2.5", "PM10", "NO2"]].isna().any(axis=1).astype(int)
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    out[num_cols] = out[num_cols].interpolate(method="linear", limit=6).ffill().bfill()
    out["AQI"] = np.round(compute_cpcb_aqi(out), 1)

    return out


def predict_live_horizons(city: str, bundles_by_horizon: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate live +24h, +48h, and +72h AQI predictions from the current timestamp.
    Returns (live_history_df, live_forecast_df, latest_feature_row).
    """
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