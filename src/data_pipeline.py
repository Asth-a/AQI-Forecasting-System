"""
Module 1 & 2: Data Acquisition, Cleaning, Gap Handling & Feature Engineering
Supports CPCB / OpenAQ live API ingestion with automatic fallback to local historical dataset.
"""

import os
import requests
import numpy as np
import pandas as pd


def compute_cpcb_aqi(df: pd.DataFrame) -> pd.Series:
    """Compute Indian CPCB National Air Quality Index (NAQI) from pollutant concentrations."""
    pm25 = df["PM2.5"].values
    pm10 = df["PM10"].values
    no2 = df["NO2"].values

    si_pm25 = np.select(
        [pm25 <= 30, pm25 <= 60, pm25 <= 90, pm25 <= 120, pm25 <= 250],
        [
            pm25 * 50 / 30,
            50 + (pm25 - 30) * 50 / 30,
            100 + (pm25 - 60) * 100 / 30,
            200 + (pm25 - 90) * 100 / 30,
            300 + (pm25 - 120) * 100 / 130,
        ],
        default=400 + (pm25 - 250) * 100 / 130,
    )
    si_pm10 = np.select(
        [pm10 <= 50, pm10 <= 100, pm10 <= 250, pm10 <= 350, pm10 <= 430],
        [
            pm10,
            pm10,
            100 + (pm10 - 100) * 100 / 150,
            200 + (pm10 - 250),
            300 + (pm10 - 350) * 100 / 80,
        ],
        default=400 + (pm10 - 430) * 100 / 80,
    )
    si_no2 = np.select(
        [no2 <= 40, no2 <= 80, no2 <= 180, no2 <= 280, no2 <= 400],
        [
            no2 * 50 / 40,
            50 + (no2 - 40) * 50 / 40,
            100 + (no2 - 80),
            200 + (no2 - 180),
            300 + (no2 - 280) * 100 / 120,
        ],
        default=400 + (no2 - 400) * 100 / 120,
    )
    return pd.Series(np.clip(np.maximum.reduce([si_pm25, si_pm10, si_no2]), 0, 500), index=df.index)


def fetch_openaq_latest(city: str = "Delhi", limit: int = 100) -> pd.DataFrame:
    """Attempt to fetch recent air quality measurements from OpenAQ v3 public API."""
    url = "https://api.openaq.org/v3/locations"
    params = {"city": city, "limit": 5}
    try:
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            data = resp.json().get("results", [])
            if data:
                return pd.json_normalize(data)
    except Exception:
        pass
    return pd.DataFrame()


def load_and_clean_data(filepath: str, city: str = "Delhi") -> pd.DataFrame:
    """Load raw hourly air quality & meteorological data, handle sensor gaps, and validate physical bounds."""
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df = df[df["city"].str.lower() == city.lower()].sort_values("timestamp").reset_index(drop=True)

    # Ensure continuous hourly index
    full_idx = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    df = df.set_index("timestamp").reindex(full_idx).rename_axis("timestamp").reset_index()
    df["city"] = city

    # Record gap flags before imputation
    df["was_imputed"] = df[["PM2.5", "PM10", "NO2"]].isna().any(axis=1).astype(int)

    # Physical range clipping (remove sensor spikes)
    bounds = {
        "PM2.5": (0, 999),
        "PM10": (0, 1200),
        "NO2": (0, 500),
        "SO2": (0, 300),
        "CO": (0, 20),
        "O3": (0, 300),
        "Temperature_C": (-10, 52),
        "Humidity_pct": (5, 100),
        "WindSpeed_ms": (0, 35),
    }
    for col, (low, high) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=low, upper=high)

    # Time-aware linear interpolation for short gaps (<= 6 hours), followed by forward/backward fill
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=6).ffill().bfill()

    # Recompute AQI cleanly after gap handling
    df["AQI"] = np.round(compute_cpcb_aqi(df), 1)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-engineered features from:
    1. Cyclical temporal & seasonal patterns
    2. Meteorological dispersion & stagnation indices
    3. Traffic rush-hour & weekend proxies
    4. Autoregressive lag & rolling pollution episode history
    """
    feat = df.copy()
    hour = feat["timestamp"].dt.hour
    month = feat["timestamp"].dt.month
    dow = feat["timestamp"].dt.dayofweek
    doy = feat["timestamp"].dt.dayofyear

    # 1. Cyclical time encodings
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    feat["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    feat["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    feat["is_weekend"] = (dow >= 5).astype(int)
    feat["is_winter_episode_season"] = month.isin([10, 11, 12, 1]).astype(int)

    # 2. Meteorological dispersion & boundary layer trapping proxies
    feat["ventilation_coeff"] = feat["WindSpeed_ms"] * feat["BoundaryLayer_m"]
    feat["stagnation_index"] = 1.0 / (feat["WindSpeed_ms"] + 0.5)
    feat["temp_humidity_ratio"] = feat["Temperature_C"] / (feat["Humidity_pct"] + 1.0)
    feat["dew_point_proxy"] = feat["Temperature_C"] - ((100.0 - feat["Humidity_pct"]) / 5.0)

    # 3. Traffic rush-hour proxy interactions
    feat["is_rush_hour"] = (((hour >= 8) & (hour <= 10)) | ((hour >= 18) & (hour <= 21))).astype(int)
    feat["traffic_x_stagnation"] = feat["Traffic_Proxy"] * feat["stagnation_index"]

    # 4. Pollution event history: lags & rolling statistics (strictly past values, no data leakage)
    for lag in [1, 2, 3, 6, 12, 24, 48, 72]:
        feat[f"AQI_lag_{lag}h"] = feat["AQI"].shift(lag)
        feat[f"PM25_lag_{lag}h"] = feat["PM2.5"].shift(lag)

    for win in [6, 12, 24, 72]:
        feat[f"AQI_roll_mean_{win}h"] = feat["AQI"].shift(1).rolling(win).mean()
        feat[f"AQI_roll_std_{win}h"] = feat["AQI"].shift(1).rolling(win).std()
        feat[f"AQI_roll_max_{win}h"] = feat["AQI"].shift(1).rolling(win).max()

    feat["AQI_diff_1h"] = feat["AQI"].shift(1) - feat["AQI"].shift(2)
    feat["AQI_diff_24h"] = feat["AQI"].shift(1) - feat["AQI"].shift(25)
    feat["severe_episode_flag_24h"] = (feat["AQI_roll_max_24h"] > 300).astype(int)

    feat = feat.dropna().reset_index(drop=True)
    return feat


def create_multi_horizon_targets(df: pd.DataFrame, horizons=(24, 48, 72)) -> pd.DataFrame:
    """Create target columns AQI_t+24h, AQI_t+48h, AQI_t+72h for direct multi-step forecasting."""
    out = df.copy()
    for h in horizons:
        out[f"target_AQI_{h}h"] = out["AQI"].shift(-h)
    out = out.dropna().reset_index(drop=True)
    return out
