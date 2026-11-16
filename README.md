# Air Quality Forecasting & Personalised Health Advisory System (Project 44)

**Course:** CSE276 — Artificial Intelligence Foundations (Session 2026–27)  
**Deliverable:** End-to-end 24–72 hour Air Quality Index (AQI) forecasting system with quantile uncertainty estimation, SHAP feature explainability, an auditable rule-based health advisory engine, a complete Jupyter Notebook, and an interactive Streamlit web dashboard.

---

## 📁 Repository Structure

```text
air-quality-forecasting-health-advisory/
├── app/
│   └── streamlit_app.py                         # Interactive Streamlit deployment dashboard
├── data/
│   ├── raw/
│   │   └── cpcb_openaq_hourly_2024_2025.csv     # 2-year hourly CPCB/OpenAQ dataset (Delhi & Chandigarh)
│   └── processed/
│       └── delhi_features_targets.csv           # Cleaned, gap-imputed & feature-engineered dataset
├── models/
│   ├── benchmark_metrics_24h.csv                # +24h model comparison vs Persistence Baseline
│   ├── benchmark_metrics_48h.csv                # +48h model comparison vs Persistence Baseline
│   ├── benchmark_metrics_72h.csv                # +72h model comparison vs Persistence Baseline
│   └── feature_importance_summary.csv           # Global feature importance rankings
├── notebooks/
│   └── 01_AQI_Forecasting_and_Health_Advisory.ipynb  # Complete end-to-end Jupyter Notebook
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py                         # CPCB AQI calculation, gap handling & feature engineering
│   ├── models.py                                # Persistence, Ridge, RF, XGBoost, Holt-Winters & Quantile PI
│   ├── explainability.py                        # SHAP & local/global feature attribution
│   └── health_advisory.py                       # Auditable rule-based health guidance for 5 user groups
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook
```bash
jupyter notebook notebooks/01_AQI_Forecasting_and_Health_Advisory.ipynb
```

### 3. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## 🧠 System Architecture & Core Modules

1. **Data Acquisition & Gap Handling (`src/data_pipeline.py`)**:
   - Supports OpenAQ v3 live API queries alongside the bundled 35,088-row hourly dataset (`2024–2025`) covering **Delhi** and **Chandigarh**.
   - Performs physical sensor bound clipping, continuous hourly re-indexing, short-gap linear interpolation ($\le 6	ext{h}$), and exact Indian **CPCB National AQI (NAQI)** sub-index computation.
2. **Domain Feature Engineering (`src/data_pipeline.py`)**:
   - **Temporal/Seasonal:** Cyclical encodings (`hour_sin/cos`, `doy_sin/cos`), weekend flags, and winter inversion/stubble-burning season flags.
   - **Meteorological Dispersion:** Ventilation coefficient ($	ext{Wind Speed} 	imes 	ext{Boundary Layer Height}$), stagnation index, and dew-point proxy.
   - **Traffic & Event History:** Rush-hour traffic proxy interactions and leakage-free autoregressive lags (`1h` to `72h`) and rolling maxima/volatility.
3. **Multi-Horizon Forecasting & Uncertainty (`src/models.py`)**:
   - Compares **Persistence Baseline**, **Ridge Regression**, **Random Forest**, **Gradient Boosting (XGBoost / HistGBM)**, and **Holt-Winters Time-Series** across **+24h, +48h, and +72h** horizons using a strict chronological split.
   - Outputs calibrated **80% Prediction Intervals (10th–90th percentile)** via Quantile Gradient Boosting rather than single overconfident point values.
4. **Explainability & Rule-Based Health Engine (`src/explainability.py`, `src/health_advisory.py`)**:
   - Generates tailored, auditable medical/activity advisories for **Asthmatics**, **Children**, **Elderly (65+)**, **Outdoor Workers**, and the **General Public**, automatically escalating precautions when the upper uncertainty bound crosses into a worse AQI category.
