# AQI Forecasting & Personalised Health Advisory System

Forecasts the Air Quality Index (CPCB scale) **24, 48 and 72 hours ahead** for an Indian city and turns the
forecast into **rule-based, group-specific health advice** (general public, asthma patients, children, elderly,
outdoor workers) with **honest uncertainty ranges**.

## Pipeline
| Notebook | What it does |
|---|---|
| `01c_OpenAQ_Data_Collection` | Reads your OpenAQ station export, builds daily CPCB AQI, downloads matching weather from Open-Meteo (older alternatives: `01` Kaggle file, `01b` Open-Meteo modelled data) |
| `02_Data_Cleaning` | Gap handling, outlier capping, exploratory analysis |
| `03_Feature_Engineering` | AQI lags, pollutants, traffic proxies, weather, seasonality, pollution-event history |
| `04_Model_Training` | Persistence baseline vs Ridge / Random Forest / XGBoost, 80% prediction intervals |
| `05_Prophet` | Prophet & ARIMA (rolling-origin), final comparison table |
| `06_SHAP` | Feature importance and explanations |
| `app.py` | Streamlit dashboard + advisory rule engine (`utils.py`) |

## How to run
```bash
pip install -r requirements.txt
# 1. put your OpenAQ export (openaq_*.csv) in data/raw/
# 2. run notebooks 01c, 02, 03, 04, 05, 06 in order (jupyter notebook)
python -m streamlit run app.py
```
Set the city name in `utils.py` (`CITY`). Train/validation/test dates are chosen automatically from the data.

## Method notes
* Chronological split: train <= 2018, validation H1 2019, test after 2019-06-30 (never shuffled).
* Features at day *t* only use data available at the end of day *t* (no leakage).
* Uncertainty: empirical 10th/90th percentile of validation error ratios -> 80% interval; coverage is checked on the test set.
* Advisory uses the *upper* bound of the interval for planning.

## Limitations
* Data covers one monitoring station, not the whole city; the dashboard's latest forecast starts from the last day in the data file.
* Weather inputs are observed values at forecast time, not weather forecasts.
* Not medical advice.
