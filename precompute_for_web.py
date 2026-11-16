import os
import pickle
import pandas as pd
from src.data_pipeline import load_and_clean_data, engineer_features, create_multi_horizon_targets
from src.models import train_and_evaluate_horizon
from src.explainability import compute_feature_importance

DATA_PATH = os.path.join("data", "raw", "cpcb_openaq_hourly_2024_2025.csv")
OUT_DIR = os.path.join("models", "web_cache")
os.makedirs(OUT_DIR, exist_ok=True)

for city in ["Delhi", "Chandigarh"]:
    print(f"Processing {city}...")
    clean_df = load_and_clean_data(DATA_PATH, city=city)
    feat_df = engineer_features(clean_df)
    model_df = create_multi_horizon_targets(feat_df)

    # Keep only the tail rows needed by the dashboard to keep files tiny & fast
    model_df_tail = model_df.tail(700).copy()
    model_df_tail.to_parquet(os.path.join(OUT_DIR, f"{city}_model_df.parquet"))

    for horizon in [24, 48, 72]:
        print(f"  Training {city} +{horizon}h...")
        metrics_df, preds_df, unc_stats, artifacts = train_and_evaluate_horizon(
            model_df, horizon=horizon, test_days=60
        )
        imp_df = compute_feature_importance(
            artifacts["rf_model"],
            model_df[artifacts["feature_cols"]].tail(500),
            artifacts["feature_cols"],
        )

        # Save only the Random Forest model needed for local SHAP/feature explanation
        light_artifacts = {
            "rf_model": artifacts["rf_model"],
            "feature_cols": artifacts["feature_cols"],
        }

        bundle = {
            "metrics_df": metrics_df,
            "preds_df": preds_df,
            "unc_stats": unc_stats,
            "artifacts": light_artifacts,
            "imp_df": imp_df,
        }
        with open(os.path.join(OUT_DIR, f"{city}_{horizon}h.pkl"), "wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Done! All web cache files saved in models/web_cache/")