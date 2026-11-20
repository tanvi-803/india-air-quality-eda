"""Feature selection engine: Multicollinearity (VIF), Mutual Information & Lasso."""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import PROCESSED_DATA_DIR, TABLES_DIR, RANDOM_SEED

def run_feature_selection():
    print("=" * 60)
    print("RUNNING STATISTICAL & ML FEATURE SELECTION")
    print("=" * 60)
    
    os.makedirs(TABLES_DIR, exist_ok=True)
    df = pd.read_csv(PROCESSED_DATA_DIR / "master_features.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    # Candidate feature list for prediction
    candidate_features = [
        'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'pm25_to_pm10_ratio',
        'pm25_lag_1', 'pm25_lag_2', 'pm25_lag_3', 'pm25_lag_7',
        'pm25_roll3_mean', 'pm25_roll7_mean', 'pm25_diff_1',
        'temp_mean', 'temp_range', 'humidity_mean', 'pressure_mean',
        'wind_speed_mean', 'u_wind_mean', 'v_wind_mean', 'precip_sum',
        'boundary_layer_height_mean', 'ventilation_index',
        'fire_count_NW_India', 'fire_count_NW_lag_1', 'fire_count_NW_lag_2',
        'month_sin', 'month_cos', 'is_weekend', 'diwali_window'
    ]
    
    # Strictly respect chronological training boundary: Jan to Aug 2025
    train_mask = (df['date'] <= '2025-08-31') & df['severe_next_day'].notnull()
    df_train = df[train_mask].dropna(subset=candidate_features + ['pm25_next_day']).copy()
    
    X_train = df_train[candidate_features]
    y_train_reg = df_train['pm25_next_day']
    
    # -------------------------------------------------------------
    # 1. Mutual Information Regression Ranking
    # -------------------------------------------------------------
    print("[Select] Calculating Mutual Information...")
    mi_scores = mutual_info_regression(X_train, y_train_reg, random_state=RANDOM_SEED)
    df_mi = pd.Series(mi_scores, index=candidate_features, name="Mutual_Information")
    
    # -------------------------------------------------------------
    # 2. Lasso L1 Shrinkage Coefficients
    # -------------------------------------------------------------
    print("[Select] Running Cross-Validated Lasso Shrinkage...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    lasso = LassoCV(cv=5, random_state=RANDOM_SEED, max_iter=2000).fit(X_scaled, y_train_reg)
    df_lasso = pd.Series(lasso.coef_, index=candidate_features, name="Lasso_Coefficient")
    
    # -------------------------------------------------------------
    # 3. Multicollinearity / VIF Analysis (on subset to avoid matrix singularity)
    # -------------------------------------------------------------
    print("[Select] Computing Variance Inflation Factors (VIF)...")
    vif_features = [
        'PM2.5', 'NO2', 'SO2', 'CO', 'pm25_lag_1', 'pm25_lag_7',
        'temp_mean', 'humidity_mean', 'wind_speed_mean',
        'boundary_layer_height_mean', 'ventilation_index',
        'fire_count_NW_lag_1', 'diwali_window'
    ]
    X_vif_df = X_train[vif_features].copy()
    X_vif_scaled = StandardScaler().fit_transform(X_vif_df)
    vif_data = [
        variance_inflation_factor(X_vif_scaled, i) for i in range(len(vif_features))
    ]
    df_vif = pd.Series(vif_data, index=vif_features, name="VIF")
    
    # Compile unified selection table
    summary = pd.DataFrame({
        "Mutual_Information": df_mi,
        "Lasso_Coefficient": df_lasso
    })
    summary = summary.join(df_vif, how="left")
    summary['Lasso_Retained'] = np.where(np.abs(summary['Lasso_Coefficient']) > 1e-4, "Retained", "Zeroed-Out")
    summary.sort_values(by="Mutual_Information", ascending=False, inplace=True)
    
    out_table = TABLES_DIR / "feature_selection_summary.csv"
    summary.to_csv(out_table)
    print(f"\n[Select] Feature Selection Summary Saved to {out_table}:\n{summary.head(15).round(3)}")
    return summary

if __name__ == "__main__":
    run_feature_selection()
