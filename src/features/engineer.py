"""Feature engineering engine for severe episode prediction and regression."""

import json
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import PROCESSED_DATA_DIR, TABLES_DIR

def engineer_all_features():
    print("=" * 60)
    print("ENGINEERING TEMPORAL, METEOROLOGICAL & STUBBLE FEATURES")
    print("=" * 60)
    
    df = pd.read_csv(PROCESSED_DATA_DIR / "master.csv")
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['city', 'date'], inplace=True)
    
    # -------------------------------------------------------------
    # 1. Target Construction (Day t+1) strictly without leakage
    # -------------------------------------------------------------
    # severe_next_day: 1 if PM2.5(t+1) >= the top-12% city-day PM2.5 value (data-driven,
    # per the project blueprint: "top 10 to 15% of days"). The fixed AQI>=300 / PM2.5>=150
    # rule gave only 1.6% positives in the training period, too few to train or tune on.
    SEVERE_QUANTILE = 0.88
    severe_thr = float(df['PM2.5'].quantile(SEVERE_QUANTILE))
    with open(TABLES_DIR / "severe_threshold.json", "w") as fh:
        json.dump({"quantile": SEVERE_QUANTILE, "pm25_threshold": round(severe_thr, 2)}, fh)
    print(f"[Features] Severe threshold = PM2.5 >= {severe_thr:.1f} ug/m3 (top {100*(1-SEVERE_QUANTILE):.0f}% of city-days)")
    df['pm25_next_day'] = df.groupby('city')['PM2.5'].shift(-1)
    df['aqi_next_day'] = df.groupby('city')['AQI'].shift(-1)
    
    df['severe_next_day'] = np.where(
        df['pm25_next_day'] >= severe_thr,
        1,
        np.where(df['pm25_next_day'].notnull(), 0, np.nan)
    )
    
    # -------------------------------------------------------------
    # 2. Autoregressive Lags (Shifted strictly backwards from day t)
    # -------------------------------------------------------------
    lag_days = [1, 2, 3, 7]
    for lag in lag_days:
        df[f'pm25_lag_{lag}'] = df.groupby('city')['PM2.5'].shift(lag)
        df[f'aqi_lag_{lag}'] = df.groupby('city')['AQI'].shift(lag)
        df[f'no2_lag_{lag}'] = df.groupby('city')['NO2'].shift(lag)
        
    # -------------------------------------------------------------
    # 3. Rolling Statistics (3, 7, 30 days) computed on historical data
    # -------------------------------------------------------------
    windows = [3, 7, 30]
    for w in windows:
        # PM2.5 rolling mean and std
        df[f'pm25_roll{w}_mean'] = df.groupby('city')['PM2.5'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=max(2, w//3)).mean()
        )
        df[f'pm25_roll{w}_std'] = df.groupby('city')['PM2.5'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=max(2, w//3)).std()
        )
        
    # -------------------------------------------------------------
    # 4. Transboundary Stubble Burning Lags (1, 2, 3 days)
    # -------------------------------------------------------------
    for f_lag in [1, 2, 3]:
        df[f'fire_count_NW_lag_{f_lag}'] = df.groupby('city')['fire_count_NW_India'].shift(f_lag)
        df[f'mean_FRP_lag_{f_lag}'] = df.groupby('city')['mean_FRP'].shift(f_lag)
        
    # -------------------------------------------------------------
    # 5. Delta Changes / Momentum Features
    # -------------------------------------------------------------
    # pm25_diff_1: today's PM2.5 minus yesterday's PM2.5
    df['pm25_diff_1'] = df['PM2.5'] - df['pm25_lag_1']
    # pm25_diff_7: today's PM2.5 minus 7 days ago
    df['pm25_diff_7'] = df['PM2.5'] - df['pm25_lag_7']
    
    # Clean infs and physical boundaries
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Save engineered feature panel
    out_features_path = PROCESSED_DATA_DIR / "master_features.csv"
    df.to_csv(out_features_path, index=False)
    print(f"[Features] Saved {len(df):,} rows with {len(df.columns)} features to {out_features_path}")
    print(f"[Features] Severe next day target balance:\n{df['severe_next_day'].value_counts(dropna=False)}")
    
    # Document Feature Metadata
    feature_meta = [
        {"Feature": "pm25_lag_1", "Formula": "PM2.5(t-1)", "Domain": "Autoregressive", "Leakage_Safe": "Yes"},
        {"Feature": "pm25_lag_2", "Formula": "PM2.5(t-2)", "Domain": "Autoregressive", "Leakage_Safe": "Yes"},
        {"Feature": "pm25_lag_3", "Formula": "PM2.5(t-3)", "Domain": "Autoregressive", "Leakage_Safe": "Yes"},
        {"Feature": "pm25_lag_7", "Formula": "PM2.5(t-7)", "Domain": "Weekly Seasonality", "Leakage_Safe": "Yes"},
        {"Feature": "pm25_roll3_mean", "Formula": "Mean(PM2.5[t-3:t-1])", "Domain": "Short-term Baseline", "Leakage_Safe": "Yes"},
        {"Feature": "pm25_roll7_mean", "Formula": "Mean(PM2.5[t-7:t-1])", "Domain": "Weekly Baseline", "Leakage_Safe": "Yes"},
        {"Feature": "pm25_roll30_mean", "Formula": "Mean(PM2.5[t-30:t-1])", "Domain": "Monthly Baseline", "Leakage_Safe": "Yes"},
        {"Feature": "pm25_diff_1", "Formula": "PM2.5(t) - PM2.5(t-1)", "Domain": "Momentum / Delta", "Leakage_Safe": "Yes"},
        {"Feature": "ventilation_index", "Formula": "(WS / 3.6) * BLH", "Domain": "Atmospheric Dispersion", "Leakage_Safe": "Yes"},
        {"Feature": "fire_count_NW_lag_1", "Formula": "Fires_NW(t-1)", "Domain": "Transboundary Biomass Plume", "Leakage_Safe": "Yes"},
        {"Feature": "fire_count_NW_lag_2", "Formula": "Fires_NW(t-2)", "Domain": "Transboundary Biomass Plume", "Leakage_Safe": "Yes"},
        {"Feature": "diwali_window", "Formula": "1 if date in [Diwali-3, Diwali+3]", "Domain": "Festive Anthropogenic Shock", "Leakage_Safe": "Yes"},
        {"Feature": "severe_next_day", "Formula": "1 if PM2.5(t+1) >= top-12% city-day threshold (see severe_threshold.json)", "Domain": "Target Variable", "Leakage_Safe": "Target"}
    ]
    pd.DataFrame(feature_meta).to_csv(TABLES_DIR / "feature_dictionary.csv", index=False)
    print(f"[Features] Saved feature documentation to {TABLES_DIR / 'feature_dictionary.csv'}")
    return df

if __name__ == "__main__":
    engineer_all_features()
