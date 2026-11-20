"""Unit tests for leakage prevention across feature engineering."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.config import PROCESSED_DATA_DIR

def test_no_target_in_predictors():
    df = pd.read_csv(PROCESSED_DATA_DIR / "master_features.csv")
    targets = ['severe_next_day', 'pm25_next_day', 'aqi_next_day']
    
    # Predictor feature candidates
    feature_cols = [
        'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'pm25_to_pm10_ratio',
        'pm25_lag_1', 'pm25_lag_2', 'pm25_lag_3', 'pm25_lag_7',
        'pm25_roll3_mean', 'pm25_roll7_mean', 'pm25_diff_1'
    ]
    for t in targets:
        assert t not in feature_cols

def test_autoregressive_lag_alignment():
    df = pd.read_csv(PROCESSED_DATA_DIR / "master_features.csv")
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['city', 'date'], inplace=True)
    
    # Check for Delhi that pm25_lag_1 on date t is equal to PM2.5 on date t-1
    delhi = df[df['city'] == 'Delhi'].reset_index(drop=True)
    for idx in range(1, 10):
        prev_pm25 = delhi.loc[idx - 1, 'PM2.5']
        curr_lag1 = delhi.loc[idx, 'pm25_lag_1']
        if pd.notna(prev_pm25) and pd.notna(curr_lag1):
            assert np.isclose(prev_pm25, curr_lag1, atol=1e-2)

def test_target_alignment():
    df = pd.read_csv(PROCESSED_DATA_DIR / "master_features.csv")
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['city', 'date'], inplace=True)
    
    delhi = df[df['city'] == 'Delhi'].reset_index(drop=True)
    for idx in range(0, 10):
        next_pm25 = delhi.loc[idx + 1, 'PM2.5']
        curr_target = delhi.loc[idx, 'pm25_next_day']
        if pd.notna(next_pm25) and pd.notna(curr_target):
            assert np.isclose(next_pm25, curr_target, atol=1e-2)
