"""Ingest and enrich national festival and cultural calendar."""

import os
import sys
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import RAW_DATA_DIR, INTERIM_DATA_DIR

def process_calendar_data():
    print("=" * 60)
    print("INGESTING & ENRICHING FESTIVAL CALENDAR DATA")
    print("=" * 60)
    
    os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
    fest_zip = RAW_DATA_DIR / "festival_calendar_raw.zip"
    
    with zipfile.ZipFile(fest_zip, 'r') as zf:
        inner = [n for n in zf.namelist() if n.endswith('.csv')][0]
        with zf.open(inner) as f:
            df = pd.read_csv(f)
            
    df['datetime'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_year'] = df['datetime'].dt.dayofyear
    
    # Cyclical trigonometric month encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    # Season indicators
    df['is_winter'] = (df['season'].str.lower() == 'winter').astype(int)
    df['is_monsoon'] = (df['season'].str.lower() == 'monsoon').astype(int)
    df['is_summer'] = (df['season'].str.lower() == 'summer').astype(int)
    df['is_post_monsoon'] = (df['season'].str.lower().isin(['post-monsoon', 'autumn'])).astype(int)
    
    # Drop intermediate datetime column
    df.drop(columns=['datetime'], inplace=True)
    
    out_path = INTERIM_DATA_DIR / "calendar.csv"
    df.to_csv(out_path, index=False)
    print(f"[Calendar Ingest] Saved {len(df):,} calendar days to {out_path}")
    return df

if __name__ == "__main__":
    process_calendar_data()
