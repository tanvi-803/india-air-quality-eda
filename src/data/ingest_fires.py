"""Ingest and aggregate NASA FIRMS active fire detections for Punjab & Haryana."""

import os
import sys
import zipfile
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import RAW_DATA_DIR, INTERIM_DATA_DIR

def process_firms_data():
    print("=" * 60)
    print("INGESTING & AGGREGATING NASA FIRMS FIRE DETECTIONS")
    print("=" * 60)
    
    os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
    firms_zip = RAW_DATA_DIR / "NASA_FIRMS_RAW.zip"
    
    # Bounding box for Punjab and Haryana (stubble burning airshed upwind of NCR/IGP)
    LAT_MIN, LAT_MAX = 28.5, 33.0
    LON_MIN, LON_MAX = 73.5, 78.0
    
    daily_fire_counts = defaultdict(int)
    daily_fire_frp = defaultdict(list)
    
    with zipfile.ZipFile(firms_zip, 'r') as zf:
        f_files = [n for n in zf.namelist() if not n.startswith('__MACOSX') and n.endswith('.csv')]
        print(f"[FIRMS Ingest] Processing {len(f_files)} CSV files...")
        
        for ff in f_files:
            with zf.open(ff) as f:
                df = pd.read_csv(f)
                if df.empty or 'acq_date' not in df.columns or 'latitude' not in df.columns:
                    continue
                    
                # Spatial bounding box filter
                mask = (
                    (df['latitude'] >= LAT_MIN) & (df['latitude'] <= LAT_MAX) &
                    (df['longitude'] >= LON_MIN) & (df['longitude'] <= LON_MAX)
                )
                df_nw = df[mask]
                
                for _, row in df_nw.iterrows():
                    d = str(row['acq_date'])
                    daily_fire_counts[d] += 1
                    frp = row.get('frp', np.nan)
                    if pd.notna(frp):
                        daily_fire_frp[d].append(float(frp))
                        
    # Construct complete date range across 2023-01-01 to 2025-12-31
    full_dates = pd.date_range(start="2023-01-01", end="2025-12-31", freq='D').strftime('%Y-%m-%d')
    records = []
    
    for d in full_dates:
        cnt = daily_fire_counts.get(d, 0)
        frps = daily_fire_frp.get(d, [])
        mean_frp = float(np.mean(frps)) if frps else 0.0
        total_frp = float(np.sum(frps)) if frps else 0.0
        records.append({
            "date": d,
            "fire_count_NW_India": cnt,
            "mean_FRP": round(mean_frp, 2),
            "total_FRP": round(total_frp, 2)
        })
        
    df_fires = pd.DataFrame(records)
    df_fires.sort_values(by="date", inplace=True)
    df_fires.reset_index(drop=True, inplace=True)
    
    out_path = INTERIM_DATA_DIR / "fires_daily.csv"
    df_fires.to_csv(out_path, index=False)
    print(f"[FIRMS Ingest] Saved {len(df_fires):,} daily records to {out_path}")
    print(f"[FIRMS Ingest] Peak fire days in NW India:\n{df_fires.sort_values(by='fire_count_NW_India', ascending=False).head(5)}")
    return df_fires

if __name__ == "__main__":
    process_firms_data()
