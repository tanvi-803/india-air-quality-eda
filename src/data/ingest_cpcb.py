"""Ingest and aggregate CPCB CAAQMS air quality data for target cities."""

import os
import sys
import zipfile
import csv
import io
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import RAW_DATA_DIR, INTERIM_DATA_DIR, load_cities
from src.preprocessing.clean import (
    clean_pollutant_series,
    calculate_cpcb_subindex,
    compute_cpcb_aqi_row
)

POLLUTANTS_MAP = {
    "PM2.5 (µg/m³)": "PM2.5",
    "PM10 (µg/m³)": "PM10",
    "NO2 (µg/m³)": "NO2",
    "SO2 (µg/m³)": "SO2",
    "CO (mg/m³)": "CO",
    "Ozone (µg/m³)": "O3",
    "NH3 (µg/m³)": "NH3"
}

def process_cpcb_data():
    print("=" * 60)
    print("INGESTING & AGGREGATING CPCB AIR QUALITY DATA")
    print("=" * 60)
    
    os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
    target_cities = set(load_cities().keys())
    
    cpcb_zips = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.startswith("cpcb-air-quality") and f.endswith(".zip")])
    if not cpcb_zips:
        print("[Error] No CPCB archives found in data/raw/")
        return None
        
    # Data structure to accumulate sums and counts per (station_id, city, date)
    # key: (station_id, city, date) -> pollutant -> {"sum": float, "count": int}
    station_day_data = defaultdict(lambda: {
        p: {"sum": 0.0, "count": 0} for p in POLLUTANTS_MAP.values()
    })
    station_day_total_intervals = defaultdict(int)
    
    for cpcb_zip in cpcb_zips:
        zip_path = RAW_DATA_DIR / cpcb_zip
        print(f"[CPCB Ingest] Parsing {cpcb_zip}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            inner_name = [n for n in zf.namelist() if n.endswith('.csv')][0]
            with zf.open(inner_name) as f:
                reader = csv.reader(io.TextIOWrapper(f, encoding='utf-8', errors='replace'))
                header = next(reader)
                
                st_id_idx = header.index("Station ID")
                city_idx = header.index("City")
                ts_idx = header.index("Timestamp")
                p_col_indices = {raw_col: header.index(raw_col) for raw_col in POLLUTANTS_MAP if raw_col in header}
                
                rows_processed = 0
                for row in reader:
                    rows_processed += 1
                    if len(row) > max(st_id_idx, city_idx, ts_idx):
                        city = row[city_idx].strip()
                        if city in target_cities:
                            st_id = row[st_id_idx].strip()
                            date_str = row[ts_idx][:10]
                            st_key = (st_id, city, date_str)
                            station_day_total_intervals[st_key] += 1
                            
                            for raw_col, short_p in POLLUTANTS_MAP.items():
                                if raw_col in p_col_indices:
                                    clean_val = clean_pollutant_series(row[p_col_indices[raw_col]], raw_col)
                                    if clean_val is not None and not np.isnan(clean_val):
                                        station_day_data[st_key][short_p]["sum"] += clean_val
                                        station_day_data[st_key][short_p]["count"] += 1
                                        
                    if rows_processed % 4000000 == 0:
                        print(f"  Processed {rows_processed:,} lines...")
                        
    print(f"[CPCB Ingest] Completed raw stream parsing. Total station-days accumulated: {len(station_day_data):,}")
    
    # Step 1: Compute Station-Daily Means (enforcing >= 64 intervals = 16 hours for validity)
    # Then aggregate to City-Day
    # city_day_accum: (city, date) -> pollutant -> list of station means
    city_day_accum = defaultdict(lambda: {p: [] for p in POLLUTANTS_MAP.values()})
    city_day_valid_hours = defaultdict(list)
    city_day_stations = defaultdict(set)
    city_day_max_pm25 = defaultdict(lambda: -1.0)
    
    for (st_id, city, date_str), p_dict in station_day_data.items():
        intervals = station_day_total_intervals[(st_id, city, date_str)]
        valid_hrs = min(24.0, intervals / 4.0)
        city_day_stations[(city, date_str)].add(st_id)
        city_day_valid_hours[(city, date_str)].append(valid_hrs)
        
        for short_p, accum in p_dict.items():
            # A station reading is valid if it has at least 64 15-minute readings (16 hours)
            if accum["count"] >= 64:
                daily_st_mean = accum["sum"] / accum["count"]
                city_day_accum[(city, date_str)][short_p].append(daily_st_mean)
                if short_p == "PM2.5" and daily_st_mean > city_day_max_pm25[(city, date_str)]:
                    city_day_max_pm25[(city, date_str)] = daily_st_mean

    # Step 2: Build City-Day Records with CPCB AQI
    city_day_records = []
    for (city, date_str), p_means in city_day_accum.items():
        rec = {
            "city": city,
            "date": date_str,
            "active_stations": len(city_day_stations[(city, date_str)]),
            "valid_hours": round(float(np.mean(city_day_valid_hours[(city, date_str)])), 1)
        }
        
        # Sub-indices container
        subs = {}
        for short_p in POLLUTANTS_MAP.values():
            vals = p_means[short_p]
            if len(vals) > 0:
                city_mean = float(np.mean(vals))
                rec[short_p] = round(city_mean, 2)
                subs[short_p] = calculate_cpcb_subindex(city_mean, short_p)
            else:
                rec[short_p] = np.nan
                subs[short_p] = np.nan
                
        rec["PM2.5_max"] = round(city_day_max_pm25[(city, date_str)], 2) if city_day_max_pm25[(city, date_str)] >= 0 else np.nan
        
        # Compute official CPCB NAQI
        aqi_val, aqi_cat, dom_pollutant = compute_cpcb_aqi_row(subs, rec["valid_hours"])
        rec["AQI"] = aqi_val
        rec["AQI_category"] = aqi_cat
        rec["dominant_pollutant"] = dom_pollutant
        
        city_day_records.append(rec)
        
    df_aq = pd.DataFrame(city_day_records)
    df_aq.sort_values(by=["city", "date"], inplace=True)
    df_aq.reset_index(drop=True, inplace=True)
    
    out_path = INTERIM_DATA_DIR / "aq_daily.csv"
    df_aq.to_csv(out_path, index=False)
    print(f"[CPCB Ingest] Saved {len(df_aq):,} city-day records to {out_path}")
    print(f"[CPCB Ingest] Non-null AQI computed for: {df_aq['AQI'].notnull().sum()} / {len(df_aq)} city-days")
    print(f"[CPCB Ingest] AQI Categories breakdown:\n{df_aq['AQI_category'].value_counts()}")
    return df_aq

if __name__ == "__main__":
    process_cpcb_data()
