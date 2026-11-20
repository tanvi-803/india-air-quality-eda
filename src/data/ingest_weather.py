"""Ingest and aggregate Open-Meteo historical weather data."""

import os
import sys
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import RAW_DATA_DIR, INTERIM_DATA_DIR, load_cities

def process_weather_data():
    print("=" * 60)
    print("INGESTING & AGGREGATING OPEN-METEO WEATHER DATA")
    print("=" * 60)
    
    os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
    weather_zip = RAW_DATA_DIR / "Open-Meto_data.zip"
    
    all_weather_records = []
    
    with zipfile.ZipFile(weather_zip, 'r') as zf:
        w_files = [n for n in zf.namelist() if not n.startswith('__MACOSX') and n.endswith('.csv')]
        
        for wf in sorted(w_files):
            city = Path(wf).stem
            with zf.open(wf) as f:
                df = pd.read_csv(f, skiprows=3)
                df['datetime'] = pd.to_datetime(df['time'])
                df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
                
                # Interpolate boundary layer height using linear/spline before aggregation
                if 'boundary_layer_height (m)' in df.columns:
                    df['boundary_layer_height (m)'] = df['boundary_layer_height (m)'].interpolate(method='linear', limit_direction='both')
                
                # Hourly wind components u (eastward) and v (northward)
                # Meteorological wind direction: angle wind is coming FROM, clockwise from North
                rad = np.deg2rad(df['wind_direction_10m (°)'])
                # u: zonal component (-speed * sin(deg)), v: meridional component (-speed * cos(deg))
                df['u_wind'] = -df['wind_speed_10m (km/h)'] * np.sin(rad)
                df['v_wind'] = -df['wind_speed_10m (km/h)'] * np.cos(rad)
                
                # Daily grouping
                daily = df.groupby('date').agg({
                    'temperature_2m (°C)': ['mean', 'max', 'min'],
                    'relative_humidity_2m (%)': 'mean',
                    'pressure_msl (hPa)': 'mean',
                    'precipitation (mm)': 'sum',
                    'wind_speed_10m (km/h)': 'mean',
                    'wind_direction_10m (°)': 'mean',
                    'boundary_layer_height (m)': 'mean',
                    'u_wind': 'mean',
                    'v_wind': 'mean'
                })
                
                # Flatten multi-level columns
                daily.columns = [
                    'temp_mean', 'temp_max', 'temp_min',
                    'humidity_mean',
                    'pressure_mean',
                    'precip_sum',
                    'wind_speed_mean',
                    'wind_dir_mean',
                    'boundary_layer_height_mean',
                    'u_wind_mean',
                    'v_wind_mean'
                ]
                daily.reset_index(inplace=True)
                daily['city'] = city
                daily['temp_range'] = daily['temp_max'] - daily['temp_min']
                daily['ventilation_index'] = (daily['wind_speed_mean'] / 3.6) * daily['boundary_layer_height_mean']
                
                all_weather_records.append(daily)
                
    df_all_weather = pd.concat(all_weather_records, ignore_index=True)
    df_all_weather.sort_values(by=['city', 'date'], inplace=True)
    df_all_weather.reset_index(drop=True, inplace=True)
    
    out_path = INTERIM_DATA_DIR / "weather_daily.csv"
    df_all_weather.to_csv(out_path, index=False)
    print(f"[Weather Ingest] Saved {len(df_all_weather):,} daily records to {out_path}")
    print(f"[Weather Ingest] Summary of variables:\n{df_all_weather[['temp_mean', 'humidity_mean', 'wind_speed_mean', 'boundary_layer_height_mean', 'ventilation_index']].describe().round(2)}")
    return df_all_weather

if __name__ == "__main__":
    process_weather_data()
