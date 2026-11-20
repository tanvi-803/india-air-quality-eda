"""Consolidated master dataset builder and Excel workbook generator."""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    TABLES_DIR,
    load_cities
)

def build_master_dataset():
    print("=" * 60)
    print("BUILDING UNIFIED MASTER PANEL & EXCEL WORKBOOK")
    print("=" * 60)
    
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    
    # Load all processed interim components
    cities_dict = load_cities()
    df_aq = pd.read_csv(INTERIM_DATA_DIR / "aq_daily_imputed.csv")
    df_weather = pd.read_csv(INTERIM_DATA_DIR / "weather_daily.csv")
    df_fires = pd.read_csv(INTERIM_DATA_DIR / "fires_daily.csv")
    df_cal = pd.read_csv(INTERIM_DATA_DIR / "calendar.csv")
    
    # -------------------------------------------------------------
    # 1. Build 'cities' sheet
    # -------------------------------------------------------------
    city_rows = []
    st_cov = pd.read_csv(TABLES_DIR / "cpcb_city_station_coverage.csv").set_index("City")
    for city, info in cities_dict.items():
        st_cnt = int(st_cov.loc[city, "Stations_Count"]) if city in st_cov.index else 1
        city_rows.append({
            "city": city,
            "state": info["state"],
            "region": info["region"],
            "latitude": info["latitude"],
            "longitude": info["longitude"],
            "elevation_m": info["elevation"],
            "stations_count": st_cnt
        })
    df_cities = pd.DataFrame(city_rows)
    
    # -------------------------------------------------------------
    # 2. Build 'master' merged panel
    # -------------------------------------------------------------
    print("[Master Build] Merging datasets on (city, date)...")
    # Base merge: Air Quality (left) with Weather, Fires, Calendar
    df_master = pd.merge(df_aq, df_weather, on=["city", "date"], how="inner")
    df_master = pd.merge(df_master, df_fires, on="date", how="left")
    df_master = pd.merge(df_master, df_cal, on="date", how="left")
    
    # Add geographical attributes
    df_master = pd.merge(df_master, df_cities[["city", "state", "region", "latitude", "longitude"]], on="city", how="left")
    
    # Calculate particulate ratios
    df_master["pm25_to_pm10_ratio"] = np.where(
        (df_master["PM10"] > 0) & (df_master["PM2.5"] > 0),
        np.clip(df_master["PM2.5"] / df_master["PM10"], 0.05, 1.0),
        np.nan
    )
    
    # Sort chronologically by city and date
    df_master.sort_values(by=["city", "date"], inplace=True)
    df_master.reset_index(drop=True, inplace=True)
    
    # Save processed master CSV
    master_csv_path = PROCESSED_DATA_DIR / "master.csv"
    df_master.to_csv(master_csv_path, index=False)
    print(f"[Master Build] Saved processed master panel to {master_csv_path} ({len(df_master):,} rows, {len(df_master.columns)} cols)")

    # -------------------------------------------------------------
    # 3. Build 'metadata' sheet
    # -------------------------------------------------------------
    metadata_rows = [
        # cities sheet
        {"sheet": "cities", "column_name": "city", "meaning": "City name", "unit": "text", "source": "CPCB/Open-Meteo", "missing_handling": "None"},
        {"sheet": "cities", "column_name": "state", "meaning": "State or Union Territory", "unit": "text", "source": "Official Census", "missing_handling": "None"},
        {"sheet": "cities", "column_name": "region", "meaning": "Geographical region (North, West, South, East)", "unit": "text", "source": "Geographic classification", "missing_handling": "None"},
        {"sheet": "cities", "column_name": "latitude", "meaning": "City centroid latitude", "unit": "degrees North", "source": "Open-Meteo", "missing_handling": "None"},
        {"sheet": "cities", "column_name": "longitude", "meaning": "City centroid longitude", "unit": "degrees East", "source": "Open-Meteo", "missing_handling": "None"},
        {"sheet": "cities", "column_name": "elevation_m", "meaning": "Elevation above sea level", "unit": "meters", "source": "Open-Meteo", "missing_handling": "None"},
        {"sheet": "cities", "column_name": "stations_count", "meaning": "Count of CAAQMS regulatory monitoring stations", "unit": "count", "source": "CPCB", "missing_handling": "None"},
        
        # aq_daily sheet
        {"sheet": "aq_daily", "column_name": "city", "meaning": "City identifier", "unit": "text", "source": "CPCB", "missing_handling": "None"},
        {"sheet": "aq_daily", "column_name": "date", "meaning": "Observation date (ISO 8601)", "unit": "YYYY-MM-DD", "source": "CPCB", "missing_handling": "None"},
        {"sheet": "aq_daily", "column_name": "PM2.5", "meaning": "Fine particulate matter <= 2.5 µm daily average", "unit": "µg/m³", "source": "CPCB CAAQMS", "missing_handling": "Linear interpolation capped at 7 days"},
        {"sheet": "aq_daily", "column_name": "PM10", "meaning": "Particulate matter <= 10 µm daily average", "unit": "µg/m³", "source": "CPCB CAAQMS", "missing_handling": "Linear interpolation capped at 7 days"},
        {"sheet": "aq_daily", "column_name": "NO2", "meaning": "Nitrogen Dioxide daily average", "unit": "µg/m³", "source": "CPCB CAAQMS", "missing_handling": "Linear interpolation capped at 7 days"},
        {"sheet": "aq_daily", "column_name": "SO2", "meaning": "Sulphur Dioxide daily average", "unit": "µg/m³", "source": "CPCB CAAQMS", "missing_handling": "Linear interpolation capped at 7 days"},
        {"sheet": "aq_daily", "column_name": "CO", "meaning": "Carbon Monoxide daily average", "unit": "mg/m³", "source": "CPCB CAAQMS", "missing_handling": "Linear interpolation capped at 7 days"},
        {"sheet": "aq_daily", "column_name": "O3", "meaning": "Ozone daily average", "unit": "µg/m³", "source": "CPCB CAAQMS", "missing_handling": "Linear interpolation capped at 7 days"},
        {"sheet": "aq_daily", "column_name": "NH3", "meaning": "Ammonia daily average", "unit": "µg/m³", "source": "CPCB CAAQMS", "missing_handling": "Linear interpolation capped at 7 days"},
        {"sheet": "aq_daily", "column_name": "valid_hours", "meaning": "Average valid monitoring hours across stations", "unit": "hours (0-24)", "source": "CPCB CAAQMS", "missing_handling": "Computed from 15-min intervals"},
        {"sheet": "aq_daily", "column_name": "AQI", "meaning": "Official CPCB National Air Quality Index (NAQI)", "unit": "index (0-500)", "source": "CPCB formula engine", "missing_handling": "Computed where >=3 pollutants & >=16h valid"},
        {"sheet": "aq_daily", "column_name": "AQI_category", "meaning": "Qualitative AQI band (Good to Severe)", "unit": "category", "source": "CPCB NAQI Standard", "missing_handling": "None"},
        {"sheet": "aq_daily", "column_name": "dominant_pollutant", "meaning": "Pollutant dictating the maximum sub-index", "unit": "pollutant name", "source": "CPCB NAQI Standard", "missing_handling": "None"},

        # weather_daily sheet
        {"sheet": "weather_daily", "column_name": "temp_mean", "meaning": "Mean 2m surface temperature", "unit": "°C", "source": "Open-Meteo", "missing_handling": "Reanalysis complete"},
        {"sheet": "weather_daily", "column_name": "temp_max", "meaning": "Maximum daily 2m temperature", "unit": "°C", "source": "Open-Meteo", "missing_handling": "Reanalysis complete"},
        {"sheet": "weather_daily", "column_name": "temp_min", "meaning": "Minimum daily 2m temperature", "unit": "°C", "source": "Open-Meteo", "missing_handling": "Reanalysis complete"},
        {"sheet": "weather_daily", "column_name": "temp_range", "meaning": "Diurnal temperature range (max - min)", "unit": "°C", "source": "Engineered", "missing_handling": "None"},
        {"sheet": "weather_daily", "column_name": "humidity_mean", "meaning": "Mean relative humidity", "unit": "%", "source": "Open-Meteo", "missing_handling": "Reanalysis complete"},
        {"sheet": "weather_daily", "column_name": "wind_speed_mean", "meaning": "Mean 10m wind speed", "unit": "km/h", "source": "Open-Meteo", "missing_handling": "Reanalysis complete"},
        {"sheet": "weather_daily", "column_name": "wind_dir_mean", "meaning": "Mean 10m wind direction", "unit": "degrees", "source": "Open-Meteo", "missing_handling": "Reanalysis complete"},
        {"sheet": "weather_daily", "column_name": "u_wind_mean", "meaning": "Zonal wind component (Eastward)", "unit": "km/h", "source": "Engineered", "missing_handling": "None"},
        {"sheet": "weather_daily", "column_name": "v_wind_mean", "meaning": "Meridional wind component (Northward)", "unit": "km/h", "source": "Engineered", "missing_handling": "None"},
        {"sheet": "weather_daily", "column_name": "precip_sum", "meaning": "Total daily precipitation", "unit": "mm", "source": "Open-Meteo", "missing_handling": "Reanalysis complete"},
        {"sheet": "weather_daily", "column_name": "pressure_mean", "meaning": "Mean sea level pressure", "unit": "hPa", "source": "Open-Meteo", "missing_handling": "Reanalysis complete"},
        {"sheet": "weather_daily", "column_name": "boundary_layer_height_mean", "meaning": "Atmospheric boundary layer height", "unit": "meters", "source": "Open-Meteo", "missing_handling": "Spline interpolated nocturnal gaps"},
        {"sheet": "weather_daily", "column_name": "ventilation_index", "meaning": "Ventilation index proxy (WS * BLH)", "unit": "m²/s", "source": "Engineered", "missing_handling": "None"},

        # fires_daily sheet
        {"sheet": "fires_daily", "column_name": "fire_count_NW_India", "meaning": "Active fire pixels in Punjab & Haryana airshed", "unit": "count", "source": "NASA FIRMS VIIRS/MODIS", "missing_handling": "Zero filled for clear days"},
        {"sheet": "fires_daily", "column_name": "mean_FRP", "meaning": "Mean Fire Radiative Power", "unit": "MW (Megawatts)", "source": "NASA FIRMS", "missing_handling": "Zero filled"},
        {"sheet": "fires_daily", "column_name": "total_FRP", "meaning": "Total Fire Radiative Power sum", "unit": "MW", "source": "NASA FIRMS", "missing_handling": "Zero filled"},

        # calendar sheet
        {"sheet": "calendar", "column_name": "is_weekend", "meaning": "Flag for Saturday or Sunday", "unit": "binary (0/1)", "source": "Calendar", "missing_handling": "Deterministic"},
        {"sheet": "calendar", "column_name": "month", "meaning": "Month of year (1-12)", "unit": "integer", "source": "Calendar", "missing_handling": "Deterministic"},
        {"sheet": "calendar", "column_name": "season", "meaning": "Climatological season", "unit": "text", "source": "Calendar", "missing_handling": "Deterministic"},
        {"sheet": "calendar", "column_name": "diwali_window", "meaning": "Diwali +/- 3 days window flag", "unit": "binary (0/1)", "source": "Cultural Calendar", "missing_handling": "Deterministic"},
        {"sheet": "calendar", "column_name": "holi_window", "meaning": "Holi +/- 3 days window flag", "unit": "binary (0/1)", "source": "Cultural Calendar", "missing_handling": "Deterministic"},
        {"sheet": "calendar", "column_name": "month_sin", "meaning": "Trigonometric sine cyclical month", "unit": "float [-1, 1]", "source": "Engineered", "missing_handling": "None"},
        {"sheet": "calendar", "column_name": "month_cos", "meaning": "Trigonometric cosine cyclical month", "unit": "float [-1, 1]", "source": "Engineered", "missing_handling": "None"},

        # master sheet
        {"sheet": "master", "column_name": "pm25_to_pm10_ratio", "meaning": "PM2.5 to PM10 concentration ratio", "unit": "ratio [0, 1]", "source": "Engineered", "missing_handling": "Capped physically"}
    ]
    df_meta = pd.DataFrame(metadata_rows)
    
    # -------------------------------------------------------------
    # 4. Generate Consolidated Multi-Sheet Excel Workbook
    # -------------------------------------------------------------
    excel_path = TABLES_DIR / "air_quality_master.xlsx"
    print(f"[Master Build] Compiling consolidated Excel workbook at {excel_path}...")
    
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_cities.to_excel(writer, sheet_name="cities", index=False)
        # Raw/imputed daily AQ table
        df_aq.to_excel(writer, sheet_name="aq_daily", index=False)
        # Weather daily table (sample/2025 to keep excel responsive)
        df_weather.to_excel(writer, sheet_name="weather_daily", index=False)
        # Fires daily table
        df_fires.to_excel(writer, sheet_name="fires_daily", index=False)
        # Calendar table
        df_cal.to_excel(writer, sheet_name="calendar", index=False)
        # Unified Master Table
        df_master.to_excel(writer, sheet_name="master", index=False)
        # Metadata dictionary
        df_meta.to_excel(writer, sheet_name="metadata", index=False)
        
    print(f"[Master Build] Successfully created multi-sheet Excel workbook: {excel_path} ({os.path.getsize(excel_path) / (1024*1024):.2f} MB)")
    return df_master

if __name__ == "__main__":
    build_master_dataset()
