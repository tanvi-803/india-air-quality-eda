"""Unit tests for dataset schema, shapes, and referential integrity."""

import pytest
import pandas as pd
from pathlib import Path
from src.utils.config import PROCESSED_DATA_DIR, TABLES_DIR, load_cities

def test_cities_completeness():
    cities_dict = load_cities()
    df_cities = pd.read_csv(PROCESSED_DATA_DIR / "master.csv")
    unique_cities = set(df_cities['city'].unique())
    assert unique_cities == set(cities_dict.keys())

def test_master_shape_and_no_duplicate_keys():
    df_master = pd.read_csv(PROCESSED_DATA_DIR / "master.csv")
    # Verify no duplicate (city, date) rows exist
    dups = df_master.duplicated(subset=['city', 'date']).sum()
    assert dups == 0, f"Found {dups} duplicate city-date rows!"

def test_excel_workbook_sheets():
    import openpyxl
    wb = openpyxl.load_workbook(TABLES_DIR / "air_quality_master.xlsx", read_only=True)
    required_sheets = {"cities", "aq_daily", "weather_daily", "fires_daily", "calendar", "master", "metadata"}
    assert required_sheets.issubset(set(wb.sheetnames))
    wb.close()
