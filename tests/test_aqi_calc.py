"""Unit tests for CPCB NAQI calculation logic and validity rules."""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing.clean import (
    clean_pollutant_series,
    calculate_cpcb_subindex,
    get_aqi_category,
    compute_cpcb_aqi_row
)

def test_clean_pollutant_series():
    # Valid concentration
    assert clean_pollutant_series(55.4, "PM2.5 (µg/m³)") == 55.4
    # Negative should become NaN
    assert np.isnan(clean_pollutant_series(-5.0, "PM2.5 (µg/m³)"))
    # Sentinel 9999 should become NaN
    assert np.isnan(clean_pollutant_series(9999.0, "PM2.5 (µg/m³)"))
    # Extreme impossible outlier > 1500
    assert np.isnan(clean_pollutant_series(2500.0, "PM2.5 (µg/m³)"))

def test_calculate_cpcb_subindex():
    # PM2.5 in Good range (0-30 -> 0-50)
    assert calculate_cpcb_subindex(15.0, "PM2.5") == 25.0
    # PM2.5 at exact breakpoint 30 -> 50
    assert calculate_cpcb_subindex(30.0, "PM2.5") == 50.0
    # PM2.5 in Moderate range (61-90 -> 101-200)
    # midpoint 75 -> 101 + ((200-101)/(90-61))*(75-61)
    sub = calculate_cpcb_subindex(75.5, "PM2.5")
    assert 101.0 <= sub <= 200.0

def test_get_aqi_category():
    assert get_aqi_category(45) == "Good"
    assert get_aqi_category(85) == "Satisfactory"
    assert get_aqi_category(150) == "Moderate"
    assert get_aqi_category(250) == "Poor"
    assert get_aqi_category(350) == "Very Poor"
    assert get_aqi_category(450) == "Severe"

def test_compute_cpcb_aqi_validity_rules():
    # Rule 1: < 16 valid hours -> Insufficient Hours
    sub_dict = {"PM2.5": 80.0, "PM10": 120.0, "NO2": 45.0}
    val, cat, dom = compute_cpcb_aqi_row(sub_dict, valid_hours=14.0)
    assert np.isnan(val)
    assert "Insufficient Hours" in cat

    # Rule 2: < 3 pollutants -> Insufficient Pollutants
    sub_dict_few = {"PM2.5": 80.0, "NO2": 45.0}
    val, cat, dom = compute_cpcb_aqi_row(sub_dict_few, valid_hours=24.0)
    assert np.isnan(val)
    assert "Insufficient Pollutants" in cat

    # Rule 3: >= 3 pollutants but NO PM -> No PM Indicator
    sub_dict_nopm = {"NO2": 45.0, "SO2": 20.0, "CO": 35.0}
    val, cat, dom = compute_cpcb_aqi_row(sub_dict_nopm, valid_hours=24.0)
    assert np.isnan(val)
    assert "No PM Indicator" in cat

    # Valid row
    sub_dict_valid = {"PM2.5": 180.0, "PM10": 120.0, "NO2": 45.0}
    val, cat, dom = compute_cpcb_aqi_row(sub_dict_valid, valid_hours=24.0)
    assert val == 180.0
    assert dom == "PM2.5"
    assert cat == "Moderate"
