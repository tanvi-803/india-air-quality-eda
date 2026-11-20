"""Data cleaning and CPCB NAQI calculation engine."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.config import load_cpcb_breakpoints

CPCB_BREAKPOINTS = load_cpcb_breakpoints()

def clean_pollutant_series(val, pollutant_name):
    """Sanitize individual pollutant value: remove negative values, sentinels and impossible spikes."""
    if val is None or pd.isna(val):
        return np.nan
    try:
        fval = float(val)
    except (ValueError, TypeError):
        return np.nan
    
    # Reject negatives and common sentinel values
    if fval < 0 or fval in (999.0, 9999.0, -999.0):
        return np.nan
    
    # Physical plausibility ceilings
    ceilings = {
        "PM2.5 (µg/m³)": 1500.0,
        "PM10 (µg/m³)": 2500.0,
        "NO2 (µg/m³)": 1000.0,
        "SO2 (µg/m³)": 2500.0,
        "CO (mg/m³)": 100.0,
        "Ozone (µg/m³)": 1000.0,
        "NH3 (µg/m³)": 2500.0
    }
    limit = ceilings.get(pollutant_name, 5000.0)
    if fval > limit:
        return np.nan
    return fval

def calculate_cpcb_subindex(conc, pollutant_key):
    """Calculate the sub-index for a single pollutant based on CPCB breakpoint table.
    Formula: I = I_lo + ((I_hi - I_lo) / (B_hi - B_lo)) * (C - B_lo)
    """
    if conc is None or pd.isna(conc) or conc < 0:
        return np.nan
    
    ranges = CPCB_BREAKPOINTS.get(pollutant_key, [])
    for bp in ranges:
        if bp["b_lo"] <= conc <= bp["b_hi"]:
            b_diff = bp["b_hi"] - bp["b_lo"]
            if b_diff == 0:
                return bp["i_lo"]
            sub_index = bp["i_lo"] + ((bp["i_hi"] - bp["i_lo"]) / b_diff) * (conc - bp["b_lo"])
            return float(sub_index)
            
    # If exceeding max breakpoint, cap at 500
    if ranges and conc > ranges[-1]["b_hi"]:
        return 500.0
    return np.nan

def get_aqi_category(aqi_val):
    """Return CPCB AQI qualitative category from numeric AQI."""
    if aqi_val is None or pd.isna(aqi_val):
        return "Insufficient Data"
    aqi_round = round(aqi_val)
    for cat in CPCB_BREAKPOINTS.get("categories", []):
        if cat["lo"] <= aqi_round <= cat["hi"]:
            return cat["category"]
    if aqi_round > 500:
        return "Severe"
    return "Unknown"

def compute_cpcb_aqi_row(subindices_dict, valid_hours):
    """Enforces official CPCB NAQI calculation conditions:
    1. At least 3 pollutants with valid sub-indices.
    2. At least one must be a particulate matter (PM2.5 or PM10).
    3. At least 16 valid hours in the day.
    Returns: (aqi_value, aqi_category, dominant_pollutant)
    """
    if valid_hours < 16.0:
        return (np.nan, "Insufficient Hours (<16h)", np.nan)
        
    valid_subs = {k: v for k, v in subindices_dict.items() if not pd.isna(v)}
    
    # Must have >= 3 pollutants
    if len(valid_subs) < 3:
        return (np.nan, "Insufficient Pollutants (<3)", np.nan)
        
    # At least one PM required
    if "PM2.5" not in valid_subs and "PM10" not in valid_subs:
        return (np.nan, "No PM Indicator Present", np.nan)
        
    dominant_pollutant = max(valid_subs, key=valid_subs.get)
    max_aqi = valid_subs[dominant_pollutant]
    cat = get_aqi_category(max_aqi)
    return (round(max_aqi, 1), cat, dominant_pollutant)
