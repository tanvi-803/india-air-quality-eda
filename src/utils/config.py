"""Configuration and path utilities for Air Quality EDA & ML Project."""

import os
import json
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"

CONFIGS_DIR = PROJECT_ROOT / "configs"

# Global Random State
RANDOM_SEED = 42

def load_cities():
    with open(CONFIGS_DIR / "cities.json", "r") as f:
        return json.load(f)

def load_cpcb_breakpoints():
    with open(CONFIGS_DIR / "cpcb_breakpoints.json", "r") as f:
        return json.load(f)
