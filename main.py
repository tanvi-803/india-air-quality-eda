"""Master Pipeline Orchestrator for Air Quality EDA & ML Project.
Executes all processing, modeling, evaluation, plotting, and dashboard compilation.
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.audit import audit_datasets
from src.data.ingest_cpcb import process_cpcb_data
from src.data.ingest_weather import process_weather_data
from src.data.ingest_fires import process_firms_data
from src.data.ingest_calendar import process_calendar_data
from src.preprocessing.impute import run_imputation_experiment
from src.data.build_master import build_master_dataset
from src.visualization.plots import generate_all_eda_plots
from src.features.engineer import engineer_all_features
from src.features.feature_selector import run_feature_selection
from src.models.clustering import run_problem1_clustering
from src.models.classification import run_problem2_classification
from src.models.regression import run_problem3_regression
from src.visualization.dashboard_builder import build_interactive_dashboard

def main():
    start_time = time.time()
    print("=" * 70)
    print("   AIR QUALITY EDA & MACHINE LEARNING END-TO-END PIPELINE")
    print("   Course: IT 462 | DA-IICT | Dr. Gopinath Panda")
    print("=" * 70)
    
    # Phase 2: Audit
    print("\n>>> Phase 2: Dataset Quality Audit")
    audit_datasets()
    
    # Phase 3: Data Engineering & Integration
    print("\n>>> Phase 3: Ingestion, CPCB NAQI Calculation & Master Integration")
    process_cpcb_data()
    process_weather_data()
    process_firms_data()
    process_calendar_data()
    run_imputation_experiment()
    build_master_dataset()
    
    # Phase 4: Exploratory Data Analysis & Visualizations
    print("\n>>> Phase 4: Publication Visualizations (Figures 1-10)")
    generate_all_eda_plots()
    
    # Phase 5: Feature Engineering & Selection
    print("\n>>> Phase 5: Leakage-Safe Feature Engineering & Selection")
    engineer_all_features()
    run_feature_selection()
    
    # Phase 6 & 7: Machine Learning Problems 1, 2, 3
    print("\n>>> Phase 6 & 7: Machine Learning Implementation & Inference")
    print("\n--- Problem 1: Unsupervised City Profiles ---")
    run_problem1_clustering()
    
    print("\n--- Problem 2: Severe Episode Forecasting (Classification) ---")
    run_problem2_classification()
    
    print("\n--- Problem 3: Drivers of PM2.5 (Regression & Inference) ---")
    run_problem3_regression()
    
    # Interactive Standalone Dashboard
    print("\n>>> Interactive Dashboard Compilation")
    build_interactive_dashboard()
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.1f} SECONDS")
    print("All artifacts, tables, models, figures, notebooks, and dashboard are ready!")
    print("=" * 70)

if __name__ == "__main__":
    main()
