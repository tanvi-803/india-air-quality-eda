"""Publication-quality Exploratory Data Analysis (EDA) visualizations generator."""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import PROCESSED_DATA_DIR, FIGURES_DIR, TABLES_DIR

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def generate_all_eda_plots():
    print("=" * 60)
    print("GENERATING PUBLICATION-GRADE EDA VISUALIZATIONS")
    print("=" * 60)
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = pd.read_csv(PROCESSED_DATA_DIR / "master.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['month_name'] = df['date'].dt.strftime('%b')
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # -------------------------------------------------------------
    # Figure 1: Univariate Distributions (PM2.5 raw, log, and AQI)
    # -------------------------------------------------------------
    print("[Plots] Generating Fig 1: Univariate Distributions...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Raw PM2.5
    sns.histplot(df['PM2.5'].dropna(), kde=True, ax=axes[0], color='#2b5c8f', bins=40)
    axes[0].set_title("Fig 1A: Distribution of Daily Mean PM2.5")
    axes[0].set_xlabel("PM2.5 Concentration (µg/m³)")
    axes[0].set_ylabel("Frequency (City-Days)")
    axes[0].axvline(df['PM2.5'].mean(), color='red', linestyle='--', label=f"Mean: {df['PM2.5'].mean():.1f}")
    axes[0].axvline(df['PM2.5'].median(), color='orange', linestyle='-', label=f"Median: {df['PM2.5'].median():.1f}")
    axes[0].legend()
    
    # Log PM2.5
    log_pm25 = np.log1p(df['PM2.5'].dropna())
    sns.histplot(log_pm25, kde=True, ax=axes[1], color='#1f77b4', bins=40)
    axes[1].set_title("Fig 1B: Log-Transformed log(PM2.5 + 1)")
    axes[1].set_xlabel("log(PM2.5 + 1)")
    axes[1].set_ylabel("Frequency (City-Days)")
    
    # Official CPCB AQI
    valid_aqi = df['AQI'].dropna()
    sns.histplot(valid_aqi, kde=True, ax=axes[2], color='#d95f02', bins=40)
    axes[2].set_title("Fig 1C: CPCB National Air Quality Index (NAQI)")
    axes[2].set_xlabel("AQI Index Value (0 - 500)")
    axes[2].set_ylabel("Frequency (City-Days)")
    axes[2].axvline(300, color='darkred', linestyle=':', linewidth=2, label="Severe Threshold (AQI 300)")
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig01_univariate_distributions.png")
    plt.close()

    # -------------------------------------------------------------
    # Figure 2: Pollutant Boxplots by City
    # -------------------------------------------------------------
    print("[Plots] Generating Fig 2: PM2.5 & PM10 Distributions Across 12 Cities...")
    fig, axes = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
    
    # Sort cities by median PM2.5
    city_order = df.groupby('city')['PM2.5'].median().sort_values(ascending=False).index.tolist()
    
    sns.boxplot(data=df, x='city', y='PM2.5', order=city_order, ax=axes[0], palette='Blues_r', hue='city', legend=False)
    axes[0].set_title("Fig 2A: Daily PM2.5 Concentration by City (Sorted by Median)")
    axes[0].set_ylabel("PM2.5 (µg/m³)")
    axes[0].axhline(60, color='red', linestyle='--', label="NAAQS 24-hr Standard (60 µg/m³)")
    axes[0].legend(loc='upper right')
    
    sns.boxplot(data=df, x='city', y='PM10', order=city_order, ax=axes[1], palette='Oranges_r', hue='city', legend=False)
    axes[1].set_title("Fig 2B: Daily PM10 Concentration by City")
    axes[1].set_ylabel("PM10 (µg/m³)")
    axes[1].set_xlabel("City")
    axes[1].axhline(100, color='red', linestyle='--', label="NAAQS 24-hr Standard (100 µg/m³)")
    axes[1].legend(loc='upper right')
    plt.xticks(rotation=30)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig02_pollutants_by_city_boxplot.png")
    plt.close()

    # -------------------------------------------------------------
    # Figure 3: Monthly Pollution Dynamics
    # -------------------------------------------------------------
    print("[Plots] Generating Fig 3: Monthly Seasonal Dynamics...")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x='month_name', y='PM2.5', order=month_order, ax=ax, palette='coolwarm', hue='month_name', legend=False)
    ax.set_title("Fig 3: Monthly Distribution of PM2.5 Across All Cities (Seasonal Inversion & Monsoon Washout)")
    ax.set_xlabel("Month")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.axhline(60, color='red', linestyle='--', label="National Ambient Standard (60 µg/m³)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig03_monthly_pollution_dynamics.png")
    plt.close()

    # -------------------------------------------------------------
    # Figure 4: Missingness Patterns Across Pollutants and Cities
    # -------------------------------------------------------------
    print("[Plots] Generating Fig 4: Missingness Patterns...")
    st_cov = pd.read_csv(TABLES_DIR / "cpcb_city_station_coverage.csv")
    miss_cols = [c for c in st_cov.columns if '_Missing_%' in c]
    df_miss = st_cov.set_index('City')[miss_cols]
    df_miss.columns = [c.replace('_Missing_%', '') for c in df_miss.columns]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(df_miss, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Missing Percentage (%)'}, ax=ax)
    ax.set_title("Fig 4: Raw 15-Minute Data Missingness Percentage by City and Pollutant")
    ax.set_ylabel("City")
    ax.set_xlabel("Pollutant Indicator")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig04_missingness_patterns.png")
    plt.close()

    # -------------------------------------------------------------
    # Figure 5: Correlation Heatmap (Pollutants & Weather)
    # -------------------------------------------------------------
    print("[Plots] Generating Fig 5: Correlation Heatmap...")
    corr_cols = [
        'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'NH3',
        'temp_mean', 'humidity_mean', 'wind_speed_mean',
        'precip_sum', 'boundary_layer_height_mean', 'ventilation_index',
        'fire_count_NW_India'
    ]
    corr_labels = [
        'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'NH3',
        'Temp', 'Humidity', 'Wind Speed',
        'Precip', 'BLH', 'Ventilation', 'NW Fires'
    ]
    df_corr = df[corr_cols].dropna().corr(method='spearman')
    df_corr.index = corr_labels
    df_corr.columns = corr_labels
    
    fig, ax = plt.subplots(figsize=(13, 10))
    mask = np.triu(np.ones_like(df_corr, dtype=bool))
    sns.heatmap(df_corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, center=0, ax=ax)
    ax.set_title("Fig 5: Spearman Rank Correlation Matrix: Criteria Pollutants vs. Meteorological & Fire Drivers")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig05_correlation_heatmap.png")
    plt.close()

    # -------------------------------------------------------------
    # Figure 6: City-by-Month PM2.5 Heatmap
    # -------------------------------------------------------------
    print("[Plots] Generating Fig 6: City-by-Month Seasonality Heatmap...")
    pivot_city_month = df.pivot_table(index='city', columns='month_name', values='PM2.5', aggfunc='mean')[month_order]
    # Re-order cities from north to south
    north_to_south = df.groupby('city')['latitude'].mean().sort_values(ascending=False).index.tolist()
    pivot_city_month = pivot_city_month.loc[north_to_south]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot_city_month, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Mean PM2.5 (µg/m³)'}, ax=ax)
    ax.set_title("Fig 6: City-by-Month Mean PM2.5 Concentration (Arranged North to South Latitude)")
    ax.set_xlabel("Month")
    ax.set_ylabel("City (North to South)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig06_city_month_heatmap.png")
    plt.close()

    # -------------------------------------------------------------
    # Figure 7: PM2.5 vs Meteorological Drivers
    # -------------------------------------------------------------
    print("[Plots] Generating Fig 7: Meteorological Drivers Scatter Plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # PM2.5 vs Wind Speed
    sns.scatterplot(data=df, x='wind_speed_mean', y='PM2.5', hue='season', alpha=0.5, ax=axes[0], palette='tab10')
    sns.regplot(data=df, x='wind_speed_mean', y='PM2.5', scatter=False, ax=axes[0], color='black', line_kws={'linestyle':'--'})
    axes[0].set_title("Fig 7A: PM2.5 vs. Horizontal Wind Speed")
    axes[0].set_xlabel("Wind Speed (km/h)")
    axes[0].set_ylabel("PM2.5 (µg/m³)")
    
    # PM2.5 vs Boundary Layer Height
    sns.scatterplot(data=df, x='boundary_layer_height_mean', y='PM2.5', hue='season', alpha=0.5, ax=axes[1], palette='tab10', legend=False)
    sns.regplot(data=df, x='boundary_layer_height_mean', y='PM2.5', scatter=False, ax=axes[1], color='black', line_kws={'linestyle':'--'})
    axes[1].set_title("Fig 7B: PM2.5 vs. Boundary Layer Height (BLH)")
    axes[1].set_xlabel("Planetary Boundary Layer Height (m)")
    axes[1].set_ylabel("PM2.5 (µg/m³)")
    
    # PM2.5 vs Ventilation Index
    sns.scatterplot(data=df, x='ventilation_index', y='PM2.5', hue='season', alpha=0.5, ax=axes[2], palette='tab10')
    sns.regplot(data=df, x='ventilation_index', y='PM2.5', scatter=False, ax=axes[2], color='black', line_kws={'linestyle':'--'})
    axes[2].set_title("Fig 7C: PM2.5 vs. Atmospheric Ventilation Index")
    axes[2].set_xlabel("Ventilation Index (m²/s)")
    axes[2].set_ylabel("PM2.5 (µg/m³)")
    axes[2].set_xlim(0, 6000)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig07_pm25_vs_meteorology.png")
    plt.close()

    # -------------------------------------------------------------
    # Figure 8: Time-Series for Contrasting Cities
    # -------------------------------------------------------------
    print("[Plots] Generating Fig 8: Longitudinal Time-Series Comparison...")
    fig, ax = plt.subplots(figsize=(16, 6))
    
    contrasting_cities = ['Delhi', 'Mumbai', 'Bengaluru']
    palette = {'Delhi': '#d95f02', 'Mumbai': '#7570b3', 'Bengaluru': '#1b9e77'}
    
    for c in contrasting_cities:
        sub = df[df['city'] == c].sort_values('date')
        # Plot 7-day rolling mean for clarity
        sub['pm25_roll7'] = sub['PM2.5'].rolling(7, min_periods=1).mean()
        ax.plot(sub['date'], sub['pm25_roll7'], label=f"{c} (7-day MA)", color=palette[c], linewidth=2)
        
    # Mark Diwali 2025 (October 20, 2025)
    diwali_2025 = pd.to_datetime('2025-10-20')
    ax.axvline(diwali_2025, color='gold', linestyle='--', linewidth=2.5, label="Diwali (Oct 20, 2025)")
    ax.text(diwali_2025, 280, " Diwali 2025", color='darkgoldenrod', fontweight='bold')
    
    # Highlight stubble burning window (Oct 25 - Nov 25)
    ax.axvspan(pd.to_datetime('2025-10-25'), pd.to_datetime('2025-11-25'), alpha=0.15, color='crimson', label="Post-Monsoon Stubble Burning Window")
    
    ax.set_title("Fig 8: 2025 Daily PM2.5 Trajectories Across Contrasting Urban Environments (Delhi, Mumbai, Bengaluru)")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig08_time_series_contrasting_cities.png")
    plt.close()

    # -------------------------------------------------------------
    # Figure 9: Punjab/Haryana Fires vs. Delhi PM2.5 with Lag Dynamics
    # -------------------------------------------------------------
    print("[Plots] Generating Fig 9: Satellite Fires vs. Delhi PM2.5...")
    df_delhi = df[df['city'] == 'Delhi'].sort_values('date').copy()
    
    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax2 = ax1.twinx()
    
    # Filter to autumn harvest period for clear visual inspection
    autumn_mask = (df_delhi['date'] >= '2025-09-15') & (df_delhi['date'] <= '2025-12-15')
    sub_d = df_delhi[autumn_mask]
    
    p1, = ax1.plot(sub_d['date'], sub_d['PM2.5'], color='#d95f02', linewidth=2.5, label="Delhi PM2.5 (µg/m³)")
    p2, = ax2.plot(sub_d['date'], sub_d['fire_count_NW_India'], color='#7570b3', linewidth=2, linestyle='--', label="NW India Fire Detections (VIIRS/MODIS)")
    
    ax1.set_title("Fig 9: Agricultural Stubble Fire Detections vs. Delhi Ambient PM2.5 During Post-Monsoon Season")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Delhi PM2.5 (µg/m³)", color='#d95f02')
    ax2.set_ylabel("Punjab & Haryana Fire Count", color='#7570b3')
    ax1.tick_params(axis='y', labelcolor='#d95f02')
    ax2.tick_params(axis='y', labelcolor='#7570b3')
    
    lines = [p1, p2]
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper left')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig09_stubble_burning_delhi_lag.png")
    plt.close()

    # -------------------------------------------------------------
    # Figure 10: Diwali Festival Pollution Shock Analysis
    # -------------------------------------------------------------
    print("[Plots] Generating Fig 10: Diwali Festival Shock Analysis...")
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Classify days into: Non-Diwali, Diwali Window (+/- 3 days), and Diwali Day
    df['diwali_status'] = np.where(df['diwali_window'] == 1, "Diwali Window (±3 days)", "Baseline Non-Festival Days")
    diwali_exact_mask = df['date'] == '2025-10-20'
    df.loc[diwali_exact_mask, 'diwali_status'] = "Diwali Night Peak"
    
    status_order = ["Baseline Non-Festival Days", "Diwali Window (±3 days)", "Diwali Night Peak"]
    palette_diw = {"Baseline Non-Festival Days": "#1f77b4", "Diwali Window (±3 days)": "#ff7f0e", "Diwali Night Peak": "#d62728"}
    
    sns.boxplot(data=df, x='diwali_status', y='PM2.5', order=status_order, palette=palette_diw, ax=ax, showmeans=True, hue='diwali_status', legend=False)
    ax.set_title("Fig 10: Empirical PM2.5 Surge During Diwali Festival Window vs. Baseline Days")
    ax.set_xlabel("Festive Interval Classification")
    ax.set_ylabel("PM2.5 (µg/m³)")
    
    # Annotate means
    means = df.groupby('diwali_status')['PM2.5'].mean()
    for idx, st in enumerate(status_order):
        m_val = means.get(st, 0)
        ax.text(idx, m_val + 5, f"Mean: {m_val:.1f} µg/m³", horizontalalignment='center', fontweight='bold', color='black')
        
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig10_diwali_impact_analysis.png")
    plt.close()
    
    print(f"[Plots] Successfully generated Figures 1 through 10 in {FIGURES_DIR}/")

if __name__ == "__main__":
    generate_all_eda_plots()
