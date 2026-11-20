# Air Quality EDA & Machine Learning Project

**Title:** *Exploratory Data Analysis of Air Pollution in Indian Cities: Pollutant Signatures, Severe-Episode Forecasting, and the Role of Weather, Festivals and Crop Burning*  
**Course:** IT 462 — Exploratory Data Analysis | Autumn Semester  
**Instructor / Academic Guide:** Dr. Gopinath Panda, DA-IICT  

---

## 1. Project Overview

This project provides an end-to-end, scientifically defensible, and reproducible data science study of ambient air quality dynamics across **12 major Indian cities** (Delhi, Mumbai, Bengaluru, Hyderabad, Kolkata, Chennai, Ahmedabad, Surat, Pune, Jaipur, Lucknow, Patna) spanning **155 CAAQMS monitoring stations** and over **5.23 million 15-minute observations**.

The core technical work addresses three fundamental environmental data science problems:
1. **Problem 1 (Unsupervised City Profiles):** Standardized PCA and K-Means/Hierarchical clustering giving 3 descriptive clusters (Delhi as an extreme outlier, a winter-smog belt, and cleaner coastal/southern cities); PC1+PC2 explain **79.45%** of variance.
2. **Problem 2 (Severe Episode Forecasting):** Next-day severe pollution (PM2.5 in the top 12% of city-days) evaluated out-of-sample against the **Persistence Baseline**. Models beat persistence on ROC-AUC (0.88–0.94 vs 0.85) and PR-AUC, but not clearly on F1; ROC-AUC on unseen cities is 0.937.
3. **Problem 3 (Drivers of PM2.5 — Econometric Inference):** Econometric panel regression with **City Fixed Effects** and **Cluster-Robust Standard Errors** ($R^2 = 0.859$, Durbin-Watson = 1.946), finding significant short-run effects of wind speed ($\beta = -0.783$) and temperature ($\beta = -0.866$) given yesterday's PM2.5, and a significant Diwali surge ($+4.80\,\mu\text{g/m}^3$, $p < 0.001$).

---

## 2. Directory Architecture

```text
air_quality_eda/
├── data/
│   ├── raw/                  # Original raw archives (CPCB, Open-Meteo, FIRMS, Calendar)
│   ├── interim/              # Validated daily aggregations (aq_daily, weather_daily, fires_daily)
│   └── processed/            # Master city-day panel (master.csv, master_features.csv)
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_clustering.ipynb
│   ├── 06_classification.ipynb
│   └── 07_regression.ipynb
│
├── src/
│   ├── data/
│   │   ├── audit.py          # Raw data schema, missingness, and coverage auditor
│   │   ├── ingest_cpcb.py    # CPCB 15-min chunk ingestion and daily aggregation
│   │   ├── ingest_weather.py # Open-Meteo hourly reanalysis processing
│   │   ├── ingest_fires.py   # NASA FIRMS Punjab/Haryana fire filtering & FRP
│   │   ├── ingest_calendar.py# Cultural and festival window construction
│   │   └── build_master.py   # Merged panel & consolidated 7-sheet Excel generator
│   ├── preprocessing/
│   │   ├── clean.py          # CPCB NAQI formula engine & sub-index breakpoints
│   │   └── impute.py         # 10% masking imputation experiment (Linear, KNN, MICE)
│   ├── features/
│   │   ├── engineer.py       # Autoregressive lags, rolling stats, ventilation index
│   │   └── feature_selector.py # Mutual info ranking, Lasso shrinkage, VIF diagnostics
│   ├── models/
│   │   ├── clustering.py     # Problem 1: PCA biplot, K-Means, Ward's dendrogram
│   │   ├── classification.py # Problem 2: Persistence baseline, RF, GBDT, PR-AUC
│   │   └── regression.py     # Problem 3: Fixed effects, clustered SEs, Ridge, RF
│   ├── visualization/
│   │   ├── plots.py          # High-resolution (300 DPI) publication figure exports
│   │   └── dashboard_builder.py # Interactive standalone HTML5 dashboard compiler
│   └── utils/
│       └── config.py         # Paths, seed controls, and configuration loaders
│
├── outputs/
│   ├── figures/              # Figures 1 to 18 (300 DPI publication charts)
│   ├── tables/               # Model comparison CSVs, feature dictionary, Excel workbook
│   ├── models/               # Serialized trained model pipelines (.joblib)
│   ├── dashboard/            # Standalone interactive dashboard (dashboard.html)
│   └── reports/              # Dataset quality audit report and monograph documentation
│
├── tests/
│   ├── test_aqi_calc.py      # CPCB sub-index logic and validity condition tests
│   ├── test_leakage.py       # Temporal leakage prevention and target alignment tests
│   └── test_shapes.py        # Dataset referential integrity and schema tests
│
├── configs/
│   ├── cities.json           # 12 cities metadata, coordinates, elevation, state
│   └── cpcb_breakpoints.json # Official CPCB NAQI breakpoints & category bands
│
├── main.py                   # Master pipeline runner orchestrating all stages
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation and execution instructions
```

---

## 3. Quick Start & Reproducibility Instructions

### 3.1 Environment Setup
```bash
# Clone or navigate to the repository
cd air_quality_eda

# Install dependencies
pip install -r requirements.txt
```

### 3.2 Execute the Full Pipeline
To run all data auditing, cleaning, NAQI calculation, feature engineering, machine learning modeling, figure generation, and dashboard compilation in one command:
```bash
python main.py
```

### 3.3 Run the Automated Test Suite
```bash
pytest tests/
```

### 3.4 Interactive Dashboard
Open `outputs/dashboard/dashboard.html` in any web browser. It features:
* **Interactive Map of India:** Displays 12 cities and 155 CAAQMS monitoring stations color-coded by severity.
* **City Profile Inspector:** Displays multi-pollutant summaries and longitudinal 2025 time-series trends.
* **Problem 1 (Profiles):** PCA 2D biplot and hierarchical dendrogram.
* **Problem 2 (Forecasting Simulator):** Interactive sliders for PM2.5, wind speed, boundary-layer height, and fire counts calculating real-time next-day severe episode probabilities.
* **Problem 3 (Econometric Drivers):** Marginal effects forest plot, residual diagnostics, and predictive ML benchmarks.

---

## 4. Key Experimental Results

### Problem 1: City Pollution Profiles (Unsupervised Learning)
* **Dimensionality Reduction:** PC1 explains **62.07%** of variance and PC2 **17.39%** (cumulative **79.45%**). The biplot axes are labelled with their top loadings.
* **Choosing k:** silhouette is highest at k=2 (0.476), but that only isolates Delhi from the other eleven cities. k=3 (0.283) is the smallest k that also splits the remaining cities; k=4 (0.316) only adds a second singleton (Surat). With 12 cities this is descriptive, not a strong statistical result.
* **Cluster Profiles ($k=3$):** K-Means and Ward hierarchical clustering give identical partitions (ARI = 1.00). Clusters are named by rank on mean PM2.5:
  1. *Extreme Smog Outlier* (Delhi): mean PM2.5 96.4 µg/m³, winter/monsoon ratio 4.39, 44% severe days.
  2. *Winter-Smog Belt & Semi-Arid* (Ahmedabad, Jaipur, Kolkata, Lucknow, Patna, Pune): mean PM2.5 49.6, winter/monsoon 2.95, 15% severe days.
  3. *Cleaner Coastal & Southern* (Bengaluru, Chennai, Hyderabad, Mumbai, Surat): mean PM2.5 32.2, winter/monsoon 1.89, 1% severe days.
* **Stability (`problem1_stability.csv`):** bootstrapping days within city gives mean ARI 0.969 (5th percentile 0.683). Re-clustering on half-years is **weak**: ARI 0.80 for Jan–Jun but only 0.11 for Jul–Dec, because the monsoon-heavy second half has little between-city contrast outside Delhi. Treat the grouping as a winter-driven pattern. Only one year of data exists, so the blueprint's per-year check could not be run.

### Problem 2: Severe Episode Prediction (Classification)
* **Severe definition:** next-day PM2.5 >= **78.39 µg/m³** (top 12% of all city-days; stored in `outputs/tables/severe_threshold.json`). The earlier fixed rule (AQI >= 300 or PM2.5 >= 150) left only 1.6% positives in training.
* **Chronological Split:** Train (Jan–Aug 2025, 2,706 samples, 7.7% severe), Validation (Sep–Oct 20, 576 samples, 3.8% severe), Test (Oct 21–Dec 31, 819 samples, 31.5% severe). Decision thresholds are tuned on validation only. PM2.5 (the target) is never imputed, so days with missing PM2.5 are dropped.
* **Benchmark (test set):**

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Persistence baseline | 0.782 | 0.791 | 0.786 | 0.845 | 0.684 |
| Logistic Regression (balanced) | 0.737 | 0.857 | **0.792** | 0.937 | 0.879 |
| Random Forest | 0.673 | 0.934 | 0.782 | 0.932 | 0.857 |
| HistGradientBoosting | 0.718 | 0.837 | 0.773 | 0.927 | 0.863 |
| SVC (RBF) | 0.624 | 0.965 | 0.758 | 0.880 | 0.681 |

* **Honest reading:** models clearly beat persistence on ranking quality (ROC-AUC, PR-AUC) but **not on F1** at the chosen cut-off (only Logistic Regression is marginally ahead, 0.792 vs 0.786). The test period is a single winter season, so differences this small are within noise.
* **Cross-City Generalization (unseen Jaipur, Lucknow, Kolkata):** ROC-AUC = **0.937**, Severe F1 = **0.681**.
* **Interpretation:** permutation importance (Fig 16, `problem2_permutation_importance.csv`) ranks today's PM2.5 and PM10 highest.

### Problem 3: Drivers of PM2.5 (Regression & Inference)
* **Econometric Panel OLS (City Fixed Effects & Clustered SEs):** $R^2 = \mathbf{0.859}$, $DW = \mathbf{1.946}$. The model includes yesterday's PM2.5 (`pm25_lag_1`, $\beta = 0.825$), so the weather coefficients below are *short-run effects holding yesterday's PM2.5 fixed*, not total effects.
  - Ambient Temperature: $\beta = -0.866$ ($p < 0.001$, 95% CI: $[-1.30, -0.43]$)
  - Wind Speed: $\beta = -0.783$ ($p = 0.001$, 95% CI: $[-1.26, -0.31]$)
  - Diwali Festive Window: $\beta = +4.80\,\mu\text{g/m}^3$ ($p < 0.001$, 95% CI: $[+2.55, +7.06]$)
  - Relative Humidity: $\beta = -0.076$ ($p = 0.045$, marginal)
  - Rainfall: $\beta = -0.050$ ($p = 0.057$, not significant) and lagged NW-India fire count: $\beta = 0.001$ ($p = 0.457$, no detectable effect once yesterday's PM2.5 is controlled for; a negative result worth stating).
* **Hypothesis Tests:**
  - Diwali vs Baseline Non-Festival: Mann-Whitney U = 249,863.5 ($p = 2.5 \times 10^{-12}$, Reject $H_0$; medians 60.4 vs 36.4 µg/m³)
  - Seasonality: Kruskal-Wallis $H = 1,672.0$ ($p \approx 0$, Reject $H_0$); no post-hoc test yet.
  - Weekend Effect: Mann-Whitney U = 1,871,506 ($p = 0.677$, fail to reject $H_0$: no detectable weekend lull).
* **Predictive ML Benchmark (Test Set):**
  - City-Mean Baseline: RMSE = 48.08 µg/m³, $R^2 = -0.106$
  - Ridge Regression: RMSE = **15.63 µg/m³**, $R^2 = \mathbf{0.883}$
  - Lasso Regression: RMSE = **15.87 µg/m³**, $R^2 = \mathbf{0.879}$
  - Random Forest: RMSE = **20.01 µg/m³**, $R^2 = \mathbf{0.808}$; HistGradientBoosting: RMSE = 20.60, $R^2 = 0.797$
  - Linear models beat tree models, which suggests the relationships are mostly linear given the autoregressive term.

---

## 5. Non-Kaggle Data Provenance
1. **Central Pollution Control Board (CPCB) / Vayuayan:** Regulatory CAAQMS stations.
2. **Open-Meteo Historical Weather API:** ECMWF/ERA5 reanalysis surface meteorology.
3. **NASA FIRMS (Fire Information for Resource Management System):** VIIRS & MODIS active fire detections.
4. **National Astronomical/Cultural Calendar:** Solar-lunar festival dates.

---

## 6. License
Academic project conducted under course IT 462 at DA-IICT. Released under the MIT License.
