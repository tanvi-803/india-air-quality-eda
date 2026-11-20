# Exploratory Data Analysis of Air Pollution in Indian Cities: Pollutant Signatures, Severe-Episode Forecasting, and the Role of Weather, Festivals and Crop Burning

**Course Code:** IT 462 — Exploratory Data Analysis  
**Semester:** Autumn Semester  
**Institution:** Dhirubhai Ambani Institute of Information and Communication Technology (DA-IICT), Gandhinagar, Gujarat, INDIA  
**Project Guide / Course Instructor:** Dr. Gopinath Panda  

---

## Preliminary Pages

### Acknowledgment
We express our sincere gratitude and indebtedness to our course instructor and project mentor, **Dr. Gopinath Panda**, for his continuous guidance, rigorous academic critique, and methodological encouragement throughout the duration of this Exploratory Data Analysis project. His insights into statistical rigor, leakage-free temporal modeling, and econometric inference significantly elevated the quality and depth of our research.

We also thank the **Central Pollution Control Board (CPCB)**, **Open-Meteo**, and **NASA FIRMS** for providing open, publicly accessible environmental, meteorological, and satellite active-fire observation archives. Finally, we thank our peers at DA-IICT for their constructive discussions and camaraderie throughout this investigation.

*Group Members: [Student Name 1, Student Name 2, Student Name 3]*  
*DA-IICT, Gandhinagar*

---

### Declaration
We hereby declare that the exploratory data analysis and machine learning work presented in this report titled **"Exploratory Data Analysis of Air Pollution in Indian Cities: Pollutant Signatures, Severe-Episode Forecasting, and the Role of Weather, Festivals and Crop Burning"** is our original work and has not been submitted for any other academic degree or credential. All sources, data repositories, scientific literature, and external packages utilized have been referenced appropriately.

We confirm that all datasets utilized in this investigation were retrieved from legitimate non-Kaggle public research portals (CPCB/Vayuayan, Open-Meteo, NASA FIRMS, and verified astronomical calendars) adhering to their terms of use. We confirm that no real-world values were fabricated, imputed targets were not used in predictive modeling, and all experimental results presented reflect genuine empirical outputs.

*Signed,*  
*[Member 1 - ID: 2025XXXXX]*  
*[Member 2 - ID: 2024XXXXX]*  
*[Member 3 - ID: 2023XXXXX]*  

---

### Certificate
This is to certify that the project report titled **"Exploratory Data Analysis of Air Pollution in Indian Cities: Pollutant Signatures, Severe-Episode Forecasting, and the Role of Weather, Festivals and Crop Burning"** submitted by Group Members is a bonafide record of research work carried out under my supervision in partial fulfillment of the requirements for the course **IT 462: Exploratory Data Analysis** during the Autumn Semester at the Dhirubhai Ambani Institute of Information and Communication Technology, Gandhinagar, Gujarat, India.

*Signed,*  
**Dr. Gopinath Panda**  
IT 462 Course Instructor  
Dhirubhai Ambani Institute of Information and Communication Technology  
Gandhinagar, Gujarat, INDIA  

---

## Contents
- **List of Figures**
  - Figure 1: Univariate Distributions of PM2.5 (Raw & Log) and CPCB NAQI
  - Figure 2: Daily PM2.5 and PM10 Concentration by City (Boxplots)
  - Figure 3: Monthly Distribution of PM2.5 Across All Cities (Seasonality)
  - Figure 4: Raw 15-Minute Data Missingness Percentage by City and Pollutant
  - Figure 5: Spearman Rank Correlation Matrix Across Criteria Pollutants & Weather
  - Figure 6: City-by-Month Mean PM2.5 Concentration Heatmap (North to South)
  - Figure 7: PM2.5 vs Horizontal Wind, Boundary Layer Height & Ventilation Index
  - Figure 8: 2025 Daily PM2.5 Trajectories Across Contrasting Urban Environments
  - Figure 9: Agricultural Stubble Fire Detections vs. Delhi Ambient PM2.5
  - Figure 10: Empirical PM2.5 Surge During Diwali Festival Window vs Baseline
  - Figure 11: PCA Biplot of City Pollution Profiles (PC1 vs PC2)
  - Figure 12: Hierarchical Clustering Dendrogram (Ward Linkage)
  - Figure 13: Normalized Multi-Pollutant & Seasonal Signatures Across Clusters
  - Figure 14: Receiver Operating Characteristic (ROC) & Precision-Recall (PR) Curves
  - Figure 15: Confusion Matrices for Persistence Baseline vs. Random Forest
  - Figure 16: Random Forest Gini Feature Importances for Next-Day Severe Forecasting
  - Figure 17: Econometric Panel Regression Coefficients with 95% Clustered CIs
  - Figure 18: Econometric Residual Diagnostics (Residuals vs Fitted & Normal Q-Q)
- **List of Tables**
  - Table 1: Raw Dataset Inventory and Source Provenance
  - Table 2: City and CAAQMS Monitoring Station Audit Summary
  - Table 3: 10% Masking Benchmark Evaluation of Imputation Algorithms
  - Table 4: Feature Selection Diagnostics (Mutual Information, Lasso & VIF)
  - Table 5: Problem 1 Clustering Evaluation Metrics (k=2 to 5)
  - Table 6: Problem 1 City Cluster Assignments and Atmospheric Characteristics
  - Table 7: Problem 2 Next-Day Severe Classification Performance Comparison
  - Table 8: Problem 2 Spatial Cross-City Generalization on Unseen Cities
  - Table 9: Problem 3 Non-Parametric Hypothesis Tests (Diwali, Season, Weekend)
  - Table 10: Problem 3 Econometric Fixed Effects Panel Regression Output
  - Table 11: Problem 3 Supervised Machine Learning Predictive Benchmark

---

# Chapter 1. Introduction

## 1.1 Project Idea and Motivation
Urban air pollution across the Indian subcontinent represents one of the most critical public health, environmental, and socio-economic challenges of the twenty-first century. While international discourse frequently labels Indian cities homogeneously as "severely polluted," atmospheric science reveals that Indian urban centers exhibit profound divergence in their chemical emission fingerprints, stoichiometric ratios, regional dispersion mechanics, and seasonal vulnerability. 

An arid city such as Jaipur suffers primarily from coarse crustal mineral dust ($\text{PM}_{10}$), coastal megacities like Mumbai and Chennai operate under marine boundary-layer ventilation, while the landlocked Indo-Gangetic Plains (IGP)—encompassing Delhi, Lucknow, and Patna—undergo catastrophic winter smog episodes driven by shallow nocturnal boundary-layer trapping, agricultural biomass burning plumes, and festive firework emissions.

This project formulates an applied, end-to-end data science study answering three core research problems:
1. **Problem 1 (City Pollution Profiles — Unsupervised Learning):** Do Indian cities form distinct, chemically and geographically coherent pollution profiles based on multi-pollutant mix ($\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2, \text{SO}_2, \text{CO}, \text{O}_3, \text{NH}_3$), seasonal amplitude (Winter/Monsoon ratios), and temporal volatility?
2. **Problem 2 (Severe Episode Forecasting — Classification):** Using strictly information available up to day $t$ (autoregressive lags, rolling statistics, meteorology, transboundary fire counts, and calendar flags), can we predict whether day $t+1$ will experience a severe pollution episode ($\text{PM}_{2.5} \ge 78.4\,\mu\text{g/m}^3$, the top 12% of city-days) outperforming a persistence baseline?
3. **Problem 3 (Drivers of PM2.5 — Econometric Regression & Inference):** After controlling for time-invariant city fixed effects, how do ambient temperature, relative humidity, horizontal wind velocity, precipitation, atmospheric boundary-layer height, festive fireworks (Diwali), and upwind agricultural stubble burning quantitatively explain daily $\text{PM}_{2.5}$ variations? Are these marginal effects statistically significant under cluster-robust standard errors?

## 1.2 Multi-Source Data Collection
To ensure scientific integrity, all data used in this project were retrieved exclusively from non-Kaggle, authoritative open repositories:
1. **Central Pollution Control Board (CPCB) / Vayuayan Mirror:** Continuous Ambient Air Quality Monitoring Station (CAAQMS) measurements recorded at 15-minute resolution across all official monitoring stations in India.
2. **Open-Meteo Historical Weather Reanalysis:** Hourly ECMWF/ERA5 atmospheric reanalysis providing surface temperature, relative humidity, barometric pressure, precipitation, 10m horizontal wind speed, wind direction, and planetary boundary-layer height ($BLH$).
3. **NASA FIRMS (Fire Information for Resource Management System):** Daily satellite active fire detections derived from the VIIRS (375m) and MODIS sensors across the Northwestern agricultural belt (Punjab and Haryana).
4. **Verified Cultural & National Astronomical Calendar:** Complete solar-lunar holiday records identifying festive windows ($\pm 3$ days surrounding Diwali and Holi) and weekend cycles.

## 1.3 Dataset Description
The target spatial domain encompasses **12 representative Indian metropolitan and industrial centers**: Ahmedabad, Bengaluru, Chennai, Delhi, Hyderabad, Jaipur, Kolkata, Lucknow, Mumbai, Patna, Pune, and Surat. Across these 12 cities, the regulatory monitoring network comprises **155 CAAQMS monitoring stations**. 

The temporal unit of analysis is the **city-day**. Over 5.23 million raw 15-minute observations were aggregated into 4,348 city-days. A consolidated multi-sheet Excel workbook (`air_quality_master.xlsx`) was engineered containing seven standardized sheets: `cities`, `aq_daily`, `weather_daily`, `fires_daily`, `calendar`, `master`, and `metadata`.

## 1.4 Packages and Computational Environment
The project was executed in a high-performance Python 3.14 analytical environment using:
- **Data Engineering:** `pandas` (3.0.5), `numpy` (2.5.2), `openpyxl` (3.1.5).
- **Statistical Modeling & Inference:** `statsmodels` (0.15.0), `scipy` (1.18.0).
- **Machine Learning & Preprocessing:** `scikit-learn` (1.9.0), `shap` (0.54.1).
- **Visualization:** `matplotlib` (3.11.1), `seaborn` (0.13.2), `missingno` (0.5.2).
- **Verification:** `pytest` (9.1.1).

---

# Chapter 2. Data Cleaning and Imputation

## 2.1 Missing Data Audit & Sensor Quality Control
Continuous ambient air quality monitoring instruments undergo periodic recalibration, zero-span calibration cycles, filter tape advances, and power outages, yielding genuine observational missingness. 

Our automated audit of the raw 15-minute readings revealed:
- $\text{PM}_{2.5}$ missing rate: **15.99%** (836,139 / 5,229,527 readings).
- $\text{PM}_{10}$ missing rate: **16.41%** (858,007 / 5,229,527 readings).
- $\text{NO}_2$ missing rate: **13.84%** (723,561 / 5,229,527 readings).
- $\text{SO}_2$ missing rate: **24.97%**; $\text{NH}_3$ missing rate: **27.06%**.

Data sanitization protocols were applied:
1. **Non-Physical Sentinel Values:** Negative concentrations, flat-line sequences exceeding 24 hours, and sensor overflow sentinels (e.g., 999, 9999, -999) were coerced to NaN.
2. **Physical Ceiling Filters:** Particulate readings exceeding physical atmospheric limits ($\text{PM}_{2.5} > 1500\,\mu\text{g/m}^3$, $\text{PM}_{10} > 2500\,\mu\text{g/m}^3$) were filtered out.
3. **Daily Station Validity Threshold:** A station-day was marked valid only if it contained $\ge 64$ valid 15-minute intervals ($\ge 16$ operational hours).
4. **Official CPCB NAQI Formulation:** The National Air Quality Index was computed using CPCB's piecewise linear sub-index formula. AQI was evaluated only when at least 3 criteria pollutants (including at least one particulate matter, $\text{PM}_{2.5}$ or $\text{PM}_{10}$) were present with $\ge 16$ valid hours.

## 2.2 Controlled Imputation & 10% Masking Benchmark Experiment
To avoid arbitrary imputation, a controlled **10% masking experiment** was executed on known daily $\text{PM}_{2.5}$ observations ($N=430$ evaluation points held out with fixed random seed $42$). Four distinct imputation strategies were evaluated against the masked ground truth:

| Imputation Methodology | Theoretical Paradigm | Evaluated Points | Test RMSE ($\mu\text{g/m}^3$) | Test MAE ($\mu\text{g/m}^3$) |
| :--- | :--- | :---: | :---: | :---: |
| **Linear Interpolation (7-day cap)** | Univariate Temporal Continuity | 430 | **11.72** | **6.25** |
| **Iterative Imputer (MICE, BayesianRidge)** | Multivariate Feature Regression | 430 | 16.10 | 8.00 |
| **Forward Fill (7-day cap)** | Univariate Autoregressive | 430 | 18.63 | 9.07 |
| **K-Nearest Neighbors (KNN, k=5)** | Multivariate Metric Space | 430 | 49.75 | 26.34 |

**Scientific Conclusion:** Linear interpolation strictly capped at 7 days achieved the lowest RMSE ($11.72\,\mu\text{g/m}^3$) and MAE ($6.25\,\mu\text{g/m}^3$), reflecting the high day-to-day temporal autocorrelation of air quality series. Gaps exceeding 7 consecutive days were retained as missing to prevent synthetic distortion, and prediction targets were never imputed.

---

# Chapter 3. Exploratory Data Analysis

## 3.1 Univariate Analysis
- **Particulate Asymmetry:** Raw $\text{PM}_{2.5}$ distributions exhibit extreme right-skewness (Mean = $52.8\,\mu\text{g/m}^3$, Median = $38.4\,\mu\text{g/m}^3$, Maximum = $445.6\,\mu\text{g/m}^3$). A natural logarithmic transformation ($\log(1 + \text{PM}_{2.5})$) normalizes the distribution, satisfying Gaussian linear regression assumptions (Figure 1).
- **National Ambient Standard Non-Compliance:** The National Ambient Air Quality Standard (NAAQS) 24-hour threshold ($60\,\mu\text{g/m}^3$) is regularly breached across all Northern and Eastern cities. Delhi, Patna, and Lucknow exceed the annual average threshold on over 65% of all days (Figure 2).
- **AQI Categorical Breakdown:** Out of 4,304 valid city-days, the CPCB NAQI categorizes 1,831 days as *Satisfactory*, 1,464 as *Moderate*, 682 as *Good*, 189 as *Poor*, 114 as *Very Poor*, and 24 as *Severe*.

## 3.2 Multivariate Analysis
- **Seasonal Inversion & Monsoon Washout:** Monthly PM2.5 boxplots (Figure 3) and the 12-city heatmap (Figure 6) reveal a dramatic seasonal cycle. During July–August, widespread south-westerly monsoon rainfall induces wet scavenging, depressing PM2.5 to annual minima ($<25\,\mu\text{g/m}^3$). Conversely, from late October through January, continental cold air stagnation, nocturnal ground inversions, and lower boundary-layer heights trigger severe pollutant accumulation.
- **Meteorological Dispersal Dynamics:** Spearman correlation analysis (Figure 5) reveals strong negative associations between PM2.5 and horizontal wind speed ($r = -0.42$), ambient temperature ($r = -0.48$), and planetary boundary layer height ($r = -0.45$). Scatter plots (Figure 7) illustrate a clear non-linear hyperbolic relationship with the **Atmospheric Ventilation Index** ($\text{Wind Speed} \times \text{BLH}$): when ventilation drops below $1,000\,\text{m}^2/\text{s}$, severe episodes become mathematically inevitable.
- **Transboundary Crop Fires & Delhi Airshed:** Time series overlay during the post-monsoon harvest (Figure 9) demonstrates that active fires in Punjab and Haryana spike to over 4,000 detections per day in late October/early November, followed within 24–48 hours by sharp PM2.5 surges in Delhi exceeding $350\,\mu\text{g/m}^3$.
- **Diwali Festive Firework Impact:** Boxplot analysis across the 12 cities (Figure 10) demonstrates that mean PM2.5 during the Diwali festive window ($\pm 3$ days) jumps to $92.4\,\mu\text{g/m}^3$ compared to the non-festival baseline of $51.6\,\mu\text{g/m}^3$ ($p = 3.94 \times 10^{-12}$).

---

# Chapter 4. Feature Engineering and Selection

## 4.1 Feature Extraction and Leakage Guardrails
To prevent data leakage, all engineered features for day $t+1$ forecasting were restricted strictly to historical information available up to day $t$:
1. **Autoregressive Lags:** $\text{PM}_{2.5}(t-1), \text{PM}_{2.5}(t-2), \text{PM}_{2.5}(t-3), \text{PM}_{2.5}(t-7)$, alongside $\text{AQI}(t-1)$ and $\text{NO}_2(t-1)$.
2. **Historical Rolling Statistics:** 3-day, 7-day, and 30-day backward-shifted rolling means and standard deviations capturing short-term persistence and intermediate baseline trends.
3. **Atmospheric Ventilation Proxy:** $\text{Ventilation Index} = (\text{Wind Speed} / 3.6) \times \text{Boundary Layer Height}$ ($\text{m}^2/\text{s}$).
4. **Vector Wind Components:** Zonal $u = -ws \cdot \sin(\theta)$ and meridional $v = -ws \cdot \cos(\theta)$ resolving directional transport.
5. **Combustion Source Indicator:** Ratio $\text{PM}_{2.5} / \text{PM}_{10}$ distinguishing fine combustion aerosols ($>0.6$) from coarse mechanical dust ($<0.4$).
6. **Transboundary Biomass Lags:** $\text{Fires}_{\text{NW}}(t-1), \text{Fires}_{\text{NW}}(t-2)$ and lagged mean FRP.
7. **Cyclical Calendar Encodings:** $\sin(2\pi \cdot m / 12)$ and $\cos(2\pi \cdot m / 12)$.
8. **Forecasting Targets:** $\text{Severe}_{t+1} \in \{0, 1\}$ (defined as $\text{AQI}_{t+1} \ge 300$ or $\text{PM}_{2.5, t+1} \ge 150\,\mu\text{g/m}^3$) and $\text{PM}_{2.5, t+1}$.

## 4.2 Feature Selection: Information Theory, Lasso & Collinearity
Feature selection was fitted strictly on the pre-monsoon training period:
- **Mutual Information Regression:** Ranked current-day $\text{PM}_{2.5}$ ($0.930$), $\text{PM}_{10}$ ($0.749$), 7-day rolling mean ($0.665$), and 1-day lag ($0.627$) as primary predictors.
- **Lasso L1 Shrinkage:** Successfully eliminated redundant collinear lags ($\text{pm25\_lag\_1}, \text{pm25\_lag\_3}$ shrank to zero in the presence of $\text{pm25\_roll7\_mean}$ and $\text{PM}_{2.5}$), retaining key meteorological drivers ($\text{humidity\_mean}: -2.14$, $\text{precip\_sum}: -0.29$).
- **Variance Inflation Factor (VIF):** Retained features exhibited VIF scores $<7.0$, confirming the absence of severe multicollinearity.

---

# Chapter 5. Model Fitting and Empirical Evaluation

## 5.1 Problem 1: Unsupervised Learning — City Pollution Profiles
- **PCA Dimensionality Reduction:** The first two principal components explain **79.45%** of cumulative variance (PC1: 62.07%, PC2: 17.39%). Axis labels in Fig 11 list the top loadings of each component.
- **Clustering Validation:** K-Means was evaluated across $k \in [2, 5]$. Silhouette is highest at $k=2$ ($0.476$), but that only separates Delhi from the other eleven cities. $k=3$ (Silhouette = $0.283$, Davies–Bouldin = $0.882$) is the smallest $k$ that also splits the remaining cities; $k=4$ ($0.316$) only adds a second singleton (Surat). K-Means and Ward hierarchical clustering give identical partitions (ARI = 1.00). With only 12 cities the result is descriptive.
- **Identified Profile Signatures (named by rank on mean PM2.5):**
  1. *Extreme Smog Outlier (Delhi):* mean $\text{PM}_{2.5}$ $96.4\,\mu\text{g/m}^3$, winter/monsoon ratio $4.39$, $44\%$ severe days.
  2. *Winter-Smog Belt & Semi-Arid (Ahmedabad, Jaipur, Kolkata, Lucknow, Patna, Pune):* mean $49.6\,\mu\text{g/m}^3$, ratio $2.95$, $15\%$ severe days.
  3. *Cleaner Coastal & Southern (Bengaluru, Chennai, Hyderabad, Mumbai, Surat):* mean $32.2\,\mu\text{g/m}^3$, ratio $1.89$, $1\%$ severe days.
- **Stability:** bootstrapping days within city gives mean ARI $0.969$; half-year re-clustering gives ARI $0.80$ (Jan–Jun) but only $0.11$ (Jul–Dec), so the grouping is driven by the winter season. Only one year of data is available, so a per-year check was not possible.

## 5.2 Problem 2: Severe Episode Prediction — Supervised Classification
Evaluated out-of-sample on the test period (October 21 to December 31, 2025; $N=819$ samples, severe prevalence = 31.5%). Severe is defined as next-day $\text{PM}_{2.5} \ge 78.39\,\mu\text{g/m}^3$ (top 12% of city-days); train has 7.7% and validation 3.8% severe days. Thresholds are tuned on validation only, and the target PM2.5 is never imputed.

| Model Architecture | Severe Precision | Severe Recall | Severe F1 | Test ROC-AUC | Test PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline (Today=Tomorrow)** | 0.782 | 0.791 | 0.786 | 0.845 | 0.684 |
| **Logistic Regression (Balanced)** | 0.737 | 0.857 | **0.792** | 0.937 | 0.879 |
| **Random Forest Classifier** | 0.673 | 0.934 | 0.782 | 0.932 | 0.857 |
| **HistGradientBoosting** | 0.718 | 0.837 | 0.773 | 0.927 | 0.863 |
| **Support Vector Classifier (RBF)** | 0.624 | 0.965 | 0.758 | 0.880 | 0.681 |

**Critical Scientific Findings:**
1. **Persistence is a strong baseline:** F1 of $0.786$ shows that stagnation locks cities into multi-day severe blocks. Only Logistic Regression exceeds it on F1 ($0.792$), by a margin within noise for one winter season.
2. **Models rank days better than persistence:** ROC-AUC $0.88$–$0.94$ vs $0.845$ and PR-AUC up to $0.879$ vs $0.684$ show better probability ranking, which helps when the alert threshold is adjustable.
3. **Spatial generalization:** trained on 9 cities and tested on 3 unseen cities (Jaipur, Lucknow, Kolkata), Random Forest reaches ROC-AUC $0.937$ and severe F1 $0.681$.
4. **Limitation:** the test period is one winter and the training period holds few severe days, so these results should be read as indicative, not definitive.

## 5.3 Problem 3: Drivers of PM2.5 — Regression & Inference
Panel OLS with city fixed effects and city-clustered standard errors ($R^2 = 0.859$, Durbin-Watson $= 1.946$). The model includes yesterday's PM2.5 ($\beta = 0.825$), so the coefficients below are short-run effects holding yesterday's level fixed, not total effects.

| Driver | Coefficient | 95% CI | p-value |
| :--- | :---: | :---: | :---: |
| Temperature (°C) | -0.866 | [-1.30, -0.43] | < 0.001 |
| Wind speed | -0.783 | [-1.26, -0.31] | 0.001 |
| Diwali window (±3 days) | +4.80 µg/m³ | [+2.55, +7.06] | < 0.001 |
| Relative humidity | -0.076 | [-0.15, -0.002] | 0.045 |
| Boundary-layer height | +0.007 | [+0.003, +0.011] | < 0.001 |
| Rainfall | -0.050 | [-0.10, +0.001] | 0.057 |
| NW-India fire count (lag 1) | +0.001 | [-0.001, +0.002] | 0.457 |

**Non-parametric tests:** Diwali vs baseline days, Mann-Whitney U $= 249{,}863.5$, $p = 2.5\times10^{-12}$ (medians 60.4 vs 36.4 µg/m³); seasons, Kruskal-Wallis $H = 1{,}672.0$, $p \approx 0$ (no post-hoc test yet); weekend vs weekday, $p = 0.677$ (no detectable weekend effect).

**Predictive benchmark (test set):** city-mean baseline RMSE $48.08$ ($R^2 = -0.106$); Ridge $15.63$ ($R^2 = 0.883$); Lasso $15.87$ ($0.879$); Random Forest $20.01$ ($0.808$); HistGradientBoosting $20.60$ ($0.797$). Linear models beat tree models, suggesting limited non-linearity once the autoregressive term is included.

**Negative results:** rainfall is only marginal ($p = 0.057$) and lagged NW-India fire counts show no detectable effect ($p = 0.457$) after controlling for yesterday's PM2.5, which already absorbs most of the transported-smoke signal.

## 6.1 Key Findings and Scientific Contributions
1. **Heterogeneous Urban Airsheds:** Cities fall into three descriptive profiles: Delhi as an extreme outlier, a winter-smog belt (Ahmedabad, Jaipur, Kolkata, Lucknow, Patna, Pune), and cleaner coastal/southern cities. The grouping is driven by winter and is unstable across half-years.
2. **Ventilation and Temperature Matter in the Short Run:** Wind speed ($-0.78$) and temperature ($-0.87$) lower next PM2.5 given yesterday's level, and boundary-layer height has a small positive coefficient.
3. **Diwali Effect:** after controlling for city, weather and autocorrelation, the Diwali window adds a significant $+4.80\,\mu\text{g/m}^3$. Lagged NW-India fire counts show no detectable effect in this model ($p = 0.457$), so we do not claim a stubble-burning link.
4. **Predictive Value Is Modest Against Persistence:** models improve ROC-AUC (up to $0.94$ vs $0.845$) and PR-AUC but match persistence on F1 ($0.79$ vs $0.786$) on a single winter test period.

## 6.2 Challenges and Methodological Limitations
1. **Temporal Horizon Overlap:** Weather, satellite fire and calendar data may span several years, but the CPCB regulatory data used here covers only 2025, so the analysis is single-year. Although 5.23 million readings provided statistical power, future expansions incorporating 2023–2024 CPCB data will enable multi-year cross-validation.
2. **Station Density Asymmetry:** Metros like Delhi (40 stations) and Mumbai (30 stations) provide dense spatial averaging, whereas Surat relies on a single CAAQMS monitor.
3. **Nocturnal Boundary-Layer Reanalysis Gaps:** Systematic missingness in Open-Meteo boundary-layer height required spline reconstruction.

## 6.3 Future Scope
1. **Aerosol Optical Depth (AOD) Integration:** Integrating satellite column AOD from Sentinel-5P or MODIS to capture total atmospheric aerosol column load.
2. **Deep Spatio-Temporal Graph Neural Networks:** Formulating spatial wind transport graphs to model inter-city transboundary smoke advection dynamically.
3. **Real-Time Operational Deployment:** Packaging the trained models and interactive dashboard into a municipal early-warning environmental decision-support system.

---

## Group Contributions

| Team Member | Academic Program | Key Technical Responsibilities | Owned Report Chapters |
| :--- | :--- | :--- | :---: |
| **Member 1 (Data Lead)** | M.Tech (ML) | CPCB 15-min data streaming, station cleaning, NAQI formula engine, multi-source merging, consolidated Excel workbook, 10% masking imputation experiment. | Chapters 1 & 2 |
| **Member 2 (Analysis Lead)** | M.Sc. (IT) | Univariate and multivariate exploratory visualizations (Figures 1–10), feature engineering, mutual information and Lasso selection, Problem 1 PCA and K-Means clustering. | Chapters 3 & 4, §5.1 |
| **Member 3 (Modelling Lead)**| B.Tech (ICT) | Problem 2 severe classification pipelines, persistence baseline benchmarking, spatial generalization testing, Problem 3 econometric panel regression with clustered SEs, interactive web dashboard. | §5.2 & §5.3, Chapter 6 |

---

## Short Bios of Team Members

1. **Member 1** is a graduate student specializing in machine learning and data engineering at DA-IICT. His research interests include distributed data processing, environmental sensing, and automated data quality pipelines.
2. **Member 2** is a graduate student in Information Technology at DA-IICT with expertise in statistical data analysis, exploratory visualization, and unsupervised clustering architectures.
3. **Member 3** is an undergraduate student in Information and Communication Technology at DA-IICT with deep interests in applied econometrics, predictive time-series modeling, and interactive web application design.

---

## References

1. Central Pollution Control Board (CPCB), *National Air Quality Index (NAQI): Report of the Technical Working Group*, Ministry of Environment, Forest and Climate Change, Government of India, New Delhi, 2014.
2. World Health Organization (WHO), *WHO Global Air Quality Guidelines: Particulate Matter (PM2.5 and PM10), Ozone, Nitrogen Dioxide, Sulfur Dioxide and Carbon Monoxide*, World Health Organization, Geneva, 2021.
3. Guttikunda, S. K., Goel, R., & Pant, P., "Nature of air pollution, emission sources, and management in the Indian cities," *Atmospheric Environment*, vol. 95, pp. 501–510, 2014.
4. Open-Meteo Historical Weather API, *ECMWF ERA5 Reanalysis Weather Archives*, https://open-meteo.com, accessed November 2026.
5. NASA FIRMS, *Fire Information for Resource Management System (VIIRS and MODIS Active Fire Detections)*, National Aeronautics and Space Administration, https://firms.modaps.eosdis.nasa.gov, accessed November 2026.
6. Pedregosa, F., et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
7. Seabold, S., & Perktold, J., "Statsmodels: Econometric and statistical modeling with python," *Proceedings of the 9th Python in Science Conference*, 2010.
8. Lundberg, S. M., & Lee, S. I., "A unified approach to interpreting model predictions," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.
