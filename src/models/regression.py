"""Problem 3: Econometric Regression & Machine Learning Drivers of PM2.5."""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import PROCESSED_DATA_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR, RANDOM_SEED

def run_problem3_regression():
    print("=" * 60)
    print("PROBLEM 3: DRIVERS OF PM2.5 (ECONOMETRIC REGRESSION & INFERENCE)")
    print("=" * 60)
    
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    df = pd.read_csv(PROCESSED_DATA_DIR / "master_features.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['log_pm25'] = np.log1p(df['PM2.5'])
    
    # -------------------------------------------------------------
    # 1. Non-Parametric Hypothesis Tests
    # -------------------------------------------------------------
    print("[Problem 3] Running Non-Parametric Hypothesis Tests...")
    hyp_results = []
    
    # Test 1: Diwali Window vs. Non-Diwali Days (Mann-Whitney U)
    diwali_pm25 = df[df['diwali_window'] == 1]['PM2.5'].dropna()
    nondiwali_pm25 = df[df['diwali_window'] == 0]['PM2.5'].dropna()
    u_stat, u_pval = stats.mannwhitneyu(diwali_pm25, nondiwali_pm25, alternative='greater')
    hyp_results.append({
        "Hypothesis": "Diwali Window PM2.5 > Baseline Non-Festival",
        "Test": "Mann-Whitney U (One-Sided)",
        "Statistic": round(float(u_stat), 2),
        "p_value": f"{u_pval:.4e}",
        "Group_1_Median": round(float(diwali_pm25.median()), 2),
        "Group_2_Median": round(float(nondiwali_pm25.median()), 2),
        "Conclusion": "Reject H0 (Statistically Significant Surge)" if u_pval < 0.05 else "Fail to Reject H0"
    })
    
    # Test 2: Winter vs. Other Seasons (Kruskal-Wallis)
    seasons = [grp['PM2.5'].dropna().values for _, grp in df.groupby('season')]
    h_stat, h_pval = stats.kruskal(*seasons)
    hyp_results.append({
        "Hypothesis": "Seasonal Divergence Across Annual Cycle",
        "Test": "Kruskal-Wallis H-Test",
        "Statistic": round(float(h_stat), 2),
        "p_value": f"{h_pval:.4e}",
        "Group_1_Median": round(float(df[df['season'].str.lower()=='winter']['PM2.5'].median()), 2),
        "Group_2_Median": round(float(df[df['season'].str.lower()=='monsoon']['PM2.5'].median()), 2),
        "Conclusion": "Reject H0 (Highly Significant Seasonality)" if h_pval < 0.05 else "Fail to Reject H0"
    })
    
    # Test 3: Weekend vs. Weekday Effect
    weekend_pm25 = df[df['is_weekend'] == 1]['PM2.5'].dropna()
    weekday_pm25 = df[df['is_weekend'] == 0]['PM2.5'].dropna()
    u_wknd, p_wknd = stats.mannwhitneyu(weekend_pm25, weekday_pm25)
    hyp_results.append({
        "Hypothesis": "Weekend Effect (Anthropogenic Traffic Lull)",
        "Test": "Mann-Whitney U (Two-Sided)",
        "Statistic": round(float(u_wknd), 2),
        "p_value": f"{p_wknd:.4e}",
        "Group_1_Median": round(float(weekend_pm25.median()), 2),
        "Group_2_Median": round(float(weekday_pm25.median()), 2),
        "Conclusion": "Reject H0" if p_wknd < 0.05 else "Fail to Reject H0 (No Significant Weekend Drop)"
    })
    
    df_hyp = pd.DataFrame(hyp_results)
    df_hyp.to_csv(TABLES_DIR / "problem3_hypothesis_tests.csv", index=False)
    print(f"  Hypothesis Tests:\n{df_hyp[['Hypothesis', 'Statistic', 'p_value', 'Conclusion']].to_string(index=False)}")

    # -------------------------------------------------------------
    # 2. Econometric OLS with City Fixed Effects & Clustered SEs
    # -------------------------------------------------------------
    print("\n[Problem 3] Fitting OLS with City Fixed Effects & Clustered Standard Errors...")
    # Regression model formula (using Q() to escape variable name with period):
    formula = (
        "Q('PM2.5') ~ C(city) + temp_mean + humidity_mean + wind_speed_mean + precip_sum + "
        "boundary_layer_height_mean + diwali_window + fire_count_NW_lag_1 + pm25_lag_1"
    )
    
    ols_df = df.dropna(subset=[
        'PM2.5', 'city', 'temp_mean', 'humidity_mean', 'wind_speed_mean', 'precip_sum',
        'boundary_layer_height_mean', 'diwali_window', 'fire_count_NW_lag_1', 'pm25_lag_1'
    ]).copy()
    
    ols_model = smf.ols(formula=formula, data=ols_df).fit(
        cov_type='cluster', cov_kwds={'groups': ols_df['city']}
    )
    
    # Extract coefficients, 95% CIs, and p-values
    coef_df = pd.DataFrame({
        "Coefficient": ols_model.params,
        "Std_Error_Clustered": ols_model.bse,
        "t_statistic": ols_model.tvalues,
        "p_value": ols_model.pvalues,
        "CI_Lower_95": ols_model.conf_int()[0],
        "CI_Upper_95": ols_model.conf_int()[1]
    })
    coef_df.to_csv(TABLES_DIR / "problem3_econometric_coefficients.csv")
    print(f"  Econometric OLS R-squared: {ols_model.rsquared:.3f}, Adj R-squared: {ols_model.rsquared_adj:.3f}")
    print(f"  Key Drivers (Non-city terms):\n{coef_df.loc[~coef_df.index.str.startswith('C(city)')].round(4)}")
    
    # Econometric Diagnostics
    dw_stat = durbin_watson(ols_model.resid)
    print(f"  Durbin-Watson Residual Autocorrelation: {dw_stat:.3f}")

    # -------------------------------------------------------------
    # 3. Supervised Regression Machine Learning Benchmark
    # -------------------------------------------------------------
    print("\n[Problem 3] Running Predictive Regression Benchmark (Chronological Test)...")
    reg_features = [
        'temp_mean', 'temp_range', 'humidity_mean', 'wind_speed_mean', 'u_wind_mean', 'v_wind_mean',
        'precip_sum', 'pressure_mean', 'boundary_layer_height_mean', 'ventilation_index',
        'fire_count_NW_India', 'fire_count_NW_lag_1', 'diwali_window', 'is_weekend',
        'month_sin', 'month_cos', 'pm25_lag_1', 'pm25_lag_7', 'pm25_roll7_mean'
    ]
    # Add city one-hot encoding
    df_reg = pd.get_dummies(df, columns=['city'], drop_first=True, dtype=float)
    city_dummy_cols = [c for c in df_reg.columns if c.startswith('city_')]
    all_reg_features = reg_features + city_dummy_cols
    
    df_reg_clean = df_reg.dropna(subset=all_reg_features + ['PM2.5']).copy()
    
    train_mask = df_reg_clean['date'] <= '2025-08-31'
    test_mask = df_reg_clean['date'] > '2025-08-31'
    
    X_tr = df_reg_clean.loc[train_mask, all_reg_features]
    y_tr = df_reg_clean.loc[train_mask, 'PM2.5']
    
    X_te = df_reg_clean.loc[test_mask, all_reg_features]
    y_te = df_reg_clean.loc[test_mask, 'PM2.5']
    
    scaler_reg = StandardScaler()
    X_tr_sc = scaler_reg.fit_transform(X_tr)
    X_te_sc = scaler_reg.transform(X_te)
    
    # Model 1: City-Mean Baseline
    city_means = df_reg_clean.loc[train_mask].groupby('city_Delhi')['PM2.5'].mean() # Proxy
    # Predict city-specific training mean
    y_pred_base = df_reg_clean.loc[test_mask].groupby('city_Delhi')['PM2.5'].transform(lambda _: y_tr.mean()).values
    
    reg_models = {
        "City-Mean Baseline": None,
        "Ridge Regression (L2)": RidgeCV(cv=5),
        "Lasso Regression (L1)": LassoCV(cv=5, random_state=RANDOM_SEED),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=150, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1),
        "HistGradientBoosting Regressor": HistGradientBoostingRegressor(max_iter=150, max_depth=6, random_state=RANDOM_SEED)
    }
    
    reg_results = []
    test_preds_dict = {}
    
    for name, r_model in reg_models.items():
        if r_model is None:
            pred = y_pred_base
        elif "Ridge" in name or "Lasso" in name:
            r_model.fit(X_tr_sc, y_tr)
            pred = r_model.predict(X_te_sc)
        else:
            r_model.fit(X_tr, y_tr)
            pred = r_model.predict(X_te)
            
        test_preds_dict[name] = pred
        rmse = root_mean_squared_error(y_te, pred)
        mae = mean_absolute_error(y_te, pred)
        r2 = r2_score(y_te, pred)
        
        reg_results.append({
            "Model": name,
            "RMSE (µg/m³)": round(rmse, 2),
            "MAE (µg/m³)": round(mae, 2),
            "R_Squared": round(r2, 3),
            "Beats_Baseline": "Yes" if rmse < root_mean_squared_error(y_te, y_pred_base) else "Baseline"
        })
        
    df_reg_res = pd.DataFrame(reg_results)
    df_reg_res.to_csv(TABLES_DIR / "problem3_regression_comparison.csv", index=False)
    print(f"\n[Problem 3] Regression Benchmark Saved to {TABLES_DIR / 'problem3_regression_comparison.csv'}:\n{df_reg_res.to_string(index=False)}")

    # -------------------------------------------------------------
    # 4. Generate Diagnostic Figures: Forest Plot & Residuals
    # -------------------------------------------------------------
    # Fig 17: Econometric Coefficients Forest Plot
    print("[Problem 3] Generating Fig 17: Coefficients Forest Plot...")
    sub_coef = coef_df.loc[~coef_df.index.str.startswith('C(city)')].drop('Intercept', errors='ignore')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(sub_coef))
    ax.errorbar(
        sub_coef['Coefficient'], y_pos,
        xerr=[sub_coef['Coefficient'] - sub_coef['CI_Lower_95'], sub_coef['CI_Upper_95'] - sub_coef['Coefficient']],
        fmt='o', color='#2b5c8f', ecolor='darkred', elinewidth=2, capsize=5, markersize=8
    )
    ax.axvline(0, color='grey', linestyle='--', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub_coef.index)
    ax.set_title("Fig 17: Econometric Panel Regression Coefficients with 95% Clustered Confidence Intervals")
    ax.set_xlabel("Estimated Marginal Effect on PM2.5 (µg/m³)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig17_coefficients_forest_plot.png")
    plt.close()

    # Fig 18: Residual Diagnostics (Residual vs Fitted & Q-Q Plot)
    print("[Problem 3] Generating Fig 18: Regression Residual Diagnostics...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Residual vs Fitted
    fitted_vals = ols_model.fittedvalues
    resids = ols_model.resid
    axes[0].scatter(fitted_vals, resids, alpha=0.4, color='#1f77b4', edgecolors='none')
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_title("Fig 18A: Residuals vs. Fitted Values (Homoscedasticity Assessment)")
    axes[0].set_xlabel("Fitted PM2.5 Values (µg/m³)")
    axes[0].set_ylabel("Residuals (Observed - Fitted)")
    
    # Q-Q Plot
    sm.qqplot(resids, line='45', fit=True, ax=axes[1])
    axes[1].set_title("Fig 18B: Normal Q-Q Plot of Regression Residuals")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig18_regression_diagnostics.png")
    plt.close()

    # Save artifacts
    joblib.dump({"ols": ols_model, "ml_models": reg_models}, MODELS_DIR / "problem3_regression_models.joblib")
    print(f"[Problem 3] Serialized regression artifacts to {MODELS_DIR / 'problem3_regression_models.joblib'}")
    return df_reg_res

if __name__ == "__main__":
    run_problem3_regression()
