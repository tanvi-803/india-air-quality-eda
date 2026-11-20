"""Missing Data Imputation & Controlled 10% Masking Benchmark Experiment."""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import INTERIM_DATA_DIR, TABLES_DIR, RANDOM_SEED

def run_imputation_experiment():
    print("=" * 60)
    print("RUNNING 10% MASKING BENCHMARK EXPERIMENT")
    print("=" * 60)
    
    os.makedirs(TABLES_DIR, exist_ok=True)
    aq_path = INTERIM_DATA_DIR / "aq_daily.csv"
    df_aq = pd.read_csv(aq_path)
    
    # Isolate known valid PM2.5 observations
    valid_idx = df_aq[df_aq['PM2.5'].notnull()].index.values
    np.random.seed(RANDOM_SEED)
    mask_size = int(0.10 * len(valid_idx))
    masked_indices = np.random.choice(valid_idx, size=mask_size, replace=False)
    
    y_true = df_aq.loc[masked_indices, 'PM2.5'].values
    
    # Create masked copy
    df_masked = df_aq.copy()
    df_masked.loc[masked_indices, 'PM2.5'] = np.nan
    
    results = []
    
    # -------------------------------------------------------------
    # 1. Forward Fill (grouped by city, max limit = 7 days)
    # -------------------------------------------------------------
    df_ffill = df_masked.groupby('city')['PM2.5'].transform(lambda s: s.ffill(limit=7))
    pred_ffill = df_ffill.loc[masked_indices].values
    valid_eval = ~np.isnan(pred_ffill)
    rmse_ffill = root_mean_squared_error(y_true[valid_eval], pred_ffill[valid_eval])
    mae_ffill = mean_absolute_error(y_true[valid_eval], pred_ffill[valid_eval])
    results.append({
        "Method": "Forward Fill (7-day cap)",
        "Type": "Univariate Autoregressive",
        "Evaluated_Points": int(valid_eval.sum()),
        "RMSE": round(rmse_ffill, 2),
        "MAE": round(mae_ffill, 2)
    })
    
    # -------------------------------------------------------------
    # 2. Linear Interpolation (grouped by city, max limit = 7 days)
    # -------------------------------------------------------------
    df_linear = df_masked.groupby('city')['PM2.5'].transform(lambda s: s.interpolate(method='linear', limit=7))
    pred_linear = df_linear.loc[masked_indices].values
    valid_eval = ~np.isnan(pred_linear)
    rmse_linear = root_mean_squared_error(y_true[valid_eval], pred_linear[valid_eval])
    mae_linear = mean_absolute_error(y_true[valid_eval], pred_linear[valid_eval])
    results.append({
        "Method": "Linear Interpolation (7-day cap)",
        "Type": "Univariate Temporal",
        "Evaluated_Points": int(valid_eval.sum()),
        "RMSE": round(rmse_linear, 2),
        "MAE": round(mae_linear, 2)
    })
    
    # -------------------------------------------------------------
    # 3. K-Nearest Neighbors (KNN Imputer, k=5)
    # -------------------------------------------------------------
    feature_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']
    imputer_knn = KNNImputer(n_neighbors=5, weights='distance')
    imputed_knn = imputer_knn.fit_transform(df_masked[feature_cols])
    pred_knn = imputed_knn[masked_indices, 0]
    rmse_knn = root_mean_squared_error(y_true, pred_knn)
    mae_knn = mean_absolute_error(y_true, pred_knn)
    results.append({
        "Method": "K-Nearest Neighbors (KNN, k=5)",
        "Type": "Multivariate Stoichiometric",
        "Evaluated_Points": len(y_true),
        "RMSE": round(rmse_knn, 2),
        "MAE": round(mae_knn, 2)
    })
    
    # -------------------------------------------------------------
    # 4. Iterative Imputer / MICE (BayesianRidge)
    # -------------------------------------------------------------
    imputer_mice = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=10,
        random_state=RANDOM_SEED,
        initial_strategy='median'
    )
    imputed_mice = imputer_mice.fit_transform(df_masked[feature_cols])
    pred_mice = imputed_mice[masked_indices, 0]
    rmse_mice = root_mean_squared_error(y_true, pred_mice)
    mae_mice = mean_absolute_error(y_true, pred_mice)
    results.append({
        "Method": "Iterative Imputer (MICE, BayesianRidge)",
        "Type": "Multivariate Regression",
        "Evaluated_Points": len(y_true),
        "RMSE": round(rmse_mice, 2),
        "MAE": round(mae_mice, 2)
    })
    
    df_res = pd.DataFrame(results)
    out_table = TABLES_DIR / "imputation_masking_experiment.csv"
    df_res.to_csv(out_table, index=False)
    print(f"\n[Masking Experiment] Comparison Table Saved to {out_table}:\n{df_res.to_string(index=False)}")
    
    # Apply imputation to PREDICTORS only. The masking table above still compares all methods on PM2.5.
    best_method = df_res.sort_values(by="RMSE").iloc[0]["Method"]
    print(f"\n[Imputation] Selected winning strategy: {best_method}")
    
    df_clean_imputed = df_aq.copy()
    df_clean_imputed['date'] = pd.to_datetime(df_clean_imputed['date'])
    # Rules (blueprint Stage C): never impute the target (PM2.5, hence AQI) and never fill
    # gaps longer than 7 days. Predictor pollutants use a CAUSAL forward-fill so the value
    # on day t never depends on later observations (linear interpolation would peek ahead).
    predictors = ['PM10', 'NO2', 'SO2', 'CO', 'O3', 'NH3']
    for col in predictors:
        df_clean_imputed[col] = df_clean_imputed.groupby('city')[col].transform(lambda s: s.ffill(limit=7))
    # Step 2: remaining short-gap cells via KNN fitted on the TRAINING period only
    # (Jan-Aug 2025); PM2.5 is used as a neighbour feature but never written back.
    knn_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']
    train_rows = df_clean_imputed['date'] <= '2025-08-31'
    knn = KNNImputer(n_neighbors=5).fit(df_clean_imputed.loc[train_rows, knn_cols])
    filled = pd.DataFrame(knn.transform(df_clean_imputed[knn_cols]), columns=knn_cols, index=df_clean_imputed.index)
    for col in ['PM10', 'NO2', 'SO2', 'CO']:
        df_clean_imputed[col] = df_clean_imputed[col].fillna(filled[col])
    print(f"[Imputation] PM2.5 left un-imputed: {df_clean_imputed['PM2.5'].isna().sum()} city-days "
          f"({df_clean_imputed['PM2.5'].isna().mean()*100:.1f}%)")
    df_clean_imputed['date'] = df_clean_imputed['date'].dt.strftime('%Y-%m-%d')
    
    out_imputed_csv = INTERIM_DATA_DIR / "aq_daily_imputed.csv"
    df_clean_imputed.to_csv(out_imputed_csv, index=False)
    print(f"[Imputation] Cleaned & imputed air quality dataset saved to {out_imputed_csv}")
    return df_res, df_clean_imputed

if __name__ == "__main__":
    run_imputation_experiment()
