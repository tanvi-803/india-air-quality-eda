"""Problem 2: Severe Pollution Episode Prediction - Supervised Classification."""

import json
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix, roc_curve, average_precision_score
)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import PROCESSED_DATA_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR, RANDOM_SEED

def run_problem2_classification():
    print("=" * 60)
    print("PROBLEM 2: SEVERE POLLUTION EPISODE PREDICTION (CLASSIFICATION)")
    print("=" * 60)
    
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    df = pd.read_csv(PROCESSED_DATA_DIR / "master_features.csv")
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['city', 'date'], inplace=True)
    
    # Feature columns strictly available at prediction time day t
    feature_cols = [
        'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'pm25_to_pm10_ratio',
        'pm25_lag_1', 'pm25_lag_2', 'pm25_lag_3', 'pm25_lag_7',
        'pm25_roll3_mean', 'pm25_roll7_mean', 'pm25_diff_1',
        'temp_mean', 'temp_range', 'humidity_mean', 'pressure_mean',
        'wind_speed_mean', 'u_wind_mean', 'v_wind_mean', 'precip_sum',
        'boundary_layer_height_mean', 'ventilation_index',
        'fire_count_NW_India', 'fire_count_NW_lag_1', 'fire_count_NW_lag_2',
        'month_sin', 'month_cos', 'is_weekend', 'diwali_window'
    ]
    
    # Filter valid rows with targets
    df_clean = df.dropna(subset=feature_cols + ['severe_next_day']).copy()
    
    # -------------------------------------------------------------
    # 1. Strict Chronological Split
    # Train: Jan 1 - Aug 31, 2025
    # Val:   Sep 1 - Oct 20, 2025
    # Test:  Oct 21 - Dec 31, 2025 (Out-of-sample severe winter & harvest peak)
    # -------------------------------------------------------------
    train_mask = df_clean['date'] <= '2025-08-31'
    val_mask = (df_clean['date'] > '2025-08-31') & (df_clean['date'] <= '2025-10-20')
    test_mask = df_clean['date'] > '2025-10-20'
    
    X_train = df_clean.loc[train_mask, feature_cols]
    y_train = df_clean.loc[train_mask, 'severe_next_day'].astype(int)
    
    X_val = df_clean.loc[val_mask, feature_cols]
    y_val = df_clean.loc[val_mask, 'severe_next_day'].astype(int)
    
    X_test = df_clean.loc[test_mask, feature_cols]
    y_test = df_clean.loc[test_mask, 'severe_next_day'].astype(int)
    
    print(f"[Problem 2] Chronological Split:")
    print(f"  Train Set: {len(X_train)} samples, Severe count: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
    print(f"  Val Set:   {len(X_val)} samples, Severe count: {y_val.sum()} ({y_val.mean()*100:.2f}%)")
    print(f"  Test Set:  {len(X_test)} samples, Severe count: {y_test.sum()} ({y_test.mean()*100:.2f}%)")
    
    # Standardize features using training statistics only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # -------------------------------------------------------------
    # 2. Persistence Baseline (Tomorrow = Today)
    # -------------------------------------------------------------
    # Today's severe status under the same data-driven PM2.5 threshold as the target
    with open(TABLES_DIR / "severe_threshold.json") as fh:
        severe_thr = json.load(fh)["pm25_threshold"]
    print(f"[Problem 2] Severe definition: PM2.5 >= {severe_thr} ug/m3 (top 12% of city-days)")
    today_severe_test = (df_clean.loc[test_mask, 'PM2.5'] >= severe_thr).astype(int).values
    
    prec_persist = precision_score(y_test, today_severe_test, zero_division=0)
    rec_persist = recall_score(y_test, today_severe_test, zero_division=0)
    f1_persist = f1_score(y_test, today_severe_test, zero_division=0)
    
    # Persistence is a hard 0/1 forecast, so its ROC/PR-AUC are computed from those
    # binary predictions (previously hardcoded 0.5 and the class prevalence).
    roc_persist = roc_auc_score(y_test, today_severe_test)
    pr_persist = average_precision_score(y_test, today_severe_test)
    prevalence = y_test.mean()

    results = [{
        "Model": "Persistence Baseline (Today=Tomorrow)",
        "Type": "Naive Benchmark",
        "Threshold": "n/a (hard rule)",
        "Severe_Precision": round(prec_persist, 3),
        "Severe_Recall": round(rec_persist, 3),
        "Severe_F1": round(f1_persist, 3),
        "ROC_AUC": round(roc_persist, 3),
        "PR_AUC": round(pr_persist, 3),
        "Test_Prevalence": round(prevalence, 3),
        "Outperforms_Persistence": "Baseline"
    }]
    
    # -------------------------------------------------------------
    # 3. Model Training & Comparison
    # -------------------------------------------------------------
    models = {
        "Logistic Regression (Balanced)": LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=150, max_depth=8, class_weight='balanced',
            random_state=RANDOM_SEED, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150, max_depth=6, class_weight='balanced',
            random_state=RANDOM_SEED
        ),
        "Support Vector Classifier (RBF)": SVC(
            kernel='rbf', probability=True, class_weight='balanced',
            random_state=RANDOM_SEED
        )
    }
    
    trained_models = {}
    test_probs = {}
    test_preds = {}
    
    def _best_f1_threshold(y_true, probs):
        """Pick the probability cut-off that maximises F1 on the VALIDATION set only."""
        prec, rec, thr = precision_recall_curve(y_true, probs)
        f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
        return float(thr[int(np.argmax(f1s))]) if len(thr) else 0.5

    def _scaled(name, X_raw, X_scaled):
        return X_scaled if ("Logistic" in name or "Vector" in name) else X_raw

    for name, clf in models.items():
        print(f"[Problem 2] Training {name}...")
        # Scale for linear/SVM, use raw for trees
        clf.fit(_scaled(name, X_train, X_train_scaled), y_train)
        probs = clf.predict_proba(_scaled(name, X_test, X_test_scaled))[:, 1]
        val_probs = clf.predict_proba(_scaled(name, X_val, X_val_scaled))[:, 1]

        # Threshold tuned on validation (never on test) to avoid optimistic bias
        thr = _best_f1_threshold(y_val, val_probs) if y_val.sum() > 0 else 0.5
        preds = (probs >= thr).astype(int)

        trained_models[name] = clf
        test_probs[name] = probs
        test_preds[name] = preds

        # Metrics
        p = precision_score(y_test, preds, zero_division=0)
        r = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        roc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)

        results.append({
            "Model": name,
            "Type": "Supervised Learning",
            "Threshold": round(thr, 3),
            "Severe_Precision": round(p, 3),
            "Severe_Recall": round(r, 3),
            "Severe_F1": round(f1, 3),
            "ROC_AUC": round(roc, 3),
            "PR_AUC": round(pr_auc, 3),
            "Test_Prevalence": round(prevalence, 3),
            "Outperforms_Persistence": "Yes" if f1 > f1_persist else "No"
        })
        
    df_comparison = pd.DataFrame(results)
    df_comparison.to_csv(TABLES_DIR / "problem2_classification_comparison.csv", index=False)
    print(f"\n[Problem 2] Model Comparison Table Saved to {TABLES_DIR / 'problem2_classification_comparison.csv'}:\n{df_comparison.to_string(index=False)}")
    
    # -------------------------------------------------------------
    # 4. Generalization Test Across Unseen Cities
    # Train on 9 cities, test strictly on 3 completely unseen cities
    # (e.g. Test on Jaipur, Lucknow, Kolkata)
    # -------------------------------------------------------------
    print("\n[Problem 2] Running Cross-City Spatial Generalization Test...")
    unseen_test_cities = ['Jaipur', 'Lucknow', 'Kolkata']
    train_cities_mask = ~df_clean['city'].isin(unseen_test_cities)
    test_cities_mask = df_clean['city'].isin(unseen_test_cities)
    
    rf_spatial = RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)
    rf_spatial.fit(df_clean.loc[train_cities_mask, feature_cols], df_clean.loc[train_cities_mask, 'severe_next_day'])
    spatial_preds = rf_spatial.predict(df_clean.loc[test_cities_mask, feature_cols])
    spatial_probs = rf_spatial.predict_proba(df_clean.loc[test_cities_mask, feature_cols])[:, 1]
    
    spatial_f1 = f1_score(df_clean.loc[test_cities_mask, 'severe_next_day'], spatial_preds, zero_division=0)
    spatial_roc = roc_auc_score(df_clean.loc[test_cities_mask, 'severe_next_day'], spatial_probs)
    print(f"  Unseen Cities Test (Jaipur, Lucknow, Kolkata): Severe F1 = {spatial_f1:.3f}, ROC-AUC = {spatial_roc:.3f}")
    
    # -------------------------------------------------------------
    # 5. Visualizations: PR & ROC Curves, Confusion Matrix, Feature Importance
    # -------------------------------------------------------------
    # Fig 14: PR & ROC Curves
    print("[Problem 2] Generating Fig 14: ROC & PR Curves...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC Curves
    for name, probs in test_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_val = roc_auc_score(y_test, probs)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_val:.3f})", linewidth=2)
    axes[0].plot([0, 1], [0, 1], 'k--', label="Random Chance (AUC=0.50)")
    axes[0].set_title("Fig 14A: Receiver Operating Characteristic (ROC) Curves")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(loc='lower right')
    
    # PR Curves
    for name, probs in test_probs.items():
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_val = auc(rec, prec)
        axes[1].plot(rec, prec, label=f"{name} (PR-AUC={pr_val:.3f})", linewidth=2)
    axes[1].axhline(y_test.mean(), color='grey', linestyle='--', label=f"No-Skill Baseline ({y_test.mean():.2f})")
    axes[1].set_title("Fig 14B: Precision-Recall Curves (Severe Class Focus)")
    axes[1].set_xlabel("Recall (Severe Episode Capture)")
    axes[1].set_ylabel("Precision (Forecasting Accuracy)")
    axes[1].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig14_pr_roc_curves.png")
    plt.close()

    # Fig 15: Confusion Matrices
    print("[Problem 2] Generating Fig 15: Confusion Matrix Comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Persistence confusion matrix
    cm_persist = confusion_matrix(y_test, today_severe_test)
    sns.heatmap(cm_persist, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False,
                xticklabels=["Non-Severe", "Severe"], yticklabels=["Non-Severe", "Severe"])
    axes[0].set_title(f"Fig 15A: Persistence Baseline Confusion Matrix (F1: {f1_persist:.2f})")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")
    
    # Best ML model (Random Forest)
    best_ml_name = "Random Forest Classifier"
    cm_rf = confusion_matrix(y_test, test_preds[best_ml_name])
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens", ax=axes[1], cbar=False,
                xticklabels=["Non-Severe", "Severe"], yticklabels=["Non-Severe", "Severe"])
    rf_f1 = df_comparison.loc[df_comparison["Model"] == best_ml_name, "Severe_F1"].values[0]
    axes[1].set_title(f"Fig 15B: {best_ml_name} Confusion Matrix (F1: {rf_f1:.2f})")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig15_confusion_matrices.png")
    plt.close()

    # Fig 16: Feature Importance
    print("[Problem 2] Generating Fig 16: Permutation Importance (held-out test set)...")
    rf_model = trained_models["Random Forest Classifier"]
    perm = permutation_importance(
        rf_model, X_test, y_test, scoring="average_precision",
        n_repeats=20, random_state=RANDOM_SEED, n_jobs=-1
    )
    importances = pd.Series(perm.importances_mean, index=feature_cols)
    errors = pd.Series(perm.importances_std, index=feature_cols)
    top = importances.sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top.index, top.values, xerr=errors[top.index].values, color='#2b5c8f')
    ax.set_title("Fig 16: Permutation Importance (drop in PR-AUC, test set) - Random Forest")
    ax.set_xlabel("Mean decrease in PR-AUC when feature is shuffled")
    importances.sort_values(ascending=False).to_csv(TABLES_DIR / "problem2_permutation_importance.csv", header=["importance"])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig16_shap_feature_importance.png")
    plt.close()

    # Save trained artifacts
    joblib.dump({
        "scaler": scaler,
        "models": trained_models,
        "feature_cols": feature_cols
    }, MODELS_DIR / "problem2_classification_models.joblib")
    print(f"[Problem 2] Serialized models to {MODELS_DIR / 'problem2_classification_models.joblib'}")
    return df_comparison

if __name__ == "__main__":
    run_problem2_classification()
