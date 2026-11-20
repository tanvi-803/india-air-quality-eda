"""Problem 1: Unsupervised Learning - City & Station Pollution Profiles."""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import PROCESSED_DATA_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR, RANDOM_SEED

NUM_COLS = [
    "pm25_mean", "pm25_p95", "pm25_cv", "pm10_mean", "pm10_p95",
    "no2_mean", "so2_mean", "co_mean", "o3_mean",
    "pm25_pm10_ratio", "winter_monsoon_ratio", "severe_day_share"
]


def _city_feature_rows(df, severe_thr):
    """One summary-feature row per city (reused for bootstrap / half-year stability)."""
    city_features = []
    for city, grp in df.groupby('city'):
        pm25 = grp['PM2.5'].dropna()
        pm10 = grp['PM10'].dropna()
        winter_pm25 = grp[grp['season'].str.lower() == 'winter']['PM2.5'].mean()
        monsoon_pm25 = grp[grp['season'].str.lower() == 'monsoon']['PM2.5'].mean()
        ratio = (winter_pm25 / monsoon_pm25) if monsoon_pm25 and monsoon_pm25 > 0 else np.nan
        city_features.append({
            "city": city,
            "region": grp['region'].iloc[0],
            "latitude": grp['latitude'].iloc[0],
            "longitude": grp['longitude'].iloc[0],
            "pm25_mean": pm25.mean(),
            "pm25_p95": pm25.quantile(0.95),
            "pm25_cv": pm25.std() / pm25.mean() if pm25.mean() > 0 else 0,
            "pm10_mean": pm10.mean(),
            "pm10_p95": pm10.quantile(0.95),
            "no2_mean": grp['NO2'].mean(),
            "so2_mean": grp['SO2'].mean(),
            "co_mean": grp['CO'].mean(),
            "o3_mean": grp['O3'].mean(),
            "pm25_pm10_ratio": (pm25.mean() / pm10.mean()) if pm10.mean() > 0 else 0.5,
            "winter_monsoon_ratio": ratio,
            "severe_day_share": (grp['PM2.5'] >= severe_thr).mean(),
        })
    return city_features


def _cluster_labels(city_feats, k, seed):
    """Standardise a city feature table and return K-Means labels."""
    X = city_feats[NUM_COLS].fillna(city_feats[NUM_COLS].median()).values
    return KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(StandardScaler().fit_transform(X))


def _stability(df, severe_thr, base_labels, k, n_boot=200):
    """Bootstrap days within each city and compare to the full-year partition (ARI),
    plus Jan-Jun vs Jul-Dec halves (one-year stand-in for the blueprint's per-year check)."""
    rng = np.random.default_rng(RANDOM_SEED)
    cities = sorted(df['city'].unique())
    aris = []
    for _ in range(n_boot):
        boot = pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1e9)))
                          for _, g in df.groupby('city')])
        f = pd.DataFrame(_city_feature_rows(boot, severe_thr)).set_index("city").loc[cities]
        aris.append(adjusted_rand_score(base_labels, _cluster_labels(f, k, RANDOM_SEED)))
    rows = [{"Check": f"Bootstrap days within city (n={n_boot})", "Mean_ARI": round(float(np.mean(aris)), 3),
             "P5_ARI": round(float(np.percentile(aris, 5)), 3)}]
    for lab, m in (("Jan-Jun only", df['date'].dt.month <= 6), ("Jul-Dec only", df['date'].dt.month > 6)):
        f = pd.DataFrame(_city_feature_rows(df[m], severe_thr)).set_index("city").loc[cities]
        ari = adjusted_rand_score(base_labels, _cluster_labels(f, k, RANDOM_SEED))
        rows.append({"Check": lab, "Mean_ARI": round(float(ari), 3), "P5_ARI": np.nan})
    return pd.DataFrame(rows)


def run_problem1_clustering():
    print("=" * 60)
    print("PROBLEM 1: UNSUPERVISED LEARNING - CITY POLLUTION PROFILES")
    print("=" * 60)
    
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    df = pd.read_csv(PROCESSED_DATA_DIR / "master.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    # -------------------------------------------------------------
    # 1. City-Level Feature Vector Extraction
    # -------------------------------------------------------------
    severe_thr = df['PM2.5'].quantile(0.88)
    city_features = _city_feature_rows(df, severe_thr)
    df_city_prof = pd.DataFrame(city_features).set_index("city")
    
    num_cols = NUM_COLS
    
    # Impute any missing city-level summary features with national median (e.g. Surat O3)
    df_city_prof[num_cols] = df_city_prof[num_cols].fillna(df_city_prof[num_cols].median())
    
    X = df_city_prof[num_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # -------------------------------------------------------------
    # 2. Principal Component Analysis (PCA)
    # -------------------------------------------------------------
    print("[Problem 1] Running Principal Component Analysis (PCA)...")
    pca = PCA(n_components=min(len(num_cols), len(df_city_prof)))
    X_pca = pca.fit_transform(X_scaled)
    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)
    
    print(f"  PC1 Explained Variance: {evr[0]*100:.2f}%")
    print(f"  PC2 Explained Variance: {evr[1]*100:.2f}%")
    print(f"  Cumulative PC1+PC2: {cum_evr[1]*100:.2f}%")
    
    # -------------------------------------------------------------
    # 3. Cluster Evaluation: K-Means & Silhouette Analysis
    # -------------------------------------------------------------
    print("[Problem 1] Evaluating K-Means Clusters (k=2 to 5)...")
    cluster_metrics = []
    for k in range(2, 6):
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20).fit(X_scaled)
        sil = silhouette_score(X_scaled, km.labels_)
        db = davies_bouldin_score(X_scaled, km.labels_)
        cluster_metrics.append({
            "k": k,
            "Inertia": round(km.inertia_, 2),
            "Silhouette_Score": round(sil, 3),
            "Davies_Bouldin_Index": round(db, 3)
        })
    df_metrics = pd.DataFrame(cluster_metrics)
    df_metrics.to_csv(TABLES_DIR / "problem1_cluster_metrics.csv", index=False)
    print(f"  Cluster Metrics:\n{df_metrics.to_string(index=False)}")
    
    # k selection. Silhouette alone picks k=2, which only isolates Delhi (one outlier city
    # versus eleven others). k=3 is the smallest k that also splits the remaining cities
    # into two interpretable groups; k=4 only adds a second singleton (Surat).
    optimal_k = 3
    km_final = KMeans(n_clusters=optimal_k, random_state=RANDOM_SEED, n_init=30).fit(X_scaled)
    df_city_prof["Cluster_KMeans"] = km_final.labels_
    
    # Agglomerative Hierarchical Clustering (Ward's Linkage). Its integer labels are arbitrary,
    # so re-map them onto the K-Means labels by majority overlap before comparing.
    agg = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward').fit(X_scaled)
    ari_km_hc = adjusted_rand_score(km_final.labels_, agg.labels_)
    relabel = {h: pd.Series(km_final.labels_[agg.labels_ == h]).mode().iloc[0] for h in np.unique(agg.labels_)}
    df_city_prof["Cluster_Hierarchical"] = pd.Series(agg.labels_, index=df_city_prof.index).map(relabel)
    print(f"  K-Means vs Hierarchical agreement (ARI): {ari_km_hc:.2f}")
    
    # Name clusters from their RANK on mean PM2.5 (data-driven, not absolute cut-offs)
    order = df_city_prof.groupby("Cluster_KMeans")["pm25_mean"].mean().sort_values(ascending=False).index.tolist()
    tier_names = ["Extreme Smog Outlier", "Winter-Smog Belt & Semi-Arid", "Cleaner Coastal & Southern"]
    cluster_names = {}
    for rank, cl in enumerate(order):
        members = ", ".join(df_city_prof.index[df_city_prof["Cluster_KMeans"] == cl])
        sub = df_city_prof[df_city_prof["Cluster_KMeans"] == cl]
        cluster_names[cl] = tier_names[min(rank, len(tier_names) - 1)]
        print(f"  Cluster {cl} -> {cluster_names[cl]} | mean PM2.5 {sub['pm25_mean'].mean():.1f}, "
              f"winter/monsoon {sub['winter_monsoon_ratio'].mean():.2f}, severe share {sub['severe_day_share'].mean():.2f}: {members}")
    df_city_prof["Cluster_Profile_Name"] = df_city_prof["Cluster_KMeans"].map(cluster_names)
    
    # Stability of the partition
    df_stab = _stability(df, severe_thr, km_final.labels_, optimal_k)
    df_stab.to_csv(TABLES_DIR / "problem1_stability.csv", index=False)
    print(f"  Stability (ARI vs full-year partition):\n{df_stab.to_string(index=False)}")
    
    # Save assignments
    df_city_prof.to_csv(TABLES_DIR / "problem1_city_cluster_assignments.csv")
    print(f"\n[Problem 1] City Cluster Assignments Saved to {TABLES_DIR / 'problem1_city_cluster_assignments.csv'}")
    
    # -------------------------------------------------------------
    # 4. Generate Figures: PCA Biplot & Dendrogram
    # -------------------------------------------------------------
    print("[Problem 1] Generating Fig 11: PCA Biplot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {0: '#d95f02', 1: '#1b9e77', 2: '#7570b3'}
    for cl in range(optimal_k):
        c_mask = df_city_prof["Cluster_KMeans"] == cl
        c_name = cluster_names[cl]
        ax.scatter(
            X_pca[c_mask, 0], X_pca[c_mask, 1],
            c=colors[cl], s=180, label=f"Cluster {cl}: {c_name}", edgecolors='black', alpha=0.9
        )
        
    for i, city in enumerate(df_city_prof.index):
        ax.annotate(
            city, (X_pca[i, 0], X_pca[i, 1]),
            fontsize=11, fontweight='bold', ha='right', va='bottom',
            xytext=(5, 5), textcoords='offset points'
        )
        
    # Overlay PCA Loadings Vectors (Biplot)
    loadings = pca.components_[:2, :] * 3.5  # Scale for visual balance
    for j, col_name in enumerate(num_cols):
        ax.arrow(0, 0, loadings[0, j], loadings[1, j], color='darkred', alpha=0.6, head_width=0.1)
        ax.text(loadings[0, j]*1.12, loadings[1, j]*1.12, col_name, color='darkred', fontsize=9, ha='center', va='center')
        
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.set_title(f"Fig 11: PCA Biplot of City Pollution Profiles (PC1: {evr[0]*100:.1f}%, PC2: {evr[1]*100:.1f}%)")
    top_load = lambda i: ", ".join(pd.Series(pca.components_[i], index=num_cols).abs().sort_values(ascending=False).index[:3])
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% variance; top loadings: {top_load(0)})")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% variance; top loadings: {top_load(1)})")
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig11_pca_biplot_clusters.png")
    plt.close()

    # Fig 12: Hierarchical Clustering Dendrogram
    print("[Problem 1] Generating Fig 12: Hierarchical Dendrogram...")
    fig, ax = plt.subplots(figsize=(11, 6))
    Z = linkage(X_scaled, method='ward')
    dendrogram(Z, labels=df_city_prof.index.tolist(), ax=ax, leaf_rotation=45, leaf_font_size=11)
    ax.set_title("Fig 12: Hierarchical Clustering Dendrogram (Ward Linkage, Euclidean Distance)")
    ax.set_ylabel("Ward Linkage Distance")
    cut = (Z[-(optimal_k - 1), 2] + Z[-optimal_k, 2]) / 2
    ax.axhline(y=cut, color='red', linestyle='--', label=f"Cut for k={optimal_k} (distance {cut:.2f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig12_hierarchical_dendrogram.png")
    plt.close()

    # Fig 13: Cluster Multi-Pollutant Radar / Bar Profiles
    print("[Problem 1] Generating Fig 13: Cluster Characteristic Profiles...")
    profile_summary = df_city_prof.groupby("Cluster_Profile_Name")[[
        "pm25_mean", "pm10_mean", "no2_mean", "so2_mean", "co_mean", "winter_monsoon_ratio"
    ]].mean()
    
    fig, ax = plt.subplots(figsize=(13, 6))
    profile_norm = (profile_summary - profile_summary.min()) / (profile_summary.max() - profile_summary.min() + 1e-6)
    profile_norm.plot(kind='bar', ax=ax, colormap='tab10', width=0.7)
    ax.set_title("Fig 13: Normalized Multi-Pollutant & Seasonal Signatures Across the 3 City Clusters")
    ax.set_ylabel("Normalized Feature Score [0, 1]")
    ax.set_xlabel("City Pollution Profile Cluster")
    plt.xticks(rotation=15, ha='right')
    ax.legend(title="Indicators", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig13_cluster_radar_profiles.png")
    plt.close()

    # Save trained artifacts
    joblib.dump({"scaler": scaler, "pca": pca, "kmeans": km_final, "features": num_cols}, MODELS_DIR / "problem1_pca_kmeans.joblib")
    print(f"[Problem 1] Model pipeline artifacts serialized to {MODELS_DIR / 'problem1_pca_kmeans.joblib'}")
    return df_city_prof

if __name__ == "__main__":
    run_problem1_clustering()
