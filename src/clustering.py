"""
Demand Segmentation & Clustering Module
========================================

Clusters parking lots and time periods by demand patterns:
- K-Means clustering with elbow method
- DBSCAN density-based clustering
- Cluster profiling and visualization
- Per-cluster pricing strategy recommendations

Upgrade 4: Demand Segmentation / Clustering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from typing import Dict, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────
# Feature Preparation for Clustering
# ─────────────────────────────────────────────────────────────────────

def prepare_lot_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-lot statistics to create lot demand profiles.

    Features per lot:
    - Mean/max/std of occupancy rate
    - Mean/max queue length
    - Fraction of special days
    - Mean traffic level
    - Capacity
    - Peak hour occupancy (8-10am, 5-7pm)
    - Weekend vs weekday occupancy ratio
    """
    lot_profiles = df.groupby("SystemCodeNumber").agg(
        occ_mean=("OccupancyRate", "mean"),
        occ_max=("OccupancyRate", "max"),
        occ_std=("OccupancyRate", "std"),
        queue_mean=("QueueLength", "mean"),
        queue_max=("QueueLength", "max"),
        special_day_frac=("IsSpecialDay", "mean"),
        traffic_mean=("Traffic", "mean"),
        capacity=("Capacity", "first"),
    ).reset_index()

    # Peak hour occupancy (8-10am and 5-7pm)
    if "Hour" in df.columns:
        peak_mask = df["Hour"].isin([8, 9, 10, 17, 18, 19])
        peak_occ = df[peak_mask].groupby("SystemCodeNumber")["OccupancyRate"].mean()
        offpeak_occ = df[~peak_mask].groupby("SystemCodeNumber")["OccupancyRate"].mean()
        lot_profiles = lot_profiles.merge(
            peak_occ.rename("peak_occ").reset_index(), on="SystemCodeNumber", how="left"
        )
        lot_profiles = lot_profiles.merge(
            offpeak_occ.rename("offpeak_occ").reset_index(), on="SystemCodeNumber", how="left"
        )
        lot_profiles["peak_ratio"] = (
            lot_profiles["peak_occ"] / lot_profiles["offpeak_occ"].replace(0, 1)
        )

    # Weekend vs weekday ratio
    if "IsWeekend" in df.columns:
        weekend = df[df["IsWeekend"] == 1].groupby("SystemCodeNumber")["OccupancyRate"].mean()
        weekday = df[df["IsWeekend"] == 0].groupby("SystemCodeNumber")["OccupancyRate"].mean()
        lot_profiles = lot_profiles.merge(
            weekend.rename("weekend_occ").reset_index(), on="SystemCodeNumber", how="left"
        )
        lot_profiles = lot_profiles.merge(
            weekday.rename("weekday_occ").reset_index(), on="SystemCodeNumber", how="left"
        )
        lot_profiles["weekend_ratio"] = (
            lot_profiles["weekend_occ"] / lot_profiles["weekday_occ"].replace(0, 1)
        )

    lot_profiles = lot_profiles.fillna(0)
    return lot_profiles


def prepare_time_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-time-slot statistics for temporal clustering.

    Groups data by (hour, day_of_week) to find demand patterns.
    """
    if "Hour" not in df.columns or "DayOfWeek" not in df.columns:
        raise ValueError("DataFrame must have 'Hour' and 'DayOfWeek' columns")

    time_profiles = df.groupby(["Hour", "DayOfWeek"]).agg(
        occ_mean=("OccupancyRate", "mean"),
        occ_std=("OccupancyRate", "std"),
        queue_mean=("QueueLength", "mean"),
        traffic_mean=("Traffic", "mean"),
        special_frac=("IsSpecialDay", "mean"),
    ).reset_index()

    time_profiles = time_profiles.fillna(0)
    return time_profiles


# ─────────────────────────────────────────────────────────────────────
# K-Means Clustering
# ─────────────────────────────────────────────────────────────────────

def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 10)) -> Dict:
    """
    Use the elbow method and silhouette scores to find optimal K.

    Returns inertia and silhouette scores for each K.
    """
    inertias = []
    silhouette_scores_list = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        sil = silhouette_score(X_scaled, labels) if k > 1 else 0
        silhouette_scores_list.append(sil)

    return {
        "k_range": list(k_range),
        "inertias": inertias,
        "silhouette_scores": silhouette_scores_list,
        "best_k": list(k_range)[np.argmax(silhouette_scores_list)],
    }


def run_kmeans(
    profiles: pd.DataFrame,
    feature_cols: list,
    n_clusters: int = 3,
) -> Tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    Run K-Means clustering on lot or time profiles.

    Parameters
    ----------
    profiles : pd.DataFrame
        Profile DataFrame with features.
    feature_cols : list
        Columns to use for clustering.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    tuple
        (profiles with cluster labels, fitted KMeans, fitted scaler)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(profiles[feature_cols])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    profiles = profiles.copy()
    profiles["Cluster"] = kmeans.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, profiles["Cluster"])
    print(f"   K-Means (k={n_clusters}) — Silhouette Score: {sil_score:.4f}")

    return profiles, kmeans, scaler


# ─────────────────────────────────────────────────────────────────────
# DBSCAN Clustering
# ─────────────────────────────────────────────────────────────────────

def run_dbscan(
    profiles: pd.DataFrame,
    feature_cols: list,
    eps: float = 1.0,
    min_samples: int = 2,
) -> Tuple[pd.DataFrame, DBSCAN, StandardScaler]:
    """
    Run DBSCAN density-based clustering.

    DBSCAN automatically determines the number of clusters and
    identifies noise points (label = -1).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(profiles[feature_cols])

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    profiles = profiles.copy()
    profiles["Cluster_DBSCAN"] = dbscan.fit_predict(X_scaled)

    n_clusters = len(set(profiles["Cluster_DBSCAN"])) - (1 if -1 in profiles["Cluster_DBSCAN"].values else 0)
    n_noise = (profiles["Cluster_DBSCAN"] == -1).sum()
    print(f"   DBSCAN — Clusters: {n_clusters}, Noise points: {n_noise}")

    return profiles, dbscan, scaler


# ─────────────────────────────────────────────────────────────────────
# Cluster Analysis & Profiling
# ─────────────────────────────────────────────────────────────────────

def profile_clusters(
    profiles: pd.DataFrame,
    feature_cols: list,
    cluster_col: str = "Cluster",
) -> pd.DataFrame:
    """
    Generate cluster profiles showing mean feature values per cluster.
    """
    cluster_profiles = profiles.groupby(cluster_col)[feature_cols].mean()
    cluster_profiles["count"] = profiles.groupby(cluster_col).size()
    return cluster_profiles


def get_cluster_labels(cluster_profiles: pd.DataFrame) -> Dict[int, str]:
    """
    Assign descriptive labels to clusters based on their profiles.

    Uses occupancy and queue characteristics to name clusters.
    """
    labels = {}
    for cluster_id in cluster_profiles.index:
        profile = cluster_profiles.loc[cluster_id]
        occ = profile.get("occ_mean", 0)
        queue = profile.get("queue_mean", 0)

        if occ > 0.6:
            demand_level = "High-Demand"
        elif occ > 0.3:
            demand_level = "Medium-Demand"
        else:
            demand_level = "Low-Demand"

        if queue > 5:
            queue_level = "High-Queue"
        elif queue > 2:
            queue_level = "Medium-Queue"
        else:
            queue_level = "Low-Queue"

        labels[cluster_id] = f"{demand_level}, {queue_level}"

    return labels


# ─────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────

def plot_elbow_silhouette(optimal_k_results: Dict, save_path: Optional[str] = None):
    """Plot elbow curve and silhouette scores."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    k_range = optimal_k_results["k_range"]
    inertias = optimal_k_results["inertias"]
    sil_scores = optimal_k_results["silhouette_scores"]
    best_k = optimal_k_results["best_k"]

    # Elbow curve
    axes[0].plot(k_range, inertias, "bo-", linewidth=2, markersize=8)
    axes[0].set_title("Elbow Method", fontweight="bold")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(True, alpha=0.3)

    # Silhouette scores
    axes[1].plot(k_range, sil_scores, "ro-", linewidth=2, markersize=8)
    axes[1].axvline(best_k, color="green", linestyle="--", linewidth=2,
                     label=f"Best K = {best_k}")
    axes[1].set_title("Silhouette Score", fontweight="bold")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Optimal Cluster Selection", fontweight="bold", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_cluster_scatter(
    profiles: pd.DataFrame,
    x_col: str,
    y_col: str,
    cluster_col: str = "Cluster",
    label_col: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Scatter plot of clusters with optional labels.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    clusters = profiles[cluster_col].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(clusters)))

    for cluster, color in zip(sorted(clusters), colors):
        mask = profiles[cluster_col] == cluster
        ax.scatter(
            profiles[mask][x_col],
            profiles[mask][y_col],
            c=[color], s=150, edgecolors="black", linewidth=1,
            label=f"Cluster {cluster}", alpha=0.8, zorder=5,
        )
        # Label points
        if label_col:
            for _, row in profiles[mask].iterrows():
                ax.annotate(
                    str(row[label_col])[:10],
                    (row[x_col], row[y_col]),
                    fontsize=8, ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points",
                )

    ax.set_xlabel(x_col, fontweight="bold")
    ax.set_ylabel(y_col, fontweight="bold")
    ax.set_title(f"Parking Lot Clusters ({x_col} vs {y_col})", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_cluster_profiles_radar(
    cluster_profiles: pd.DataFrame,
    feature_cols: list,
    save_path: Optional[str] = None,
):
    """
    Radar chart showing cluster profiles across all features.
    """
    from matplotlib.patches import FancyBboxPatch

    # Normalize features to 0-1 for radar chart
    normalized = cluster_profiles[feature_cols].copy()
    for col in feature_cols:
        col_min = normalized[col].min()
        col_max = normalized[col].max()
        if col_max - col_min > 0:
            normalized[col] = (normalized[col] - col_min) / (col_max - col_min)
        else:
            normalized[col] = 0.5

    # Radar chart setup
    categories = feature_cols
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(normalized)))

    for idx, (cluster_id, row) in enumerate(normalized.iterrows()):
        values = row.values.tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=f"Cluster {cluster_id}",
                color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=9)
    ax.set_title("Cluster Profiles (Radar Chart)", fontweight="bold", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
