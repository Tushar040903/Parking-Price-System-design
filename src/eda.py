"""
Exploratory Data Analysis & Statistical Testing Module
======================================================

Provides comprehensive EDA functions for the ParkWise system:
- Distribution analysis across lots, hours, and days
- Correlation analysis with heatmaps
- Temporal pattern decomposition
- Hypothesis testing (special days vs normal days)
- Occupancy pattern visualization

Upgrade 1: Deep EDA & Statistical Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────
# Plot Configuration
# ─────────────────────────────────────────────────────────────────────

def set_plot_style():
    """Set a professional, consistent plotting style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


# ─────────────────────────────────────────────────────────────────────
# Distribution Analysis
# ─────────────────────────────────────────────────────────────────────

def plot_occupancy_distribution(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot the distribution of occupancy rates across all lots.

    Shows histogram + KDE for overall distribution and box plots per lot.
    """
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Overall distribution
    axes[0].hist(df["OccupancyRate"], bins=50, alpha=0.7, color="#4C72B0",
                 edgecolor="white", density=True)
    df["OccupancyRate"].plot.kde(ax=axes[0], color="#C44E52", linewidth=2)
    axes[0].set_title("Distribution of Occupancy Rates", fontweight="bold")
    axes[0].set_xlabel("Occupancy Rate")
    axes[0].set_ylabel("Density")
    axes[0].axvline(df["OccupancyRate"].mean(), color="#DD8452", linestyle="--",
                     linewidth=2, label=f'Mean: {df["OccupancyRate"].mean():.3f}')
    axes[0].legend()

    # Box plot per lot
    lot_data = [group["OccupancyRate"].values
                for _, group in df.groupby("SystemCodeNumber")]
    lot_names = [name for name, _ in df.groupby("SystemCodeNumber")]
    bp = axes[1].boxplot(lot_data, vert=True, patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(lot_names)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    axes[1].set_xticklabels([n[:8] for n in lot_names], rotation=45, ha="right")
    axes[1].set_title("Occupancy Rate by Parking Lot", fontweight="bold")
    axes[1].set_ylabel("Occupancy Rate")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_feature_distributions(df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot distributions of all key numerical features."""
    set_plot_style()
    features = ["OccupancyRate", "QueueLength", "Traffic", "VehicleWeight"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, feat in zip(axes.ravel(), features):
        ax.hist(df[feat], bins=30, alpha=0.7, color="#4C72B0", edgecolor="white")
        ax.set_title(f"Distribution of {feat}", fontweight="bold")
        ax.set_xlabel(feat)
        ax.set_ylabel("Count")
        ax.axvline(df[feat].mean(), color="#C44E52", linestyle="--",
                    linewidth=2, label=f"Mean: {df[feat].mean():.2f}")
        ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ─────────────────────────────────────────────────────────────────────
# Correlation Analysis
# ─────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Generate a correlation heatmap for numerical features.

    Highlights which features are most correlated with demand.
    """
    set_plot_style()
    numeric_cols = [
        "OccupancyRate", "QueueLength", "Traffic", "IsSpecialDay",
        "VehicleWeight", "Capacity", "Occupancy",
    ]
    # Add temporal features if available
    for col in ["Hour", "DayOfWeek", "IsWeekend"]:
        if col in df.columns:
            numeric_cols.append(col)

    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmin=-1, vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontweight="bold", fontsize=16, pad=20)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

    return corr_matrix


# ─────────────────────────────────────────────────────────────────────
# Temporal Patterns
# ─────────────────────────────────────────────────────────────────────

def plot_hourly_patterns(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Visualize occupancy patterns across hours of the day.

    Shows average occupancy rate per hour with confidence intervals.
    """
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Average occupancy by hour
    hourly = df.groupby("Hour")["OccupancyRate"].agg(["mean", "std"]).reset_index()
    axes[0].fill_between(
        hourly["Hour"],
        hourly["mean"] - hourly["std"],
        hourly["mean"] + hourly["std"],
        alpha=0.2, color="#4C72B0",
    )
    axes[0].plot(hourly["Hour"], hourly["mean"], "o-", color="#4C72B0",
                  linewidth=2, markersize=6)
    axes[0].set_title("Average Occupancy Rate by Hour", fontweight="bold")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Occupancy Rate")
    axes[0].set_xticks(range(0, 24))

    # Heatmap: Hour vs Day of Week
    pivot = df.pivot_table(
        values="OccupancyRate",
        index="DayOfWeek",
        columns="Hour",
        aggfunc="mean",
    )
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        annot=False,
        ax=axes[1],
        yticklabels=day_labels[:len(pivot)],
        cbar_kws={"label": "Avg Occupancy Rate"},
    )
    axes[1].set_title("Occupancy: Hour × Day of Week", fontweight="bold")
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("Day of Week")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_daily_patterns(df: pd.DataFrame, save_path: Optional[str] = None):
    """Visualize daily occupancy trends."""
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Average by day of week
    daily = df.groupby("DayOfWeek")["OccupancyRate"].mean()
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    colors = ["#4C72B0"] * 5 + ["#C44E52"] * 2  # Highlight weekends
    axes[0].bar(range(len(daily)), daily.values, color=colors[:len(daily)],
                edgecolor="white")
    axes[0].set_xticks(range(len(daily)))
    axes[0].set_xticklabels(day_labels[:len(daily)])
    axes[0].set_title("Average Occupancy by Day of Week", fontweight="bold")
    axes[0].set_ylabel("Occupancy Rate")

    # Time series trend
    daily_ts = df.groupby(df["Timestamp"].dt.date)["OccupancyRate"].mean()
    axes[1].plot(daily_ts.index, daily_ts.values, color="#4C72B0", linewidth=1.5)
    axes[1].fill_between(daily_ts.index, daily_ts.values, alpha=0.3, color="#4C72B0")
    axes[1].set_title("Daily Average Occupancy Over Time", fontweight="bold")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Average Occupancy Rate")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_lot_comparison(df: pd.DataFrame, save_path: Optional[str] = None):
    """Compare key metrics across all 14 parking lots."""
    set_plot_style()
    lot_stats = df.groupby("SystemCodeNumber").agg(
        avg_occupancy=("OccupancyRate", "mean"),
        avg_queue=("QueueLength", "mean"),
        capacity=("Capacity", "first"),
        total_records=("OccupancyRate", "count"),
    ).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Occupancy
    lot_stats_sorted = lot_stats.sort_values("avg_occupancy", ascending=True)
    axes[0, 0].barh(
        range(len(lot_stats_sorted)),
        lot_stats_sorted["avg_occupancy"],
        color=plt.cm.RdYlGn_r(lot_stats_sorted["avg_occupancy"]),
        edgecolor="white",
    )
    axes[0, 0].set_yticks(range(len(lot_stats_sorted)))
    axes[0, 0].set_yticklabels([n[:12] for n in lot_stats_sorted["SystemCodeNumber"]])
    axes[0, 0].set_title("Avg Occupancy Rate by Lot", fontweight="bold")
    axes[0, 0].set_xlabel("Occupancy Rate")

    # Queue length
    lot_stats_sorted = lot_stats.sort_values("avg_queue", ascending=True)
    axes[0, 1].barh(
        range(len(lot_stats_sorted)),
        lot_stats_sorted["avg_queue"],
        color=plt.cm.Blues(np.linspace(0.3, 0.9, len(lot_stats_sorted))),
        edgecolor="white",
    )
    axes[0, 1].set_yticks(range(len(lot_stats_sorted)))
    axes[0, 1].set_yticklabels([n[:12] for n in lot_stats_sorted["SystemCodeNumber"]])
    axes[0, 1].set_title("Avg Queue Length by Lot", fontweight="bold")
    axes[0, 1].set_xlabel("Queue Length")

    # Capacity
    lot_stats_sorted = lot_stats.sort_values("capacity", ascending=True)
    axes[1, 0].barh(
        range(len(lot_stats_sorted)),
        lot_stats_sorted["capacity"],
        color=plt.cm.Oranges(np.linspace(0.3, 0.9, len(lot_stats_sorted))),
        edgecolor="white",
    )
    axes[1, 0].set_yticks(range(len(lot_stats_sorted)))
    axes[1, 0].set_yticklabels([n[:12] for n in lot_stats_sorted["SystemCodeNumber"]])
    axes[1, 0].set_title("Capacity by Lot", fontweight="bold")
    axes[1, 0].set_xlabel("Capacity (spots)")

    # Vehicle type distribution
    vtype_counts = df.groupby("VehicleType").size()
    axes[1, 1].pie(
        vtype_counts.values,
        labels=vtype_counts.index,
        autopct="%1.1f%%",
        colors=plt.cm.Set2.colors[:len(vtype_counts)],
        startangle=90,
    )
    axes[1, 1].set_title("Vehicle Type Distribution", fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ─────────────────────────────────────────────────────────────────────
# Statistical Hypothesis Testing
# ─────────────────────────────────────────────────────────────────────

def run_statistical_tests(df: pd.DataFrame) -> Dict[str, dict]:
    """
    Run statistical hypothesis tests on the parking data.

    Tests performed:
    1. T-test: Is occupancy significantly different on special days?
    2. Mann-Whitney U: Non-parametric test for special day occupancy
    3. ANOVA: Does occupancy differ significantly across parking lots?
    4. Chi-squared: Is traffic condition independent of special days?

    Returns
    -------
    dict
        Dictionary of test results with statistics and p-values.
    """
    results = {}

    # ── Test 1: Special Day vs Normal Day (Occupancy) ──
    special = df[df["IsSpecialDay"] == 1]["OccupancyRate"]
    normal = df[df["IsSpecialDay"] == 0]["OccupancyRate"]

    t_stat, t_pval = stats.ttest_ind(special, normal, equal_var=False)
    results["t_test_special_day"] = {
        "test": "Welch's t-test",
        "hypothesis": "H0: Mean occupancy is the same on special vs normal days",
        "statistic": round(t_stat, 4),
        "p_value": round(t_pval, 6),
        "significant": t_pval < 0.05,
        "special_day_mean": round(special.mean(), 4),
        "normal_day_mean": round(normal.mean(), 4),
    }

    # ── Test 2: Mann-Whitney U (non-parametric) ──
    u_stat, u_pval = stats.mannwhitneyu(special, normal, alternative="two-sided")
    results["mann_whitney_special_day"] = {
        "test": "Mann-Whitney U test",
        "hypothesis": "H0: Occupancy distributions are identical for special vs normal days",
        "statistic": round(u_stat, 4),
        "p_value": round(u_pval, 6),
        "significant": u_pval < 0.05,
    }

    # ── Test 3: ANOVA across lots ──
    lot_groups = [group["OccupancyRate"].values
                  for _, group in df.groupby("SystemCodeNumber")]
    f_stat, f_pval = stats.f_oneway(*lot_groups)
    results["anova_across_lots"] = {
        "test": "One-way ANOVA",
        "hypothesis": "H0: Mean occupancy is the same across all parking lots",
        "statistic": round(f_stat, 4),
        "p_value": round(f_pval, 6),
        "significant": f_pval < 0.05,
    }

    # ── Test 4: Chi-squared (Traffic vs Special Day) ──
    contingency = pd.crosstab(df["TrafficConditionNearby"], df["IsSpecialDay"])
    chi2, chi_pval, dof, expected = stats.chi2_contingency(contingency)
    results["chi2_traffic_special"] = {
        "test": "Chi-squared test",
        "hypothesis": "H0: Traffic condition is independent of special day status",
        "statistic": round(chi2, 4),
        "p_value": round(chi_pval, 6),
        "degrees_of_freedom": dof,
        "significant": chi_pval < 0.05,
    }

    return results


def print_test_results(results: Dict[str, dict]):
    """Pretty-print statistical test results."""
    for test_name, result in results.items():
        print(f"\n{'='*60}")
        print(f"📊 {result['test']}")
        print(f"{'='*60}")
        print(f"   Hypothesis: {result['hypothesis']}")
        print(f"   Test Statistic: {result['statistic']}")
        print(f"   P-value: {result['p_value']}")
        verdict = "✅ REJECT H0 (significant)" if result["significant"] else "❌ FAIL TO REJECT H0"
        print(f"   Verdict (α=0.05): {verdict}")
        for key in result:
            if key not in ["test", "hypothesis", "statistic", "p_value", "significant"]:
                print(f"   {key}: {result[key]}")
