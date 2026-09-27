"""
Price Elasticity Estimation Module
====================================

Estimates how sensitive parking demand is to price changes:
- Overall price elasticity of demand
- Per-lot elasticity estimation
- Per-vehicle-type elasticity
- Per-time-period elasticity (peak vs off-peak)
- Elasticity curve visualization

Upgrade 6: Price Elasticity Estimation

Theory:
    Price Elasticity = (% change in demand) / (% change in price)
    E < -1: Elastic (demand is very sensitive to price)
    -1 < E < 0: Inelastic (demand is not very sensitive)
    E = -1: Unit elastic
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────
# Elasticity Estimation
# ─────────────────────────────────────────────────────────────────────

def estimate_point_elasticity(
    prices: np.ndarray,
    demands: np.ndarray,
) -> Dict[str, float]:
    """
    Estimate point elasticity using log-log regression.

    ln(demand) = α + β * ln(price) + ε
    β is the price elasticity of demand.

    Parameters
    ----------
    prices : array-like
        Price values.
    demands : array-like
        Demand values (e.g., occupancy rate).

    Returns
    -------
    dict
        Elasticity coefficient, R², p-value, and interpretation.
    """
    # Filter positive values for log transform
    mask = (prices > 0) & (demands > 0)
    log_price = np.log(prices[mask])
    log_demand = np.log(demands[mask])

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_price, log_demand)

    interpretation = (
        "Elastic (demand very sensitive to price)"
        if slope < -1
        else "Unit elastic"
        if abs(slope + 1) < 0.1
        else "Inelastic (demand not very sensitive to price)"
        if -1 < slope < 0
        else "Positive relationship (unusual — Giffen/Veblen good?)"
    )

    return {
        "elasticity": round(slope, 4),
        "intercept": round(intercept, 4),
        "r_squared": round(r_value ** 2, 4),
        "p_value": round(p_value, 6),
        "std_error": round(std_err, 4),
        "significant": p_value < 0.05,
        "interpretation": interpretation,
    }


def estimate_arc_elasticity(
    p1: float, p2: float,
    q1: float, q2: float,
) -> float:
    """
    Estimate arc elasticity between two points.

    Uses the midpoint method for more stable estimates.
    """
    pct_change_q = (q2 - q1) / ((q1 + q2) / 2)
    pct_change_p = (p2 - p1) / ((p1 + p2) / 2)

    if pct_change_p == 0:
        return 0.0

    return pct_change_q / pct_change_p


# ─────────────────────────────────────────────────────────────────────
# Segmented Elasticity Analysis
# ─────────────────────────────────────────────────────────────────────

def elasticity_by_lot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate price elasticity separately for each parking lot.

    Returns a DataFrame with elasticity metrics per lot.
    """
    results = []

    for lot_id in df["SystemCodeNumber"].unique():
        lot_data = df[df["SystemCodeNumber"] == lot_id]
        if len(lot_data) < 10:
            continue

        elasticity = estimate_point_elasticity(
            lot_data["Price"].values,
            lot_data["OccupancyRate"].values,
        )
        elasticity["lot_id"] = lot_id
        results.append(elasticity)

    return pd.DataFrame(results).set_index("lot_id")


def elasticity_by_vehicle_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate price elasticity separately for each vehicle type.
    """
    results = []

    for vtype in df["VehicleType"].unique():
        vtype_data = df[df["VehicleType"] == vtype]
        if len(vtype_data) < 10:
            continue

        elasticity = estimate_point_elasticity(
            vtype_data["Price"].values,
            vtype_data["OccupancyRate"].values,
        )
        elasticity["vehicle_type"] = vtype
        results.append(elasticity)

    return pd.DataFrame(results).set_index("vehicle_type")


def elasticity_by_time_period(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate elasticity for different time periods:
    - Morning (6-12), Afternoon (12-18), Evening (18-24), Night (0-6)
    - Peak vs Off-peak hours
    """
    results = []

    if "Hour" not in df.columns:
        return pd.DataFrame()

    time_periods = {
        "Morning (6-12)": df[(df["Hour"] >= 6) & (df["Hour"] < 12)],
        "Afternoon (12-18)": df[(df["Hour"] >= 12) & (df["Hour"] < 18)],
        "Evening (18-24)": df[(df["Hour"] >= 18) & (df["Hour"] < 24)],
        "Night (0-6)": df[(df["Hour"] >= 0) & (df["Hour"] < 6)],
        "Peak Hours": df[df["Hour"].isin([8, 9, 10, 17, 18, 19])],
        "Off-Peak Hours": df[~df["Hour"].isin([8, 9, 10, 17, 18, 19])],
    }

    for period_name, period_data in time_periods.items():
        if len(period_data) < 10:
            continue

        elasticity = estimate_point_elasticity(
            period_data["Price"].values,
            period_data["OccupancyRate"].values,
        )
        elasticity["time_period"] = period_name
        results.append(elasticity)

    return pd.DataFrame(results).set_index("time_period")


# ─────────────────────────────────────────────────────────────────────
# Demand Curve Estimation
# ─────────────────────────────────────────────────────────────────────

def estimate_demand_curve(
    df: pd.DataFrame,
    n_bins: int = 20,
) -> pd.DataFrame:
    """
    Estimate the demand curve by binning prices and computing
    average demand at each price level.
    """
    df_copy = df.copy()
    df_copy["PriceBin"] = pd.cut(df_copy["Price"], bins=n_bins)
    demand_curve = df_copy.groupby("PriceBin", observed=True).agg(
        avg_price=("Price", "mean"),
        avg_demand=("OccupancyRate", "mean"),
        count=("OccupancyRate", "count"),
    ).reset_index()

    return demand_curve


# ─────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────

def plot_elasticity_by_lot(elasticity_df: pd.DataFrame, save_path: Optional[str] = None):
    """Bar chart of elasticity coefficients per parking lot."""
    fig, ax = plt.subplots(figsize=(14, 6))

    sorted_df = elasticity_df.sort_values("elasticity")
    colors = ["#C44E52" if e < -1 else "#DD8452" if e < 0 else "#55A868"
              for e in sorted_df["elasticity"]]

    ax.barh(range(len(sorted_df)), sorted_df["elasticity"], color=colors, edgecolor="white")
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels([str(n)[:12] for n in sorted_df.index])
    ax.axvline(-1, color="black", linestyle="--", linewidth=1.5,
               label="Unit Elastic (E=-1)")
    ax.axvline(0, color="gray", linestyle="-", linewidth=1)

    ax.set_xlabel("Price Elasticity of Demand", fontweight="bold")
    ax.set_title("Price Elasticity by Parking Lot", fontweight="bold", fontsize=14)
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)

    # Add annotations
    for i, (idx, row) in enumerate(sorted_df.iterrows()):
        sig = "✓" if row.get("significant", False) else ""
        ax.text(
            row["elasticity"] + 0.01 * abs(sorted_df["elasticity"].max()),
            i, f'{row["elasticity"]:.2f} {sig}',
            va="center", fontsize=9,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_demand_curve(demand_curve: pd.DataFrame, save_path: Optional[str] = None):
    """Plot the estimated demand curve."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(demand_curve["avg_price"], demand_curve["avg_demand"],
               s=demand_curve["count"] / 5, alpha=0.7, color="#4C72B0",
               edgecolors="black", linewidth=0.5)

    # Fit and plot trend line
    z = np.polyfit(demand_curve["avg_price"], demand_curve["avg_demand"], 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(demand_curve["avg_price"].min(),
                            demand_curve["avg_price"].max(), 100)
    ax.plot(x_smooth, p(x_smooth), "r-", linewidth=2, label="Fitted Demand Curve")

    ax.set_xlabel("Price ($)", fontweight="bold")
    ax.set_ylabel("Average Demand (Occupancy Rate)", fontweight="bold")
    ax.set_title("Estimated Demand Curve", fontweight="bold", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_elasticity_comparison(
    lot_elasticity: pd.DataFrame,
    vehicle_elasticity: pd.DataFrame,
    time_elasticity: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """
    Comprehensive elasticity comparison across segments.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # By vehicle type
    if not vehicle_elasticity.empty:
        colors = plt.cm.Set2(np.linspace(0, 0.8, len(vehicle_elasticity)))
        axes[0].bar(vehicle_elasticity.index, vehicle_elasticity["elasticity"],
                     color=colors, edgecolor="white")
        axes[0].axhline(-1, color="red", linestyle="--", label="Unit Elastic")
        axes[0].set_title("By Vehicle Type", fontweight="bold")
        axes[0].set_ylabel("Elasticity")
        axes[0].legend()

    # By time period
    if not time_elasticity.empty:
        colors = plt.cm.Set3(np.linspace(0, 0.8, len(time_elasticity)))
        axes[1].barh(range(len(time_elasticity)), time_elasticity["elasticity"],
                      color=colors, edgecolor="white")
        axes[1].set_yticks(range(len(time_elasticity)))
        axes[1].set_yticklabels(time_elasticity.index, fontsize=9)
        axes[1].axvline(-1, color="red", linestyle="--", label="Unit Elastic")
        axes[1].set_title("By Time Period", fontweight="bold")
        axes[1].set_xlabel("Elasticity")
        axes[1].legend()

    # R² values (model fit quality)
    if not lot_elasticity.empty:
        axes[2].scatter(
            lot_elasticity["elasticity"],
            lot_elasticity["r_squared"],
            s=100, alpha=0.7, color="#4C72B0", edgecolors="black",
        )
        for idx, row in lot_elasticity.iterrows():
            axes[2].annotate(str(idx)[:8], (row["elasticity"], row["r_squared"]),
                             fontsize=7, xytext=(5, 5), textcoords="offset points")
        axes[2].set_xlabel("Elasticity", fontweight="bold")
        axes[2].set_ylabel("R² (Model Fit)", fontweight="bold")
        axes[2].set_title("Elasticity vs Model Fit", fontweight="bold")
        axes[2].grid(True, alpha=0.3)

    plt.suptitle("Price Elasticity Analysis Across Segments",
                 fontweight="bold", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
