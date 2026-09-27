"""
A/B Testing & Simulation Framework Module
==========================================

Simulates and compares different pricing strategies:
- Static pricing (fixed $10)
- Linear pricing (original formula)
- Demand-based pricing (weighted formula)
- ML-based pricing (trained model predictions)
- Competitive pricing (with competitor pressure)

Evaluates each strategy on:
- Total revenue
- Average lot utilization
- Price stability (coefficient of variation)
- Queue reduction effectiveness
- Statistical significance testing between strategies

Upgrade 7: A/B Testing / Simulation Framework
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Optional, Callable


# ─────────────────────────────────────────────────────────────────────
# Pricing Strategy Definitions
# ─────────────────────────────────────────────────────────────────────

def static_pricing(row: pd.Series, base_price: float = 10.0) -> float:
    """Fixed static pricing — always returns the base price."""
    return base_price


def linear_pricing(row: pd.Series, alpha: float = 2.0, base_price: float = 10.0) -> float:
    """
    Original linear pricing model.
    Price = base_price + α × (Occupancy / Capacity)
    """
    occ_rate = row.get("OccupancyRate", 0)
    return base_price + alpha * occ_rate


def demand_based_pricing(
    row: pd.Series,
    alpha: float = 1.5,
    beta: float = 0.3,
    gamma: float = 0.4,
    delta: float = 2.0,
    base_price: float = 10.0,
) -> float:
    """
    Enhanced demand-based pricing (hand-tuned weights).
    Considers occupancy, queue, traffic, and special days.
    """
    occ_rate = row.get("OccupancyRate", 0)
    queue = row.get("QueueLength", 0)
    traffic = row.get("Traffic", 1)
    is_special = row.get("IsSpecialDay", 0)

    price = (
        base_price
        + alpha * (occ_rate ** 1.3)
        + beta * (queue / 10)
        + gamma * ((traffic - 1) / 2)
        + delta * is_special
    )
    return max(price, 2.0)


def competitive_pricing(
    row: pd.Series,
    base_price: float = 10.0,
    pressure_weight: float = -0.5,
) -> float:
    """
    Competitive pricing that adjusts based on competitor pressure.
    Higher competitor pressure → lower prices to stay competitive.
    """
    occ_rate = row.get("OccupancyRate", 0)
    pressure = row.get("CompetitorPressure", 0)

    price = base_price + 2.0 * occ_rate + pressure_weight * pressure
    return max(price, 2.0)


# ─────────────────────────────────────────────────────────────────────
# Simulation Engine
# ─────────────────────────────────────────────────────────────────────

def simulate_strategy(
    df: pd.DataFrame,
    strategy_fn: Callable,
    strategy_name: str,
    **kwargs,
) -> Dict:
    """
    Simulate a pricing strategy across the entire dataset.

    Computes prices for each record and calculates performance metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with demand features.
    strategy_fn : callable
        Pricing function that takes a row and returns a price.
    strategy_name : str
        Name of the strategy for reporting.
    **kwargs
        Additional keyword arguments passed to the strategy function.

    Returns
    -------
    dict
        Strategy results including prices and metrics.
    """
    prices = df.apply(lambda row: strategy_fn(row, **kwargs), axis=1)

    # Simulate demand response to prices (higher price → slightly lower demand)
    # Using a simple elasticity model for simulation
    base_demand = df["OccupancyRate"].values
    price_ratio = prices / 10.0  # Relative to base price
    elasticity = -0.3  # Inelastic demand
    adjusted_demand = base_demand * (price_ratio ** elasticity)
    adjusted_demand = np.clip(adjusted_demand, 0, 1)

    # Revenue = price × demand × capacity
    revenue = prices * adjusted_demand * df["Capacity"]

    # Metrics
    metrics = {
        "strategy": strategy_name,
        "avg_price": round(prices.mean(), 2),
        "median_price": round(prices.median(), 2),
        "price_std": round(prices.std(), 2),
        "price_cv": round(prices.std() / prices.mean() * 100, 1) if prices.mean() > 0 else 0,
        "total_revenue": round(revenue.sum(), 2),
        "avg_revenue_per_record": round(revenue.mean(), 2),
        "avg_utilization": round(adjusted_demand.mean() * 100, 1),
        "min_price": round(prices.min(), 2),
        "max_price": round(prices.max(), 2),
        "price_range": round(prices.max() - prices.min(), 2),
    }

    return {
        "name": strategy_name,
        "prices": prices,
        "adjusted_demand": adjusted_demand,
        "revenue": revenue,
        "metrics": metrics,
    }


def simulate_ml_strategy(
    df: pd.DataFrame,
    model,
    scaler,
    feature_cols: List[str],
    strategy_name: str = "ML-Based",
) -> Dict:
    """
    Simulate pricing using a trained ML model.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with features.
    model : trained sklearn/xgboost model
        Pricing model.
    scaler : fitted StandardScaler
        Feature scaler.
    feature_cols : list
        Feature column names.
    """
    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].fillna(0)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=available_cols, index=X.index)

    prices = pd.Series(model.predict(X_scaled), index=df.index)
    prices = prices.clip(2.0, 100.0)

    # Demand response
    base_demand = df["OccupancyRate"].values
    price_ratio = prices / 10.0
    elasticity = -0.3
    adjusted_demand = base_demand * (price_ratio.values ** elasticity)
    adjusted_demand = np.clip(adjusted_demand, 0, 1)

    revenue = prices * adjusted_demand * df["Capacity"]

    metrics = {
        "strategy": strategy_name,
        "avg_price": round(prices.mean(), 2),
        "median_price": round(prices.median(), 2),
        "price_std": round(prices.std(), 2),
        "price_cv": round(prices.std() / prices.mean() * 100, 1),
        "total_revenue": round(revenue.sum(), 2),
        "avg_revenue_per_record": round(revenue.mean(), 2),
        "avg_utilization": round(adjusted_demand.mean() * 100, 1),
        "min_price": round(prices.min(), 2),
        "max_price": round(prices.max(), 2),
        "price_range": round(prices.max() - prices.min(), 2),
    }

    return {
        "name": strategy_name,
        "prices": prices,
        "adjusted_demand": adjusted_demand,
        "revenue": revenue,
        "metrics": metrics,
    }


# ─────────────────────────────────────────────────────────────────────
# Statistical Significance Testing
# ─────────────────────────────────────────────────────────────────────

def test_strategy_significance(
    results_a: Dict,
    results_b: Dict,
) -> Dict:
    """
    Test whether two strategies produce significantly different revenues.

    Uses Welch's t-test (unequal variance assumed).
    """
    rev_a = results_a["revenue"]
    rev_b = results_b["revenue"]

    t_stat, p_value = stats.ttest_ind(rev_a, rev_b, equal_var=False)

    return {
        "comparison": f"{results_a['name']} vs {results_b['name']}",
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
        f"{results_a['name']}_mean_rev": round(rev_a.mean(), 2),
        f"{results_b['name']}_mean_rev": round(rev_b.mean(), 2),
        "revenue_diff_%": round(
            (rev_b.mean() - rev_a.mean()) / rev_a.mean() * 100, 2
        ) if rev_a.mean() > 0 else 0,
    }


def run_all_significance_tests(all_results: Dict) -> pd.DataFrame:
    """
    Run pairwise significance tests between all strategy pairs.
    """
    strategy_names = list(all_results.keys())
    test_results = []

    for i in range(len(strategy_names)):
        for j in range(i + 1, len(strategy_names)):
            result = test_strategy_significance(
                all_results[strategy_names[i]],
                all_results[strategy_names[j]],
            )
            test_results.append(result)

    return pd.DataFrame(test_results)


# ─────────────────────────────────────────────────────────────────────
# Comparison & Visualization
# ─────────────────────────────────────────────────────────────────────

def compare_strategies(all_results: Dict) -> pd.DataFrame:
    """
    Create a comparison table of all strategy metrics.
    """
    metrics_list = [res["metrics"] for res in all_results.values()]
    comparison_df = pd.DataFrame(metrics_list).set_index("strategy")
    return comparison_df


def plot_strategy_comparison(all_results: Dict, save_path: Optional[str] = None):
    """
    Comprehensive visualization comparing all pricing strategies.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    strategies = list(all_results.keys())
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(strategies)))

    # 1. Revenue comparison
    revenues = [all_results[s]["metrics"]["total_revenue"] for s in strategies]
    bars = axes[0, 0].bar(strategies, revenues, color=colors, edgecolor="white")
    axes[0, 0].set_title("Total Revenue by Strategy", fontweight="bold")
    axes[0, 0].set_ylabel("Total Revenue ($)")
    for bar, rev in zip(bars, revenues):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"${rev:,.0f}", ha="center", va="bottom", fontsize=9,
                         fontweight="bold")
    axes[0, 0].tick_params(axis="x", rotation=30)

    # 2. Average price distribution (violin plot)
    price_data = [all_results[s]["prices"].values for s in strategies]
    parts = axes[0, 1].violinplot(price_data, positions=range(len(strategies)),
                                   showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    axes[0, 1].set_xticks(range(len(strategies)))
    axes[0, 1].set_xticklabels(strategies, rotation=30, ha="right")
    axes[0, 1].set_title("Price Distribution by Strategy", fontweight="bold")
    axes[0, 1].set_ylabel("Price ($)")

    # 3. Utilization comparison
    utilizations = [all_results[s]["metrics"]["avg_utilization"] for s in strategies]
    axes[1, 0].bar(strategies, utilizations, color=colors, edgecolor="white")
    axes[1, 0].set_title("Avg Utilization by Strategy", fontweight="bold")
    axes[1, 0].set_ylabel("Utilization (%)")
    axes[1, 0].tick_params(axis="x", rotation=30)
    for i, (bar_rect, util) in enumerate(zip(axes[1, 0].patches, utilizations)):
        axes[1, 0].text(bar_rect.get_x() + bar_rect.get_width() / 2,
                         bar_rect.get_height(), f"{util:.1f}%",
                         ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 4. Price stability (CV)
    cvs = [all_results[s]["metrics"]["price_cv"] for s in strategies]
    bars = axes[1, 1].bar(strategies, cvs, color=colors, edgecolor="white")
    axes[1, 1].set_title("Price Stability (Lower CV = More Stable)", fontweight="bold")
    axes[1, 1].set_ylabel("Coefficient of Variation (%)")
    axes[1, 1].tick_params(axis="x", rotation=30)

    plt.suptitle("📊 A/B Testing: Pricing Strategy Comparison",
                 fontweight="bold", fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_price_timeseries(
    df: pd.DataFrame,
    all_results: Dict,
    n_samples: int = 500,
    save_path: Optional[str] = None,
):
    """
    Plot price trajectories over time for each strategy.
    Shows only the first n_samples records for clarity.
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    colors_map = {
        "Static": "#8172B3",
        "Linear": "#4C72B0",
        "Demand-Based": "#DD8452",
        "Competitive": "#55A868",
        "ML-Based": "#C44E52",
    }

    for strategy_name, result in all_results.items():
        prices = result["prices"].values[:n_samples]
        color = colors_map.get(strategy_name, "#333333")
        ax.plot(prices, label=strategy_name, alpha=0.8, linewidth=1.2, color=color)

    ax.set_title("Price Trajectories Across Strategies", fontweight="bold", fontsize=14)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Price ($)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
