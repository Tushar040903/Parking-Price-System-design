"""
ML-Based Pricing Model Module
==============================

Replaces the rule-based pricing formula with trained ML models.
Implements model comparison across:
- Linear Regression (baseline)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- Random Forest (ensemble trees)
- XGBoost (gradient boosting)

Includes cross-validation, hyperparameter tuning, and evaluation.

Upgrade 3: ML-Based Pricing Model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────
# Data Preparation
# ─────────────────────────────────────────────────────────────────────

def prepare_pricing_data(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "Price",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple:
    """
    Prepare feature matrix and target for pricing model training.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and synthetic price target.
    feature_cols : list of str, optional
        Feature columns to use. Defaults to standard ML features.
    target_col : str
        Name of the target column.
    test_size : float
        Fraction of data for testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, scaler, feature_names)
    """
    if feature_cols is None:
        feature_cols = [
            "OccupancyRate", "QueueLength", "Traffic", "IsSpecialDay",
            "VehicleWeight", "Hour", "DayOfWeek", "IsWeekend",
            "Hour_sin", "Hour_cos", "DayOfWeek_sin", "DayOfWeek_cos",
        ]

    # Add lag/rolling features if available
    for col in df.columns:
        if col.startswith("OccupancyRate_lag_") or col.startswith("OccupancyRate_rolling_"):
            if col not in feature_cols:
                feature_cols.append(col)

    # Filter to available columns
    available = [c for c in feature_cols if c in df.columns]

    # Drop rows with NaN (from lag features)
    subset = df[available + [target_col]].dropna()
    X = subset[available]
    y = subset[target_col]

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=available, index=X.index)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, scaler, available


# ─────────────────────────────────────────────────────────────────────
# Model Training
# ─────────────────────────────────────────────────────────────────────

def train_all_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, dict]:
    """
    Train and evaluate all pricing models.

    Models: Linear Regression, Ridge, Lasso, Random Forest, XGBoost.

    Returns a dictionary of model results with predictions and metrics.
    """
    from xgboost import XGBRegressor

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge (α=1.0)": Ridge(alpha=1.0),
        "Lasso (α=0.1)": Lasso(alpha=0.1),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_split=5,
            random_state=42, n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"   Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "R²": round(r2_score(y_test, y_pred), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            "MAE": round(mean_absolute_error(y_test, y_pred), 4),
        }

        # Cross-validation R²
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        metrics["CV R² (mean)"] = round(cv_scores.mean(), 4)
        metrics["CV R² (std)"] = round(cv_scores.std(), 4)

        results[name] = {
            "model": model,
            "predictions": y_pred,
            "metrics": metrics,
        }
        print(f"   ✅ {name} — R²: {metrics['R²']}, RMSE: {metrics['RMSE']}")

    return results


# ─────────────────────────────────────────────────────────────────────
# Hyperparameter Tuning
# ─────────────────────────────────────────────────────────────────────

def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict:
    """
    Perform grid search hyperparameter tuning for XGBoost.

    Tunes: max_depth, learning_rate, n_estimators, subsample.
    """
    from xgboost import XGBRegressor

    param_grid = {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 200, 300],
        "subsample": [0.7, 0.8, 0.9],
    }

    xgb = XGBRegressor(random_state=42, colsample_bytree=0.8)

    grid_search = GridSearchCV(
        xgb, param_grid,
        cv=3, scoring="r2",
        n_jobs=-1, verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    metrics = {
        "R²": round(r2_score(y_test, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "MAE": round(mean_absolute_error(y_test, y_pred), 4),
    }

    return {
        "best_params": grid_search.best_params_,
        "best_score": round(grid_search.best_score_, 4),
        "model": best_model,
        "predictions": y_pred,
        "metrics": metrics,
    }


# ─────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────

def plot_model_comparison(results: Dict, save_path: Optional[str] = None):
    """
    Bar chart comparing R², RMSE, and MAE across all pricing models.
    """
    metrics_df = pd.DataFrame({
        name: res["metrics"] for name, res in results.items()
    }).T

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics_to_plot = ["R²", "RMSE", "MAE"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    for ax, metric in zip(axes, metrics_to_plot):
        bars = ax.bar(
            range(len(metrics_df)),
            metrics_df[metric],
            color=colors[:len(metrics_df)],
            edgecolor="white",
            width=0.6,
        )
        ax.set_xticks(range(len(metrics_df)))
        ax.set_xticklabels(metrics_df.index, rotation=30, ha="right")
        ax.set_title(metric, fontweight="bold", fontsize=14)
        ax.set_ylabel(metric)

        # Add value labels
        for bar, val in zip(bars, metrics_df[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    plt.suptitle("Pricing Model Comparison", fontweight="bold", fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_actual_vs_predicted(
    y_test: pd.Series,
    results: Dict,
    save_path: Optional[str] = None,
):
    """
    Scatter plots of actual vs predicted prices for each model.
    """
    n_models = len(results)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    for i, (name, result) in enumerate(results.items()):
        ax = axes[i]
        y_pred = result["predictions"]

        ax.scatter(y_test.values, y_pred, alpha=0.3, s=10, color="#4C72B0")

        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect")

        ax.set_title(f"{name}\nR²={result['metrics']['R²']}", fontweight="bold")
        ax.set_xlabel("Actual Price ($)")
        ax.set_ylabel("Predicted Price ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Actual vs Predicted Prices", fontweight="bold", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_residuals(y_test: pd.Series, results: Dict, save_path: Optional[str] = None):
    """Plot residual distributions for each model."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (name, result) in zip(axes, results.items()):
        residuals = y_test.values - result["predictions"]
        ax.hist(residuals, bins=50, alpha=0.7, color="#4C72B0", edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", linewidth=2)
        ax.set_title(f"{name}\nMean Residual: {residuals.mean():.3f}", fontweight="bold")
        ax.set_xlabel("Residual (Actual - Predicted)")
        ax.set_ylabel("Count")

    plt.suptitle("Residual Distributions", fontweight="bold", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
