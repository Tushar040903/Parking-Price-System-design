"""
Time-Series Demand Forecasting Module
======================================

Implements demand forecasting for parking lot occupancy:
- ARIMA / SARIMA (classical statistical models)
- XGBoost with lag features (gradient boosting)
- Model evaluation with RMSE, MAE, MAPE
- Forecast comparison across models

Upgrade 2: Time-Series Demand Forecasting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────
# Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive forecast evaluation metrics.

    Returns RMSE, MAE, MAPE, and R² score.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape_val = mape(y_true, y_pred)

    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE (%)": round(mape_val, 2),
        "R²": round(r2, 4),
    }


# ─────────────────────────────────────────────────────────────────────
# Data Preparation for Forecasting
# ─────────────────────────────────────────────────────────────────────

def prepare_lot_series(
    df: pd.DataFrame,
    lot_id: str,
    target: str = "OccupancyRate",
    freq: str = "h",
) -> pd.Series:
    """
    Extract a time-series for a specific parking lot.

    Resamples to hourly frequency and forward-fills missing values.
    """
    lot_df = df[df["SystemCodeNumber"] == lot_id].copy()
    lot_df = lot_df.set_index("Timestamp")
    series = lot_df[target].resample(freq).mean().ffill().bfill()
    return series


def prepare_xgboost_features(
    series: pd.Series,
    lags: list = [1, 2, 3, 6, 12, 24],
    rolling_windows: list = [3, 6, 12, 24],
) -> pd.DataFrame:
    """
    Create feature matrix for XGBoost time-series forecasting.

    Features include:
    - Lag values (t-1, t-2, ..., t-24)
    - Rolling mean and std (windows: 3, 6, 12, 24)
    - Hour of day, day of week (cyclical encoded)
    """
    feat_df = pd.DataFrame(index=series.index)
    feat_df["target"] = series.values

    # Lag features
    for lag in lags:
        feat_df[f"lag_{lag}"] = series.shift(lag)

    # Rolling features
    for window in rolling_windows:
        feat_df[f"rolling_mean_{window}"] = series.rolling(window, min_periods=1).mean()
        feat_df[f"rolling_std_{window}"] = series.rolling(window, min_periods=1).std().fillna(0)

    # Temporal features
    feat_df["hour"] = series.index.hour
    feat_df["day_of_week"] = series.index.dayofweek
    feat_df["hour_sin"] = np.sin(2 * np.pi * feat_df["hour"] / 24)
    feat_df["hour_cos"] = np.cos(2 * np.pi * feat_df["hour"] / 24)
    feat_df["dow_sin"] = np.sin(2 * np.pi * feat_df["day_of_week"] / 7)
    feat_df["dow_cos"] = np.cos(2 * np.pi * feat_df["day_of_week"] / 7)

    feat_df = feat_df.dropna()
    return feat_df


# ─────────────────────────────────────────────────────────────────────
# ARIMA / SARIMA Forecasting
# ─────────────────────────────────────────────────────────────────────

def train_arima(
    series: pd.Series,
    order: Tuple[int, int, int] = (2, 1, 2),
    test_size: int = 48,
) -> Dict:
    """
    Train an ARIMA model on the time series and forecast.

    Parameters
    ----------
    series : pd.Series
        Time series to forecast.
    order : tuple
        (p, d, q) order for ARIMA.
    test_size : int
        Number of time steps to hold out for testing.

    Returns
    -------
    dict
        Contains model, predictions, actuals, and metrics.
    """
    from statsmodels.tsa.arima.model import ARIMA

    train = series[:-test_size]
    test = series[-test_size:]

    model = ARIMA(train, order=order)
    fitted = model.fit()

    # Forecast
    predictions = fitted.forecast(steps=test_size)
    metrics = evaluate_forecast(test.values, predictions.values)

    return {
        "model_name": f"ARIMA{order}",
        "model": fitted,
        "train": train,
        "test": test,
        "predictions": predictions,
        "metrics": metrics,
    }


def train_sarima(
    series: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24),
    test_size: int = 48,
) -> Dict:
    """
    Train a SARIMA model (seasonal ARIMA) on the time series.

    Default seasonal period is 24 (daily seasonality for hourly data).
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    train = series[:-test_size]
    test = series[-test_size:]

    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False, maxiter=200)

    predictions = fitted.forecast(steps=test_size)
    metrics = evaluate_forecast(test.values, predictions.values)

    return {
        "model_name": f"SARIMA{order}x{seasonal_order}",
        "model": fitted,
        "train": train,
        "test": test,
        "predictions": predictions,
        "metrics": metrics,
    }


# ─────────────────────────────────────────────────────────────────────
# XGBoost Forecasting
# ─────────────────────────────────────────────────────────────────────

def train_xgboost_forecaster(
    feat_df: pd.DataFrame,
    test_size: int = 48,
    params: Optional[Dict] = None,
) -> Dict:
    """
    Train an XGBoost model for time-series forecasting.

    Uses lag features, rolling statistics, and cyclical time encodings.
    """
    from xgboost import XGBRegressor

    target = feat_df["target"]
    features = feat_df.drop(columns=["target"])

    X_train = features[:-test_size]
    X_test = features[-test_size:]
    y_train = target[:-test_size]
    y_test = target[-test_size:]

    if params is None:
        params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }

    model = XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    predictions = model.predict(X_test)
    metrics = evaluate_forecast(y_test.values, predictions)

    return {
        "model_name": "XGBoost",
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
        "metrics": metrics,
        "feature_names": list(features.columns),
    }


# ─────────────────────────────────────────────────────────────────────
# Model Comparison
# ─────────────────────────────────────────────────────────────────────

def compare_forecasting_models(
    df: pd.DataFrame,
    lot_id: str,
    test_size: int = 48,
) -> Dict:
    """
    Compare ARIMA, SARIMA, and XGBoost forecasting models for a lot.

    Returns results for all three models with metrics comparison.
    """
    print(f"\n[Forecasting] Lot: {lot_id}")
    print("=" * 50)

    # Prepare series
    series = prepare_lot_series(df, lot_id)
    print(f"   Series length: {len(series)} hourly observations")

    results = {}

    # ARIMA
    try:
        print("   Training ARIMA(2,1,2)...")
        results["ARIMA"] = train_arima(series, order=(2, 1, 2), test_size=test_size)
        print(f"   ✅ ARIMA — RMSE: {results['ARIMA']['metrics']['RMSE']}")
    except Exception as e:
        print(f"   ⚠️ ARIMA failed: {e}")

    # SARIMA
    try:
        # Use smaller seasonal period if series is short
        seasonal_period = min(24, len(series) // 4)
        if seasonal_period < 2:
            seasonal_period = 2
        print(f"   Training SARIMA (seasonal period={seasonal_period})...")
        results["SARIMA"] = train_sarima(
            series,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, seasonal_period),
            test_size=test_size,
        )
        print(f"   ✅ SARIMA — RMSE: {results['SARIMA']['metrics']['RMSE']}")
    except Exception as e:
        print(f"   ⚠️ SARIMA failed: {e}")

    # XGBoost
    try:
        print("   Training XGBoost with lag features...")
        feat_df = prepare_xgboost_features(series)
        results["XGBoost"] = train_xgboost_forecaster(feat_df, test_size=test_size)
        print(f"   ✅ XGBoost — RMSE: {results['XGBoost']['metrics']['RMSE']}")
    except Exception as e:
        print(f"   ⚠️ XGBoost failed: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────

def plot_forecast_comparison(results: Dict, lot_id: str, save_path: Optional[str] = None):
    """
    Plot forecast results for all models on the same chart.
    """
    n_models = len(results)
    if n_models == 0:
        print("No models to plot.")
        return

    fig, axes = plt.subplots(n_models, 1, figsize=(14, 5 * n_models), sharex=False)
    if n_models == 1:
        axes = [axes]

    for ax, (name, result) in zip(axes, results.items()):
        if "test" in result:
            test = result["test"]
            preds = result["predictions"]
            ax.plot(range(len(test)), test.values, "b-", label="Actual", linewidth=2)
            ax.plot(range(len(preds)), preds if isinstance(preds, np.ndarray) else preds.values,
                    "r--", label="Predicted", linewidth=2)
        elif "y_test" in result:
            y_test = result["y_test"]
            preds = result["predictions"]
            ax.plot(range(len(y_test)), y_test.values, "b-", label="Actual", linewidth=2)
            ax.plot(range(len(preds)), preds, "r--", label="Predicted", linewidth=2)

        metrics = result["metrics"]
        title = f"{name} — RMSE: {metrics['RMSE']}, MAE: {metrics['MAE']}, MAPE: {metrics['MAPE (%)']}%"
        ax.set_title(title, fontweight="bold")
        ax.legend()
        ax.set_ylabel("Occupancy Rate")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time Steps")
    plt.suptitle(f"Demand Forecasting Comparison — Lot: {lot_id}", fontweight="bold", fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_metrics_comparison(results: Dict, save_path: Optional[str] = None):
    """Bar chart comparing metrics across all forecasting models."""
    metrics_df = pd.DataFrame({name: res["metrics"] for name, res in results.items()}).T

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["RMSE", "MAE", "MAPE (%)"]):
        colors = plt.cm.Set2(np.linspace(0, 0.8, len(metrics_df)))
        ax.bar(metrics_df.index, metrics_df[metric], color=colors, edgecolor="white")
        ax.set_title(metric, fontweight="bold")
        ax.set_ylabel(metric)
        for i, v in enumerate(metrics_df[metric]):
            ax.text(i, v + 0.01 * max(metrics_df[metric]), f"{v:.3f}",
                    ha="center", fontsize=10, fontweight="bold")

    plt.suptitle("Forecasting Model Comparison", fontweight="bold", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
