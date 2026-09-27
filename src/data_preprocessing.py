"""
Data Preprocessing & Feature Engineering Module
================================================

Handles data loading, cleaning, and comprehensive feature engineering
for the ParkWise dynamic pricing system.

Features engineered:
- Temporal: hour, day_of_week, month, is_weekend + cyclical encodings
- Demand: occupancy_rate, lag features, rolling statistics
- Contextual: traffic_numeric, vehicle_weight, is_special_day
- Synthetic: simulated base prices and demand scores for ML training
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

TRAFFIC_MAP = {"low": 1, "average": 2, "high": 3}
VEHICLE_WEIGHTS = {"car": 1.0, "bike": 0.5, "cycle": 0.3, "truck": 1.5}
BASE_PRICE = 10.0


# ─────────────────────────────────────────────────────────────────────
# Data Loading & Cleaning
# ─────────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the parking dataset and perform basic cleaning.

    Parameters
    ----------
    filepath : str
        Path to the CSV dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with parsed timestamps.
    """
    df = pd.read_csv(filepath)

    # Parse timestamp
    df["Timestamp"] = pd.to_datetime(
        df["LastUpdatedDate"] + " " + df["LastUpdatedTime"],
        format="%d-%m-%Y %H:%M:%S",
    )
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Map categorical features to numeric
    df["Traffic"] = df["TrafficConditionNearby"].map(TRAFFIC_MAP)
    df["VehicleWeight"] = df["VehicleType"].map(VEHICLE_WEIGHTS)

    # Compute occupancy rate (core metric)
    df["OccupancyRate"] = df["Occupancy"] / df["Capacity"].replace(0, 1)
    df["OccupancyRate"] = df["OccupancyRate"].clip(0, 1)

    return df


# ─────────────────────────────────────────────────────────────────────
# Temporal Feature Engineering
# ─────────────────────────────────────────────────────────────────────

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from the Timestamp column.

    Creates: Hour, DayOfWeek, Month, IsWeekend, DayName
    Plus cyclical sine/cosine encodings for hour and day_of_week.
    """
    df = df.copy()
    df["Hour"] = df["Timestamp"].dt.hour
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
    df["Month"] = df["Timestamp"].dt.month
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["DayName"] = df["Timestamp"].dt.day_name()

    # Cyclical encoding — preserves the circular nature of time
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

    return df


# ─────────────────────────────────────────────────────────────────────
# Lag & Rolling Features
# ─────────────────────────────────────────────────────────────────────

def create_lag_features(
    df: pd.DataFrame,
    target_col: str = "OccupancyRate",
    lags: List[int] = [1, 2, 3, 6],
    group_col: str = "SystemCodeNumber",
) -> pd.DataFrame:
    """
    Create lag features for each parking lot.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame sorted by Timestamp.
    target_col : str
        Column to create lags for.
    lags : list of int
        Lag periods to create.
    group_col : str
        Column to group by (parking lot ID).
    """
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df.groupby(group_col)[target_col].shift(lag)
    return df


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = "OccupancyRate",
    windows: List[int] = [3, 6, 12],
    group_col: str = "SystemCodeNumber",
) -> pd.DataFrame:
    """
    Create rolling mean and std features for each parking lot.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame sorted by Timestamp.
    target_col : str
        Column to compute rolling stats on.
    windows : list of int
        Rolling window sizes.
    group_col : str
        Column to group by (parking lot ID).
    """
    df = df.copy()
    for window in windows:
        rolling = df.groupby(group_col)[target_col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"{target_col}_rolling_mean_{window}"] = rolling

        rolling_std = df.groupby(group_col)[target_col].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
        df[f"{target_col}_rolling_std_{window}"] = rolling_std.fillna(0)

    return df


# ─────────────────────────────────────────────────────────────────────
# Synthetic Price Target Generation
# ─────────────────────────────────────────────────────────────────────

def create_synthetic_price(df: pd.DataFrame, noise_std: float = 0.5) -> pd.DataFrame:
    """
    Generate a synthetic 'optimal price' target for ML training.

    The price is computed using a sophisticated multi-factor formula
    that considers occupancy, queue, traffic, events, vehicle type,
    and temporal patterns. Gaussian noise is added for realism.

    This simulates what a real pricing oracle would produce, allowing
    ML models to learn the underlying pricing logic.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with engineered features.
    noise_std : float
        Standard deviation of Gaussian noise to add.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'Price' column added.
    """
    df = df.copy()

    # Multi-factor pricing formula
    occupancy_factor = 1.0 + 2.5 * (df["OccupancyRate"] ** 1.5)
    queue_factor = 1.0 + 0.15 * df["QueueLength"]
    traffic_factor = 1.0 + 0.2 * (df["Traffic"] - 1)
    special_day_factor = 1.0 + 0.5 * df["IsSpecialDay"]
    vehicle_factor = df["VehicleWeight"]

    # Time-of-day surge: peak hours (8-10am, 5-7pm) get premium
    if "Hour" in df.columns:
        peak_morning = np.exp(-0.5 * ((df["Hour"] - 9) / 2) ** 2) * 0.3
        peak_evening = np.exp(-0.5 * ((df["Hour"] - 18) / 2) ** 2) * 0.3
        time_surge = 1.0 + peak_morning + peak_evening
    else:
        time_surge = 1.0

    # Weekend premium
    if "IsWeekend" in df.columns:
        weekend_factor = 1.0 + 0.15 * df["IsWeekend"]
    else:
        weekend_factor = 1.0

    # Compute synthetic price
    price = (
        BASE_PRICE
        * occupancy_factor
        * queue_factor
        * traffic_factor
        * special_day_factor
        * vehicle_factor
        * time_surge
        * weekend_factor
    )

    # Add realistic noise
    np.random.seed(42)
    noise = np.random.normal(0, noise_std, len(df))
    price = (price + noise).clip(2.0, 100.0)  # Reasonable price bounds

    df["Price"] = np.round(price, 2)
    return df


# ─────────────────────────────────────────────────────────────────────
# Demand Score
# ─────────────────────────────────────────────────────────────────────

def create_demand_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a composite demand score (0-100) for clustering and analysis.

    Combines occupancy rate, queue length, traffic, and special day
    into a single demand metric.
    """
    df = df.copy()

    # Normalize each component to 0-1
    occ_norm = df["OccupancyRate"]
    queue_norm = df["QueueLength"] / df["QueueLength"].max()
    traffic_norm = (df["Traffic"] - 1) / 2  # Maps 1-3 to 0-1
    special = df["IsSpecialDay"]

    # Weighted composite score
    df["DemandScore"] = (
        0.50 * occ_norm
        + 0.25 * queue_norm
        + 0.15 * traffic_norm
        + 0.10 * special
    ) * 100

    return df


# ─────────────────────────────────────────────────────────────────────
# Full Pipeline
# ─────────────────────────────────────────────────────────────────────

def full_preprocessing_pipeline(filepath: str) -> pd.DataFrame:
    """
    Run the complete preprocessing and feature engineering pipeline.

    Parameters
    ----------
    filepath : str
        Path to the CSV dataset.

    Returns
    -------
    pd.DataFrame
        Fully preprocessed DataFrame with all engineered features.
    """
    df = load_data(filepath)
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_synthetic_price(df)
    df = create_demand_score(df)

    return df


def get_ml_features() -> List[str]:
    """Return the list of feature column names used for ML models."""
    return [
        "OccupancyRate",
        "QueueLength",
        "Traffic",
        "IsSpecialDay",
        "VehicleWeight",
        "Hour",
        "DayOfWeek",
        "IsWeekend",
        "Hour_sin",
        "Hour_cos",
        "DayOfWeek_sin",
        "DayOfWeek_cos",
        "OccupancyRate_lag_1",
        "OccupancyRate_lag_2",
        "OccupancyRate_lag_3",
        "OccupancyRate_rolling_mean_3",
        "OccupancyRate_rolling_mean_6",
        "OccupancyRate_rolling_std_3",
    ]
