"""
Feature engineering module for NYC Taxi Fare MLOps pipeline.

Extracts temporal features, computes haversine distance, encodes
categoricals, and normalizes numerics.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURE_COLUMNS,
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
    TrainingConfig,
)

logger = logging.getLogger(__name__)


def haversine_distance(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    """
    Calculate the Haversine distance in miles between two points.

    Uses vectorized Pandas operations for performance on large datasets.
    """
    R = 3958.8  # Earth radius in miles

    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from pickup datetime.

    Features: hour, day of month, weekday (0=Mon..6=Sun), month, is_weekend.
    """
    dt = pd.to_datetime(df["tpep_pickup_datetime"])

    df = df.copy()
    df["pickup_hour"] = dt.dt.hour
    df["pickup_day"] = dt.dt.day
    df["pickup_weekday"] = dt.dt.weekday  # 0=Monday
    df["pickup_month"] = dt.dt.month
    df["is_weekend"] = (dt.dt.weekday >= 5).astype(int)

    logger.info("Extracted datetime features: hour, day, weekday, month, is_weekend.")
    return df


def compute_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute haversine distance from lat/lon if available.

    Falls back to trip_distance alone if lat/lon are missing.
    """
    df = df.copy()

    if all(c in df.columns for c in ["pickup_latitude", "pickup_longitude",
                                      "dropoff_latitude", "dropoff_longitude"]):
        df["haversine_distance"] = haversine_distance(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["dropoff_latitude"],
            df["dropoff_longitude"],
        )
        # Replace NaN / inf with 0
        df["haversine_distance"] = df["haversine_distance"].replace([np.inf, -np.inf], 0).fillna(0)
        logger.info("Computed haversine_distance from lat/lon.")
    else:
        # Use trip_distance as proxy
        df["haversine_distance"] = df["trip_distance"].fillna(0)
        logger.info("Lat/lon not available; using trip_distance as haversine_distance proxy.")

    return df


def prepare_location_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure PULocationID and DOLocationID exist.

    For older data with lat/lon, bin coordinates into integer IDs.
    For newer data, these columns already exist.
    """
    df = df.copy()

    if "PULocationID" not in df.columns:
        if "pickup_latitude" in df.columns:
            # Simple binning: 100 bins per axis
            df["PULocationID"] = (
                pd.cut(df["pickup_latitude"], bins=100, labels=False).fillna(0).astype(int)
            )
        else:
            df["PULocationID"] = 0

    if "DOLocationID" not in df.columns:
        if "dropoff_latitude" in df.columns:
            df["DOLocationID"] = (
                pd.cut(df["dropoff_latitude"], bins=100, labels=False).fillna(0).astype(int)
            )
        else:
            df["DOLocationID"] = 0

    return df


def normalize_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Normalize numerical features using StandardScaler.

    The scaler is fit on the training set only to prevent data leakage.
    Optionally persists the scaler as a pickle artifact.
    """
    scaler = StandardScaler()

    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    logger.info("Normalizing %d numerical features.", len(numerical_cols))

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

    if scaler_path:
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        logger.info("Scaler saved to %s", scaler_path)

    return X_train_scaled, X_test_scaled, scaler


def build_features(
    df: pd.DataFrame,
    config: TrainingConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Full feature engineering pipeline.

    1. Extract datetime features.
    2. Compute distance features.
    3. Prepare location IDs.
    4. Select feature columns + target.
    5. Train/test split.
    6. Normalize.

    Returns:
        (X_train, X_test, y_train, y_test, scaler)
    """
    if config is None:
        config = TrainingConfig()

    logger.info("Starting feature engineering on %d rows ...", len(df))

    # Step 1: Datetime features
    df = extract_datetime_features(df)

    # Step 2: Distance features
    df = compute_distance_features(df)

    # Step 3: Location IDs
    df = prepare_location_ids(df)

    # Step 4: Ensure passenger_count exists
    if "passenger_count" not in df.columns:
        df["passenger_count"] = 1

    # Step 5: Select features & target
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not available_features:
        raise ValueError(f"No feature columns found in DataFrame. Available: {list(df.columns)}")

    X = df[available_features].copy()
    y = df[TARGET_COLUMN].copy()

    # Handle any remaining NaN
    X = X.fillna(0)
    y = y.fillna(y.median())

    logger.info("Features selected: %s", available_features)
    logger.info("Feature matrix shape: %s, Target shape: %s", X.shape, y.shape)

    # Step 6: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    # Step 7: Normalize
    scaler_path = PROCESSED_DATA_DIR / "scaler.pkl"
    X_train, X_test, scaler = normalize_features(X_train, X_test, scaler_path=scaler_path)

    logger.info(
        "Feature engineering complete. Train: %d, Test: %d", len(X_train), len(X_test)
    )
    return X_train, X_test, y_train, y_test, scaler
