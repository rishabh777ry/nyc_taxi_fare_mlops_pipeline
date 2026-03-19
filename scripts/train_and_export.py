"""
Train a model on real NYC Taxi data and save it for deployment.

This script:
1. Downloads real NYC TLC Yellow Taxi data
2. Validates and cleans the data
3. Engineers features (haversine, datetime, etc.)
4. Trains multiple models (LinearRegression, RandomForest, XGBoost)
5. Saves the best model + scaler as .joblib files
6. Prints evaluation metrics

Usage:
    python scripts/train_and_export.py
"""

from __future__ import annotations

import os
import sys
import json
import math
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "raw"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# NYC bounding box
NYC_LAT_MIN, NYC_LAT_MAX = 40.4774, 40.9176
NYC_LON_MIN, NYC_LON_MAX = -74.2591, -73.7004


def download_data(year: int = 2023, month: int = 1) -> pd.DataFrame:
    """Download NYC TLC Yellow Taxi Parquet data."""
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    filepath = DATA_DIR / f"yellow_tripdata_{year}-{month:02d}.parquet"

    if filepath.exists():
        logger.info("Using cached data: %s", filepath)
        return pd.read_parquet(filepath)

    logger.info("Downloading NYC taxi data from: %s", url)
    df = pd.read_parquet(url)
    df.to_parquet(filepath)
    logger.info("Downloaded %d rows.", len(df))
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the taxi data."""
    original_len = len(df)
    logger.info("Cleaning data (%d rows)...", original_len)

    required_cols = ["tpep_pickup_datetime", "trip_distance", "fare_amount"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Filter fare amount
    df = df[(df["fare_amount"] >= 2.5) & (df["fare_amount"] <= 200)]

    # Filter trip distance
    df = df[(df["trip_distance"] > 0.1) & (df["trip_distance"] <= 100)]

    # Filter passenger count
    if "passenger_count" in df.columns:
        df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 9)]

    # Drop nulls in key columns
    df = df.dropna(subset=["fare_amount", "trip_distance", "tpep_pickup_datetime"])

    logger.info("Cleaned: %d → %d rows (removed %d).", original_len, len(df), original_len - len(df))
    return df


def haversine_vectorized(lat1, lon1, lat2, lon2):
    """Compute haversine distance in miles."""
    R = 3958.8
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ML features from raw data."""
    logger.info("Engineering features...")

    # Datetime features
    dt = pd.to_datetime(df["tpep_pickup_datetime"])
    df = df.copy()
    df["pickup_hour"] = dt.dt.hour
    df["pickup_day"] = dt.dt.day
    df["pickup_weekday"] = dt.dt.weekday
    df["pickup_month"] = dt.dt.month
    df["is_weekend"] = (dt.dt.weekday >= 5).astype(int)

    # Haversine distance (use lat/lon if available, else trip_distance)
    if all(c in df.columns for c in ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]):
        df["haversine_distance"] = haversine_vectorized(
            df["pickup_latitude"], df["pickup_longitude"],
            df["dropoff_latitude"], df["dropoff_longitude"],
        )
    else:
        df["haversine_distance"] = df["trip_distance"]

    # Location IDs
    if "PULocationID" not in df.columns:
        df["PULocationID"] = 0
    if "DOLocationID" not in df.columns:
        df["DOLocationID"] = 0

    # Passenger count
    if "passenger_count" not in df.columns:
        df["passenger_count"] = 1

    return df


def train_models(df: pd.DataFrame, sample_size: int = 50000) -> dict:
    """Train multiple models and return the best one."""
    feature_cols = [
        "pickup_hour", "pickup_weekday", "pickup_month", "is_weekend",
        "trip_distance", "haversine_distance", "passenger_count",
        "PULocationID", "DOLocationID",
    ]
    target_col = "fare_amount"

    # Sample for speed
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
        logger.info("Sampled to %d rows for training.", sample_size)

    X = df[feature_cols].fillna(0)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info("Training on %d samples, testing on %d...", len(X_train), len(X_test))

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42,
        ),
    }

    results = {}
    best_model = None
    best_rmse = float("inf")
    best_name = ""

    for name, model in models.items():
        logger.info("Training %s...", name)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        results[name] = {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}
        logger.info("  %s → RMSE: %.4f, MAE: %.4f, R²: %.4f", name, rmse, mae, r2)

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_name = name

    logger.info("Best model: %s (RMSE: %.4f)", best_name, best_rmse)

    return {
        "best_model": best_model,
        "best_name": best_name,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "results": results,
    }


def save_artifacts(training_output: dict) -> None:
    """Save model, scaler, and metadata to disk."""
    # Save model
    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(training_output["best_model"], model_path)
    logger.info("Saved model: %s", model_path)

    # Save scaler
    scaler_path = MODEL_DIR / "scaler.joblib"
    joblib.dump(training_output["scaler"], scaler_path)
    logger.info("Saved scaler: %s", scaler_path)

    # Save metadata
    metadata = {
        "best_model": training_output["best_name"],
        "feature_columns": training_output["feature_cols"],
        "metrics": training_output["results"],
    }
    metadata_path = MODEL_DIR / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata: %s", metadata_path)


def main():
    """Full training pipeline."""
    logger.info("=" * 60)
    logger.info("NYC Taxi Fare Model Training Pipeline")
    logger.info("=" * 60)

    # 1. Download data
    df = download_data(year=2023, month=1)

    # 2. Clean data
    df = clean_data(df)

    # 3. Engineer features
    df = engineer_features(df)

    # 4. Train models
    output = train_models(df, sample_size=50000)

    # 5. Save artifacts
    save_artifacts(output)

    # 6. Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model: {output['best_name']}")
    print(f"\nAll results:")
    for name, metrics in output["results"].items():
        print(f"  {name}: RMSE={metrics['rmse']}, MAE={metrics['mae']}, R²={metrics['r2']}")
    print(f"\nArtifacts saved to: {MODEL_DIR}/")
    print(f"  - model.joblib")
    print(f"  - scaler.joblib")
    print(f"  - metadata.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
