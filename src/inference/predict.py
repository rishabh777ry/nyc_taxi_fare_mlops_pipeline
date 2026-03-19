"""
Inference module for NYC Taxi Fare MLOps pipeline.

Loads the Production model from MLflow Model Registry and runs
predictions on new data.
"""

from __future__ import annotations

import logging
import pickle
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.config import (
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    PROCESSED_DATA_DIR,
)
from src.features.engineer import (
    compute_distance_features,
    extract_datetime_features,
    haversine_distance,
    prepare_location_ids,
)

logger = logging.getLogger(__name__)

# ─── Cached model & scaler ────────────────────────────────────────
_model_cache: dict = {"model": None, "version": None, "scaler": None}


class ModelNotFoundError(Exception):
    """Raised when no Production model is found in the registry."""


def load_production_model(force_reload: bool = False):
    """
    Load the latest Production model from MLflow Model Registry.

    Uses an in-memory cache to avoid reloading on every request.

    Args:
        force_reload: Force reload even if cached.

    Returns:
        The loaded model.

    Raises:
        ModelNotFoundError: If no model is in Production stage.
    """
    if _model_cache["model"] is not None and not force_reload:
        return _model_cache["model"]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/Production"
        model = mlflow.pyfunc.load_model(model_uri)
        _model_cache["model"] = model
        logger.info("Loaded Production model: %s", model_uri)
    except Exception as exc:
        raise ModelNotFoundError(
            f"No Production model found for '{MLFLOW_MODEL_NAME}'. "
            "Run the training pipeline first."
        ) from exc

    return model


def load_scaler(scaler_path: Path | None = None):
    """Load the fitted StandardScaler from disk."""
    if _model_cache["scaler"] is not None:
        return _model_cache["scaler"]

    if scaler_path is None:
        scaler_path = PROCESSED_DATA_DIR / "scaler.pkl"

    if not scaler_path.exists():
        logger.warning("Scaler not found at %s. Skipping normalization.", scaler_path)
        return None

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    _model_cache["scaler"] = scaler
    logger.info("Loaded scaler from %s", scaler_path)
    return scaler


def prepare_single_input(
    pickup_latitude: float,
    pickup_longitude: float,
    dropoff_latitude: float,
    dropoff_longitude: float,
    pickup_datetime: str | datetime,
    passenger_count: int = 1,
    trip_distance: float | None = None,
) -> pd.DataFrame:
    """
    Prepare a single prediction input into a feature DataFrame.

    This applies the same feature engineering pipeline used during training.

    Args:
        pickup_latitude: Pickup latitude.
        pickup_longitude: Pickup longitude.
        dropoff_latitude: Dropoff latitude.
        dropoff_longitude: Dropoff longitude.
        pickup_datetime: Pickup datetime (string or datetime object).
        passenger_count: Number of passengers.
        trip_distance: Trip distance in miles (auto-computed if None).

    Returns:
        Feature DataFrame ready for model prediction.
    """
    if isinstance(pickup_datetime, str):
        pickup_datetime = pd.to_datetime(pickup_datetime)

    # Compute haversine distance if trip_distance not provided
    if trip_distance is None:
        h_dist = haversine_distance(
            pd.Series([pickup_latitude]),
            pd.Series([pickup_longitude]),
            pd.Series([dropoff_latitude]),
            pd.Series([dropoff_longitude]),
        ).iloc[0]
        trip_distance = float(h_dist)

    # Build raw input DataFrame
    raw = pd.DataFrame([{
        "tpep_pickup_datetime": pickup_datetime,
        "pickup_latitude": pickup_latitude,
        "pickup_longitude": pickup_longitude,
        "dropoff_latitude": dropoff_latitude,
        "dropoff_longitude": dropoff_longitude,
        "passenger_count": passenger_count,
        "trip_distance": trip_distance,
    }])

    # Apply feature engineering
    raw = extract_datetime_features(raw)
    raw = compute_distance_features(raw)
    raw = prepare_location_ids(raw)

    # Select same feature columns used in training
    from src.config import FEATURE_COLUMNS
    available = [c for c in FEATURE_COLUMNS if c in raw.columns]
    features = raw[available].fillna(0)

    # Apply scaler if available
    scaler = load_scaler()
    if scaler is not None:
        try:
            numerical_cols = features.select_dtypes(include=[np.number]).columns.tolist()
            features[numerical_cols] = scaler.transform(features[numerical_cols])
        except Exception as e:
            logger.warning("Scaler transform failed: %s. Using unscaled features.", e)

    return features


def predict_single(
    pickup_latitude: float,
    pickup_longitude: float,
    dropoff_latitude: float,
    dropoff_longitude: float,
    pickup_datetime: str | datetime,
    passenger_count: int = 1,
    trip_distance: float | None = None,
) -> float:
    """
    Predict fare for a single trip.

    Returns:
        Predicted fare amount in dollars.
    """
    model = load_production_model()
    features = prepare_single_input(
        pickup_latitude, pickup_longitude,
        dropoff_latitude, dropoff_longitude,
        pickup_datetime, passenger_count, trip_distance,
    )

    prediction = model.predict(features)
    fare = float(prediction[0])

    # Clamp to reasonable range
    fare = max(0.0, fare)

    logger.info("Predicted fare: $%.2f", fare)
    return fare


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run batch inference on a DataFrame.

    The DataFrame must contain the raw columns needed for feature engineering.

    Returns:
        Input DataFrame with an added 'predicted_fare' column.
    """
    model = load_production_model()

    # Apply feature pipeline
    df_feat = extract_datetime_features(df.copy())
    df_feat = compute_distance_features(df_feat)
    df_feat = prepare_location_ids(df_feat)

    from src.config import FEATURE_COLUMNS
    available = [c for c in FEATURE_COLUMNS if c in df_feat.columns]
    features = df_feat[available].fillna(0)

    scaler = load_scaler()
    if scaler is not None:
        try:
            numerical_cols = features.select_dtypes(include=[np.number]).columns.tolist()
            features[numerical_cols] = scaler.transform(features[numerical_cols])
        except Exception:
            pass

    predictions = model.predict(features)
    df = df.copy()
    df["predicted_fare"] = np.maximum(predictions, 0.0)

    logger.info("Batch prediction complete. %d rows processed.", len(df))
    return df
