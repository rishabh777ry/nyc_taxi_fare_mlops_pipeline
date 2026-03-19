"""
Trained model loader for deployment.

Loads the trained model, scaler, and metadata from the models/ directory.
This is used when deploying without MLflow (e.g., on Render.com).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# ─── Singleton cache ──────────────────────────────────────────────
_cache = {"model": None, "scaler": None, "metadata": None}


def load_trained_model():
    """Load the trained model and scaler from disk."""
    if _cache["model"] is not None:
        return _cache["model"], _cache["scaler"], _cache["metadata"]

    model_path = MODEL_DIR / "model.joblib"
    scaler_path = MODEL_DIR / "scaler.joblib"
    metadata_path = MODEL_DIR / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Run 'python scripts/train_and_export.py' first."
        )

    logger.info("Loading trained model from %s", model_path)
    _cache["model"] = joblib.load(model_path)
    _cache["scaler"] = joblib.load(scaler_path) if scaler_path.exists() else None
    _cache["metadata"] = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    logger.info(
        "Loaded model: %s (features: %s)",
        _cache["metadata"].get("best_model", "unknown"),
        _cache["metadata"].get("feature_columns", []),
    )

    return _cache["model"], _cache["scaler"], _cache["metadata"]


def predict_fare(
    pickup_latitude: float,
    pickup_longitude: float,
    dropoff_latitude: float,
    dropoff_longitude: float,
    pickup_hour: int,
    pickup_weekday: int,
    pickup_month: int = 1,
    is_weekend: int = 0,
    passenger_count: int = 1,
    trip_distance: float | None = None,
) -> float:
    """
    Predict fare using the trained model.

    Returns:
        Predicted fare in USD.
    """
    model, scaler, metadata = load_trained_model()

    # Compute haversine distance
    R = 3958.8
    dlat = math.radians(dropoff_latitude - pickup_latitude)
    dlon = math.radians(dropoff_longitude - pickup_longitude)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(pickup_latitude)) * math.cos(math.radians(dropoff_latitude)) * math.sin(dlon / 2) ** 2
    haversine_dist = R * 2 * math.asin(math.sqrt(a))

    if trip_distance is None:
        trip_distance = haversine_dist * 1.15

    features = pd.DataFrame([{
        "pickup_hour": pickup_hour,
        "pickup_weekday": pickup_weekday,
        "pickup_month": pickup_month,
        "is_weekend": is_weekend,
        "trip_distance": trip_distance,
        "haversine_distance": haversine_dist,
        "passenger_count": passenger_count,
        "PULocationID": 0,
        "DOLocationID": 0,
    }])

    # Ensure same column order as training
    feature_cols = metadata.get("feature_columns", list(features.columns))
    features = features[feature_cols]

    if scaler is not None:
        features = scaler.transform(features)

    prediction = float(model.predict(features)[0])
    return max(2.50, round(prediction, 2))
