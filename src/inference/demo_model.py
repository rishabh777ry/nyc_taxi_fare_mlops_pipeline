"""
Demo model for standalone deployment without MLflow.

Uses a simple fare estimation formula based on NYC taxi pricing rules,
enhanced with a lightweight sklearn model trained on synthetic data.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

logger = logging.getLogger(__name__)

# ─── Singleton model instance ─────────────────────────────────────
_demo_model = None


def _generate_training_data(n: int = 5000) -> tuple[pd.DataFrame, pd.Series]:
    """Generate realistic synthetic NYC taxi trip data for demo model."""
    np.random.seed(42)

    # Pickup locations (Manhattan-centric)
    pu_lat = np.random.normal(40.75, 0.03, n)
    pu_lon = np.random.normal(-73.98, 0.02, n)
    do_lat = np.random.normal(40.75, 0.04, n)
    do_lon = np.random.normal(-73.98, 0.03, n)

    # Haversine distance
    R = 3958.8
    dlat = np.radians(do_lat - pu_lat)
    dlon = np.radians(do_lon - pu_lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(pu_lat)) * np.cos(np.radians(do_lat)) * np.sin(dlon / 2) ** 2
    distances = R * 2 * np.arcsin(np.sqrt(a))

    hours = np.random.randint(0, 24, n)
    weekdays = np.random.randint(0, 7, n)
    passengers = np.random.randint(1, 5, n)

    # Realistic fare = base + per-mile + time-based + noise
    base_fare = 3.00
    per_mile = 2.50
    night_surcharge = np.where((hours >= 20) | (hours < 6), 1.0, 0.0)
    rush_surcharge = np.where(((hours >= 7) & (hours <= 10)) | ((hours >= 16) & (hours <= 19)), 1.5, 0.0)
    weekend_surcharge = np.where(weekdays >= 5, 0.5, 0.0)
    noise = np.random.normal(0, 1.5, n)

    fares = base_fare + per_mile * distances + night_surcharge + rush_surcharge + weekend_surcharge + noise
    fares = np.maximum(fares, 2.50)  # Minimum fare

    X = pd.DataFrame({
        "pickup_hour": hours,
        "pickup_weekday": weekdays,
        "is_weekend": (weekdays >= 5).astype(int),
        "haversine_distance": distances,
        "trip_distance": distances * np.random.uniform(0.9, 1.3, n),  # Add road distance factor
        "passenger_count": passengers,
    })

    return X, pd.Series(fares)


def get_demo_model() -> GradientBoostingRegressor:
    """Get or create the demo model (trained on synthetic data)."""
    global _demo_model

    if _demo_model is not None:
        return _demo_model

    logger.info("Training demo model on synthetic data...")
    X, y = _generate_training_data(5000)

    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X, y)

    _demo_model = model
    logger.info("Demo model trained successfully.")
    return model


def predict_demo(
    pickup_latitude: float,
    pickup_longitude: float,
    dropoff_latitude: float,
    dropoff_longitude: float,
    pickup_hour: int,
    pickup_weekday: int,
    passenger_count: int = 1,
    trip_distance: float | None = None,
) -> float:
    """
    Predict fare using the demo model.

    Returns:
        Predicted fare in USD.
    """
    # Compute haversine distance
    R = 3958.8
    dlat = math.radians(dropoff_latitude - pickup_latitude)
    dlon = math.radians(dropoff_longitude - pickup_longitude)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(pickup_latitude)) * math.cos(math.radians(dropoff_latitude)) * math.sin(dlon / 2) ** 2
    haversine_dist = R * 2 * math.asin(math.sqrt(a))

    if trip_distance is None:
        trip_distance = haversine_dist * 1.15  # Road distance ~15% more than straight line

    is_weekend = 1 if pickup_weekday >= 5 else 0

    features = pd.DataFrame([{
        "pickup_hour": pickup_hour,
        "pickup_weekday": pickup_weekday,
        "is_weekend": is_weekend,
        "haversine_distance": haversine_dist,
        "trip_distance": trip_distance,
        "passenger_count": passenger_count,
    }])

    model = get_demo_model()
    prediction = float(model.predict(features)[0])

    # Clamp to reasonable range
    return max(2.50, round(prediction, 2))
