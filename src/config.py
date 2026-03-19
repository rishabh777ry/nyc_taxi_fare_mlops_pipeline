"""
Shared configuration for the NYC Taxi Fare MLOps pipeline.

Centralizes all environment variables, paths, and constants.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, PREDICTIONS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─── NYC TLC Dataset ──────────────────────────────────────────────
NYC_TLC_BASE_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
)
DEFAULT_YEAR = 2023
DEFAULT_MONTH = 1

# ─── NYC Bounding Box (for validation) ────────────────────────────
NYC_LAT_MIN, NYC_LAT_MAX = 40.4774, 40.9176
NYC_LON_MIN, NYC_LON_MAX = -74.2591, -73.7004

# ─── Validation Thresholds ────────────────────────────────────────
MAX_FARE_AMOUNT = 500.0
MIN_FARE_AMOUNT = 0.0
MAX_TRIP_DISTANCE = 200.0  # miles
MAX_NULL_FRACTION = 0.3  # fail if >30% nulls in any critical column

# ─── Expected Schema ──────────────────────────────────────────────
EXPECTED_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "fare_amount",
]

# Columns that are OK to derive from location IDs if lat/lon missing
LOCATION_ID_COLUMNS = ["PULocationID", "DOLocationID"]

# ─── Feature Engineering ──────────────────────────────────────────
FEATURE_COLUMNS = [
    "pickup_hour",
    "pickup_day",
    "pickup_weekday",
    "pickup_month",
    "is_weekend",
    "trip_distance",
    "haversine_distance",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
]
TARGET_COLUMN = "fare_amount"

# ─── MLflow ───────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "nyc-taxi-fare")
MLFLOW_MODEL_NAME = "nyc-taxi-fare-model"

# ─── MinIO / S3 ──────────────────────────────────────────────────
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "mlops_admin")
MINIO_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "mlops_secret_key")
MINIO_BUCKET = os.getenv("MINIO_BUCKET_NAME", "mlflow-artifacts")

# ─── PostgreSQL ───────────────────────────────────────────────────
POSTGRES_USER = os.getenv("POSTGRES_USER", "mlflow")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "mlflow_secret")
POSTGRES_DB = os.getenv("POSTGRES_DB", "mlflow_db")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

# ─── API ──────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))


@dataclass
class TrainingConfig:
    """Configuration for model training runs."""

    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 3
    models: list[str] = field(
        default_factory=lambda: ["linear_regression", "random_forest", "xgboost"]
    )
    # Hyperparameter grids for tuning
    rf_param_grid: dict = field(
        default_factory=lambda: {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        }
    )
    xgb_param_grid: dict = field(
        default_factory=lambda: {
            "n_estimators": [100, 200],
            "max_depth": [5, 10],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
        }
    )
