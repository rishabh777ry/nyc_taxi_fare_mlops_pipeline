"""
Shared test fixtures for NYC Taxi Fare MLOps pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Create a sample raw DataFrame mimicking NYC TLC yellow taxi data."""
    np.random.seed(42)
    n = 100

    return pd.DataFrame({
        "tpep_pickup_datetime": pd.date_range("2023-01-15 08:00", periods=n, freq="15min"),
        "tpep_dropoff_datetime": pd.date_range("2023-01-15 08:20", periods=n, freq="15min"),
        "passenger_count": np.random.randint(1, 5, n),
        "trip_distance": np.abs(np.random.exponential(3.0, n)),
        "pickup_longitude": np.random.uniform(-74.05, -73.75, n),
        "pickup_latitude": np.random.uniform(40.63, 40.85, n),
        "dropoff_longitude": np.random.uniform(-74.05, -73.75, n),
        "dropoff_latitude": np.random.uniform(40.63, 40.85, n),
        "fare_amount": np.abs(np.random.normal(15.0, 8.0, n)),
        "PULocationID": np.random.randint(1, 265, n),
        "DOLocationID": np.random.randint(1, 265, n),
    })


@pytest.fixture
def sample_df_with_nulls(sample_raw_df) -> pd.DataFrame:
    """Sample DataFrame with injected null values."""
    df = sample_raw_df.copy()
    # Inject some nulls
    df.loc[0:4, "fare_amount"] = np.nan
    df.loc[10:14, "trip_distance"] = np.nan
    df.loc[20:24, "passenger_count"] = np.nan
    return df


@pytest.fixture
def sample_df_with_outliers(sample_raw_df) -> pd.DataFrame:
    """Sample DataFrame with injected outlier values."""
    df = sample_raw_df.copy()
    # Extreme fare
    df.loc[0, "fare_amount"] = 9999.0
    # Negative fare
    df.loc[1, "fare_amount"] = -50.0
    # Extreme distance
    df.loc[2, "trip_distance"] = 500.0
    # Zero passenger
    df.loc[3, "passenger_count"] = 0
    # Out-of-bounds coordinates
    df.loc[4, "pickup_latitude"] = 0.0
    df.loc[4, "pickup_longitude"] = 0.0
    return df


@pytest.fixture
def sample_df_bad_schema() -> pd.DataFrame:
    """Sample DataFrame missing required columns."""
    return pd.DataFrame({
        "some_column": [1, 2, 3],
        "another_column": [4, 5, 6],
    })


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    """Sample feature DataFrame after feature engineering."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "pickup_hour": np.random.randint(0, 24, n),
        "pickup_day": np.random.randint(1, 32, n),
        "pickup_weekday": np.random.randint(0, 7, n),
        "pickup_month": np.ones(n, dtype=int),
        "is_weekend": np.random.randint(0, 2, n),
        "trip_distance": np.abs(np.random.exponential(3.0, n)),
        "haversine_distance": np.abs(np.random.exponential(2.5, n)),
        "passenger_count": np.random.randint(1, 5, n),
        "PULocationID": np.random.randint(1, 265, n),
        "DOLocationID": np.random.randint(1, 265, n),
    })
