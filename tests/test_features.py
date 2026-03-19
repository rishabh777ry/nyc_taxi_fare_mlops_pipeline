"""
Unit tests for feature engineering module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import (
    compute_distance_features,
    extract_datetime_features,
    haversine_distance,
    prepare_location_ids,
)


class TestHaversineDistance:
    """Tests for haversine distance calculation."""

    def test_same_point_returns_zero(self):
        """Distance between same point should be ~0."""
        dist = haversine_distance(
            pd.Series([40.7128]), pd.Series([-74.0060]),
            pd.Series([40.7128]), pd.Series([-74.0060]),
        )
        assert abs(dist.iloc[0]) < 0.01

    def test_known_distance(self):
        """Test known distance: Times Square to Central Park (~1.3 mi)."""
        dist = haversine_distance(
            pd.Series([40.7580]), pd.Series([-73.9855]),
            pd.Series([40.7829]), pd.Series([-73.9654]),
        )
        # Approximately 1.3 miles (allow tolerance)
        assert 1.0 < dist.iloc[0] < 2.5

    def test_vectorized_multiple_points(self):
        """Should handle multiple points in a vectorized manner."""
        n = 100
        lat1 = pd.Series(np.random.uniform(40.5, 40.9, n))
        lon1 = pd.Series(np.random.uniform(-74.2, -73.7, n))
        lat2 = pd.Series(np.random.uniform(40.5, 40.9, n))
        lon2 = pd.Series(np.random.uniform(-74.2, -73.7, n))

        dist = haversine_distance(lat1, lon1, lat2, lon2)
        assert len(dist) == n
        assert (dist >= 0).all()

    def test_negative_coordinates(self):
        """Should handle negative longitudes correctly."""
        dist = haversine_distance(
            pd.Series([40.7128]), pd.Series([-74.0060]),
            pd.Series([40.7580]), pd.Series([-73.9855]),
        )
        assert dist.iloc[0] > 0


class TestExtractDatetimeFeatures:
    """Tests for datetime feature extraction."""

    def test_extracts_hour(self):
        """Should extract correct hour."""
        df = pd.DataFrame({"tpep_pickup_datetime": ["2023-01-15 14:30:00"]})
        result = extract_datetime_features(df)
        assert result["pickup_hour"].iloc[0] == 14

    def test_extracts_weekday(self):
        """Should extract correct weekday (Sunday=6)."""
        df = pd.DataFrame({"tpep_pickup_datetime": ["2023-01-15 14:30:00"]})  # Sunday
        result = extract_datetime_features(df)
        assert result["pickup_weekday"].iloc[0] == 6  # 0=Monday, 6=Sunday

    def test_is_weekend_flag(self):
        """Should flag weekends correctly."""
        df = pd.DataFrame({
            "tpep_pickup_datetime": [
                "2023-01-16 10:00:00",  # Monday
                "2023-01-15 10:00:00",  # Sunday
                "2023-01-14 10:00:00",  # Saturday
            ]
        })
        result = extract_datetime_features(df)
        assert result["is_weekend"].iloc[0] == 0  # Monday
        assert result["is_weekend"].iloc[1] == 1  # Sunday
        assert result["is_weekend"].iloc[2] == 1  # Saturday

    def test_extracts_month(self):
        """Should extract correct month."""
        df = pd.DataFrame({"tpep_pickup_datetime": ["2023-06-15 14:30:00"]})
        result = extract_datetime_features(df)
        assert result["pickup_month"].iloc[0] == 6

    def test_does_not_modify_original(self):
        """Should not modify the original DataFrame."""
        df = pd.DataFrame({"tpep_pickup_datetime": ["2023-01-15 14:30:00"]})
        original_cols = list(df.columns)
        extract_datetime_features(df)
        assert list(df.columns) == original_cols


class TestComputeDistanceFeatures:
    """Tests for distance feature computation."""

    def test_with_latlon(self):
        """Should compute haversine distance when lat/lon present."""
        df = pd.DataFrame({
            "pickup_latitude": [40.7128],
            "pickup_longitude": [-74.0060],
            "dropoff_latitude": [40.7580],
            "dropoff_longitude": [-73.9855],
            "trip_distance": [3.5],
        })
        result = compute_distance_features(df)
        assert "haversine_distance" in result.columns
        assert result["haversine_distance"].iloc[0] > 0

    def test_without_latlon_uses_trip_distance(self):
        """Should fallback to trip_distance when lat/lon missing."""
        df = pd.DataFrame({"trip_distance": [5.0]})
        result = compute_distance_features(df)
        assert "haversine_distance" in result.columns
        assert result["haversine_distance"].iloc[0] == 5.0


class TestPrepareLocationIds:
    """Tests for location ID preparation."""

    def test_existing_location_ids_preserved(self):
        """Should keep existing PULocationID/DOLocationID."""
        df = pd.DataFrame({"PULocationID": [100], "DOLocationID": [200]})
        result = prepare_location_ids(df)
        assert result["PULocationID"].iloc[0] == 100
        assert result["DOLocationID"].iloc[0] == 200

    def test_creates_location_ids_from_latlon(self):
        """Should create binned location IDs from lat/lon."""
        df = pd.DataFrame({
            "pickup_latitude": [40.75],
            "dropoff_latitude": [40.80],
        })
        result = prepare_location_ids(df)
        assert "PULocationID" in result.columns
        assert "DOLocationID" in result.columns

    def test_creates_zero_ids_when_no_location_data(self):
        """Should default to 0 when no location info is available."""
        df = pd.DataFrame({"col": [1]})
        result = prepare_location_ids(df)
        assert result["PULocationID"].iloc[0] == 0
        assert result["DOLocationID"].iloc[0] == 0
