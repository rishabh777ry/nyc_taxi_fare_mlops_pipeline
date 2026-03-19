"""
Unit tests for FastAPI prediction service.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a FastAPI test client with mocked model."""
    with patch("api.main.load_production_model"):
        from api.main import app
        return TestClient(app)


@pytest.fixture
def valid_prediction_payload():
    """Valid prediction request payload."""
    return {
        "pickup_latitude": 40.7128,
        "pickup_longitude": -74.0060,
        "dropoff_latitude": 40.7580,
        "dropoff_longitude": -73.9855,
        "pickup_datetime": "2023-01-15T14:30:00",
        "passenger_count": 2,
    }


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_health_response_schema(self, client):
        """Health response should match expected schema."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    @patch("api.main.predict_single", return_value=15.50)
    @patch("api.main.load_production_model")
    def test_valid_prediction(self, mock_load, mock_predict, client, valid_prediction_payload):
        """Should return prediction for valid input."""
        response = client.post("/predict", json=valid_prediction_payload)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_fare" in data
        assert isinstance(data["predicted_fare"], float)
        assert data["predicted_fare"] >= 0

    def test_invalid_latitude_rejected(self, client):
        """Should reject coordinates outside valid range."""
        payload = {
            "pickup_latitude": 0.0,  # Invalid: not in NYC
            "pickup_longitude": -74.0060,
            "dropoff_latitude": 40.7580,
            "dropoff_longitude": -73.9855,
            "pickup_datetime": "2023-01-15T14:30:00",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_invalid_datetime_rejected(self, client):
        """Should reject invalid datetime format."""
        payload = {
            "pickup_latitude": 40.7128,
            "pickup_longitude": -74.0060,
            "dropoff_latitude": 40.7580,
            "dropoff_longitude": -73.9855,
            "pickup_datetime": "not-a-date",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_missing_required_fields_rejected(self, client):
        """Should reject request with missing required fields."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_invalid_passenger_count(self, client):
        """Should reject passenger_count > 9 or < 1."""
        payload = {
            "pickup_latitude": 40.7128,
            "pickup_longitude": -74.0060,
            "dropoff_latitude": 40.7580,
            "dropoff_longitude": -73.9855,
            "pickup_datetime": "2023-01-15T14:30:00",
            "passenger_count": 15,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""

    def test_metrics_returns_200(self, client):
        """Metrics endpoint should return valid Prometheus format."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")


class TestModelInfoEndpoint:
    """Tests for /model-info endpoint."""

    @patch("api.main.mlflow")
    def test_model_info_no_model(self, mock_mlflow, client):
        """Should handle case when no model is registered."""
        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []
        mock_mlflow.tracking.MlflowClient.return_value = mock_client

        response = client.get("/model-info")
        # May return 200 with empty info or 503 depending on MLflow availability
        assert response.status_code in [200, 503]
