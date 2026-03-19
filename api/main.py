"""
FastAPI prediction service for NYC Taxi Fare MLOps pipeline.

Endpoints:
  POST /predict     — Predict taxi fare from trip details.
  GET  /health      — Health check.
  GET  /model-info  — Current production model info.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Prometheus Metrics ───────────────────────────────────────────
REQUEST_COUNT = Counter(
    "api_request_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
PREDICTION_ERROR = Counter(
    "api_prediction_errors_total",
    "Total prediction errors",
)


# ─── Pydantic Models ─────────────────────────────────────────────
class PredictionRequest(BaseModel):
    """Input schema for fare prediction."""

    pickup_latitude: float = Field(
        ..., ge=40.0, le=41.5,
        description="Pickup latitude (NYC area)",
        examples=[40.7128],
    )
    pickup_longitude: float = Field(
        ..., ge=-74.5, le=-73.0,
        description="Pickup longitude (NYC area)",
        examples=[-74.0060],
    )
    dropoff_latitude: float = Field(
        ..., ge=40.0, le=41.5,
        description="Dropoff latitude (NYC area)",
        examples=[40.7580],
    )
    dropoff_longitude: float = Field(
        ..., ge=-74.5, le=-73.0,
        description="Dropoff longitude (NYC area)",
        examples=[-73.9855],
    )
    pickup_datetime: str = Field(
        ...,
        description="Pickup datetime in ISO format",
        examples=["2023-01-15T14:30:00"],
    )
    passenger_count: int = Field(
        default=1, ge=1, le=9,
        description="Number of passengers",
    )

    @field_validator("pickup_datetime")
    @classmethod
    def validate_datetime(cls, v):
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid datetime format: {v}. Use ISO format (YYYY-MM-DDTHH:MM:SS).")
        return v


class PredictionResponse(BaseModel):
    """Output schema for fare prediction."""

    predicted_fare: float = Field(..., description="Predicted fare in USD")
    pickup_datetime: str
    trip_distance_miles: float
    model_version: str = "latest"
    prediction_timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model info response."""

    model_name: str
    stage: str
    version: str | None = None
    metrics: dict | None = None


# ─── Application Lifecycle ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load model on startup for faster first request."""
    logger.info("Starting API service ...")
    try:
        from src.inference.predict import load_production_model
        load_production_model()
        logger.info("Production model pre-loaded.")
    except Exception as e:
        logger.warning("Could not pre-load model: %s (will retry on first request)", e)
    yield
    logger.info("Shutting down API service.")


# ─── FastAPI App ──────────────────────────────────────────────────
app = FastAPI(
    title="NYC Taxi Fare Prediction API",
    description="Predict taxi fare amount using ML models trained on NYC TLC data.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Middleware for metrics ───────────────────────────────────────
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Record request count and latency for Prometheus."""
    start = time.time()
    response = await call_next(request)
    latency = time.time() - start

    endpoint = request.url.path
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        status=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)

    return response


# ─── Endpoints ────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    from src.inference.predict import _model_cache

    return HealthResponse(
        status="healthy",
        model_loaded=_model_cache["model"] is not None,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict taxi fare for a given trip.

    Accepts pickup/dropoff coordinates and datetime.
    Returns the predicted fare amount in USD.
    """
    try:
        from src.inference.predict import predict_single, load_production_model
        from src.features.engineer import haversine_distance
        import pandas as pd

        # Ensure model is loaded
        load_production_model()

        # Compute trip distance for response
        trip_distance = float(haversine_distance(
            pd.Series([request.pickup_latitude]),
            pd.Series([request.pickup_longitude]),
            pd.Series([request.dropoff_latitude]),
            pd.Series([request.dropoff_longitude]),
        ).iloc[0])

        # Predict
        fare = predict_single(
            pickup_latitude=request.pickup_latitude,
            pickup_longitude=request.pickup_longitude,
            dropoff_latitude=request.dropoff_latitude,
            dropoff_longitude=request.dropoff_longitude,
            pickup_datetime=request.pickup_datetime,
            passenger_count=request.passenger_count,
            trip_distance=trip_distance,
        )

        return PredictionResponse(
            predicted_fare=round(fare, 2),
            pickup_datetime=request.pickup_datetime,
            trip_distance_miles=round(trip_distance, 2),
            prediction_timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        PREDICTION_ERROR.inc()
        logger.error("Prediction failed: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}. Ensure the model is trained and registered.",
        )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Get information about the current production model."""
    try:
        import mlflow
        from src.config import MLFLOW_MODEL_NAME, MLFLOW_TRACKING_URI

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        # Get latest Production version
        versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
        if not versions:
            return ModelInfoResponse(
                model_name=MLFLOW_MODEL_NAME,
                stage="None",
                version=None,
                metrics=None,
            )

        latest = versions[0]
        run = client.get_run(latest.run_id)

        return ModelInfoResponse(
            model_name=MLFLOW_MODEL_NAME,
            stage="Production",
            version=latest.version,
            metrics=run.data.metrics,
        )

    except Exception as e:
        logger.error("Could not fetch model info: %s", e)
        raise HTTPException(status_code=503, detail=f"MLflow unavailable: {str(e)}")


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
