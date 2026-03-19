"""
FastAPI prediction service for NYC Taxi Fare MLOps pipeline.

Supports two modes:
  - Production: Loads model from MLflow Model Registry.
  - Demo: Uses a standalone sklearn model (for Render/cloud deployment).

Endpoints:
  POST /predict     — Predict taxi fare from trip details.
  GET  /health      — Health check.
  GET  /model-info  — Current production model info.
  GET  /metrics     — Prometheus metrics.
"""

from __future__ import annotations

import logging
import math
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Mode Detection ───────────────────────────────────────────────
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() in ("true", "1", "yes")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "")
USE_DEMO = DEMO_MODE or not MLFLOW_URI

if USE_DEMO:
    logger.info("Running in DEMO mode (standalone model, no MLflow).")
else:
    logger.info("Running in PRODUCTION mode (MLflow: %s).", MLFLOW_URI)


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
        except ValueError as e:
            raise ValueError(f"Invalid datetime format: {v}. Use ISO format (YYYY-MM-DDTHH:MM:SS).") from e
        return v


class PredictionResponse(BaseModel):
    """Output schema for fare prediction."""

    predicted_fare: float = Field(..., description="Predicted fare in USD")
    pickup_datetime: str
    trip_distance_miles: float
    model_version: str = "latest"
    prediction_timestamp: str
    mode: str = "production"


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    mode: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model info response."""

    model_name: str
    stage: str
    version: str | None = None
    metrics: dict | None = None
    mode: str = "production"


# ─── Helper: Haversine Distance ──────────────────────────────────
def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute haversine distance in miles between two points."""
    r_earth = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return r_earth * 2 * math.asin(math.sqrt(a))


# ─── Model State ──────────────────────────────────────────────────
_model_loaded = False


# ─── Application Lifecycle ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load model on startup."""
    global _model_loaded
    logger.info("Starting API service ...")

    if USE_DEMO:
        from src.inference.demo_model import get_demo_model
        get_demo_model()
        _model_loaded = True
        logger.info("Demo model pre-loaded.")
    else:
        try:
            from src.inference.predict import load_production_model
            load_production_model()
            _model_loaded = True
            logger.info("Production model pre-loaded from MLflow.")
        except Exception as e:
            logger.warning("Could not pre-load model: %s (will use demo fallback)", e)
            from src.inference.demo_model import get_demo_model
            get_demo_model()
            _model_loaded = True

    yield
    logger.info("Shutting down API service.")


# ─── FastAPI App ──────────────────────────────────────────────────
app = FastAPI(
    title="NYC Taxi Fare Prediction API",
    description=(
        "Predict taxi fare amount using ML models trained on NYC TLC data.\n\n"
        "**Modes:**\n"
        "- **Production**: Uses MLflow Model Registry.\n"
        "- **Demo**: Uses a standalone sklearn model.\n\n"
        "Built as part of an end-to-end MLOps pipeline."
    ),
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
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "NYC Taxi Fare Prediction API",
        "version": "1.0.0",
        "mode": "demo" if USE_DEMO else "production",
        "docs": "/docs",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "model_info": "GET /model-info",
            "metrics": "GET /metrics",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model_loaded,
        mode="demo" if USE_DEMO else "production",
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
        # Parse datetime
        dt = datetime.fromisoformat(request.pickup_datetime)

        # Compute trip distance
        trip_distance = _haversine(
            request.pickup_latitude, request.pickup_longitude,
            request.dropoff_latitude, request.dropoff_longitude,
        )

        if USE_DEMO:
            # Check if real trained model exists in models/ folder
            try:
                from src.inference.trained_model import predict_fare as real_predict
                fare = real_predict(
                    pickup_latitude=request.pickup_latitude,
                    pickup_longitude=request.pickup_longitude,
                    dropoff_latitude=request.dropoff_latitude,
                    dropoff_longitude=request.dropoff_longitude,
                    pickup_hour=dt.hour,
                    pickup_weekday=dt.weekday(),
                    pickup_month=dt.month,
                    is_weekend=1 if dt.weekday() >= 5 else 0,
                    passenger_count=request.passenger_count,
                    trip_distance=trip_distance,
                )
                mode = "production (standalone)"
            except Exception as e:
                logger.warning("Could not use real model, falling back to demo: %s", e)
                from src.inference.demo_model import predict_demo
                fare = predict_demo(
                    pickup_latitude=request.pickup_latitude,
                    pickup_longitude=request.pickup_longitude,
                    dropoff_latitude=request.dropoff_latitude,
                    dropoff_longitude=request.dropoff_longitude,
                    pickup_hour=dt.hour,
                    pickup_weekday=dt.weekday(),
                    passenger_count=request.passenger_count,
                    trip_distance=trip_distance,
                )
                mode = "demo"
        else:
            # ─── Production mode: use MLflow model ────────
            from src.inference.predict import predict_single
            fare = predict_single(
                pickup_latitude=request.pickup_latitude,
                pickup_longitude=request.pickup_longitude,
                dropoff_latitude=request.dropoff_latitude,
                dropoff_longitude=request.dropoff_longitude,
                pickup_datetime=request.pickup_datetime,
                passenger_count=request.passenger_count,
                trip_distance=trip_distance,
            )
            mode = "production"

        return PredictionResponse(
            predicted_fare=round(fare, 2),
            pickup_datetime=request.pickup_datetime,
            trip_distance_miles=round(trip_distance, 2),
            prediction_timestamp=datetime.utcnow().isoformat(),
            mode=mode,
        )

    except Exception as e:
        PREDICTION_ERROR.inc()
        logger.error("Prediction failed: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        ) from e


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Get information about the current production model."""
    if USE_DEMO:
        return ModelInfoResponse(
            model_name="nyc-taxi-fare-model",
            stage="Production",
            version="demo-1.0",
            mode="demo",
            metrics={
                "rmse": 4.24,
                "mae": 2.81,
                "r2": 0.89,
                "linear_regression_rmse": 6.12,
                "linear_regression_mae": 4.31,
                "random_forest_rmse": 4.85,
                "random_forest_mae": 3.22,
                "xgboost_rmse": 4.24,
                "xgboost_mae": 2.81,
            },
        )

    try:
        import mlflow

        from src.config import MLFLOW_MODEL_NAME, MLFLOW_TRACKING_URI

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
        if not versions:
            return ModelInfoResponse(
                model_name=MLFLOW_MODEL_NAME, stage="None",
                version=None, metrics=None, mode="production",
            )

        latest = versions[0]
        run = client.get_run(latest.run_id)

        return ModelInfoResponse(
            model_name=MLFLOW_MODEL_NAME,
            stage="Production",
            version=latest.version,
            metrics=run.data.metrics,
            mode="production",
        )

    except Exception as e:
        logger.error("Could not fetch model info: %s", e)
        raise HTTPException(status_code=503, detail=f"MLflow unavailable: {str(e)}") from e


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
