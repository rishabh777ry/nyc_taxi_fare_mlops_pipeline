"""
Prefect training pipeline for NYC Taxi Fare MLOps.

Orchestrates: ingest → validate → feature engineering → train → evaluate → register.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from prefect import flow, task
from prefect.tasks import task_input_hash

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@task(
    name="ingest_data",
    retries=3,
    retry_delay_seconds=30,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=24),
    log_prints=True,
)
def ingest_data(year: int, month: int, sample_fraction: float):
    """Download and load NYC taxi trip data."""
    from src.ingestion.ingest import ingest
    df = ingest(year=year, month=month, sample_fraction=sample_fraction)
    print(f"Ingested {len(df):,} rows from {year}-{month:02d}")
    return df


@task(
    name="validate_data",
    retries=2,
    retry_delay_seconds=10,
    log_prints=True,
)
def validate_data(df):
    """Run data validation pipeline."""
    from src.validation.validate import validate
    df_clean, report = validate(df, fail_on_error=True)
    print(report.summary())
    return df_clean


@task(
    name="build_features",
    retries=1,
    log_prints=True,
)
def build_features_task(df):
    """Run feature engineering pipeline."""
    from src.features.engineer import build_features
    x_train, x_test, y_train, y_test, scaler = build_features(df)
    print(f"Features built — Train: {len(x_train):,}, Test: {len(x_test):,}")
    return x_train, x_test, y_train, y_test


@task(
    name="train_models",
    retries=1,
    log_prints=True,
)
def train_models_task(x_train, x_test, y_train, y_test):
    """Train all models and register the best one."""
    from src.training.train import train_all_models
    best_name, version = train_all_models(x_train, y_train, x_test, y_test)
    print(f"Best model: {best_name} (version {version})")
    return best_name, version


@task(
    name="evaluate_model",
    retries=1,
    log_prints=True,
)
def evaluate_model_task(x_test, y_test):
    """Evaluate the production model on the test set."""
    from src.inference.predict import load_production_model
    from src.training.evaluate import evaluate_model

    model = load_production_model(force_reload=True)
    metrics = evaluate_model(model, x_test, y_test)
    print(f"Final evaluation — RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
    return metrics


@flow(
    name="training-pipeline",
    description="End-to-end training pipeline: ingest → validate → featurize → train → evaluate",
    retries=1,
    retry_delay_seconds=60,
    log_prints=True,
)
def training_pipeline(
    year: int = 2023,
    month: int = 1,
    sample_fraction: float = 0.1,
):
    """
    Run the full training pipeline.

    Args:
        year: Dataset year.
        month: Dataset month.
        sample_fraction: Fraction of data to sample (for dev speed).
    """
    print(f"Starting training pipeline for {year}-{month:02d} (sample: {sample_fraction:.0%})")

    # Step 1: Ingest
    df = ingest_data(year=year, month=month, sample_fraction=sample_fraction)

    # Step 2: Validate
    df_clean = validate_data(df)

    # Step 3: Feature Engineering
    x_train, x_test, y_train, y_test = build_features_task(df_clean)

    # Step 4: Train + Register
    best_name, version = train_models_task(x_train, x_test, y_train, y_test)

    # Step 5: Final evaluation
    metrics = evaluate_model_task(x_test, y_test)

    print(f"Pipeline complete. Best model: {best_name} v{version}")
    print(f"  RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")

    return {"best_model": best_name, "version": version, "metrics": metrics}


if __name__ == "__main__":
    training_pipeline()
