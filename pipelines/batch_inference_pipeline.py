"""
Prefect batch inference pipeline for NYC Taxi Fare MLOps.

Orchestrates: load data → validate → featurize → predict → store results.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from prefect import flow, task

from src.config import PREDICTIONS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@task(
    name="load_inference_data",
    retries=3,
    retry_delay_seconds=30,
    log_prints=True,
)
def load_inference_data(filepath: str):
    """Load new data for batch inference."""
    import pandas as pd

    fp = Path(filepath)
    if not fp.exists():
        raise FileNotFoundError(f"Inference data not found: {filepath}")

    df = pd.read_parquet(fp)
    print(f"Loaded {len(df):,} rows for inference from {fp.name}")
    return df


@task(
    name="validate_inference_data",
    retries=2,
    retry_delay_seconds=10,
    log_prints=True,
)
def validate_inference_data(df):
    """Validate incoming inference data (lighter checks)."""
    from src.validation.validate import validate

    df_clean, report = validate(df, fail_on_error=False)
    print(report.summary())

    if not report.is_valid:
        print(f"WARNING: Validation found issues but proceeding with {len(df_clean):,} rows.")
    return df_clean


@task(
    name="run_predictions",
    retries=2,
    retry_delay_seconds=15,
    log_prints=True,
)
def run_predictions(df):
    """Run batch predictions using the Production model."""
    from src.inference.predict import predict_batch

    df_with_preds = predict_batch(df)
    print(f"Predictions complete. Mean predicted fare: ${df_with_preds['predicted_fare'].mean():.2f}")
    return df_with_preds


@task(
    name="store_results",
    retries=2,
    retry_delay_seconds=10,
    log_prints=True,
)
def store_results(df, output_dir: str | None = None):
    """Store prediction results to filesystem."""
    if output_dir is None:
        output_dir = str(PREDICTIONS_DIR)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"predictions_{timestamp}.parquet"
    filepath = out_path / filename

    df.to_parquet(filepath, index=False)
    print(f"Predictions saved to {filepath} ({len(df):,} rows)")
    return str(filepath)


@flow(
    name="batch-inference-pipeline",
    description="Batch inference pipeline: load → validate → predict → store",
    retries=1,
    retry_delay_seconds=60,
    log_prints=True,
)
def batch_inference_pipeline(
    input_filepath: str | None = None,
    output_dir: str | None = None,
):
    """
    Run batch inference on new data.

    Args:
        input_filepath: Path to the Parquet file with new trip data.
        output_dir: Directory to save predictions.
    """
    if input_filepath is None:
        # Default: use the most recent raw data file
        from src.config import RAW_DATA_DIR
        parquet_files = sorted(RAW_DATA_DIR.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError("No Parquet files found in data/raw/")
        input_filepath = str(parquet_files[-1])

    print(f"Starting batch inference on: {input_filepath}")

    # Step 1: Load data
    df = load_inference_data(input_filepath)

    # Step 2: Validate
    df_clean = validate_inference_data(df)

    # Step 3: Predict
    df_with_preds = run_predictions(df_clean)

    # Step 4: Store results
    output_path = store_results(df_with_preds, output_dir)

    print(f"Batch inference complete. Results: {output_path}")
    return output_path


if __name__ == "__main__":
    batch_inference_pipeline()
