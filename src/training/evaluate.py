"""
Model evaluation module.

Computes hold-out metrics and generates evaluation plots saved as MLflow artifacts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for Docker/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path | None = None,
) -> dict[str, float]:
    """
    Evaluate a trained model on the test set and generate diagnostic plots.

    Args:
        model: Trained model with a .predict() method.
        X_test: Test features.
        y_test: Test target.
        output_dir: Directory to save evaluation plots. If None, uses data/processed/.

    Returns:
        Dictionary of evaluation metrics.
    """
    from src.config import PROCESSED_DATA_DIR

    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR / "eval_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(x_test)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "median_absolute_error": float(np.median(np.abs(y_test - y_pred))),
    }

    logger.info(
        "Evaluation metrics — RMSE: %.4f, MAE: %.4f, R²: %.4f",
        metrics["rmse"], metrics["mae"], metrics["r2"],
    )

    # ─── Plot 1: Actual vs Predicted ──────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.3, s=5, color="#2196F3")
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--", linewidth=2, label="Perfect Prediction",
    )
    ax.set_xlabel("Actual Fare ($)")
    ax.set_ylabel("Predicted Fare ($)")
    ax.set_title("Actual vs Predicted Taxi Fare")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    # ─── Plot 2: Residual Distribution ────────────────────────
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(residuals, bins=100, alpha=0.7, color="#4CAF50", edgecolor="black")
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Residual (Actual - Predicted) ($)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "residuals.png", dpi=150)
    plt.close(fig)

    # ─── Plot 3: Error by Fare Range ──────────────────────────
    bins = [0, 5, 10, 20, 30, 50, 100, 500]
    y_test_series = pd.Series(y_test.values if hasattr(y_test, 'values') else y_test)
    fare_bins = pd.cut(y_test_series, bins=bins)
    errors = pd.Series(np.abs(y_test.values - y_pred), index=y_test.index)
    error_by_bin = errors.groupby(fare_bins).mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    error_by_bin.plot(kind="bar", ax=ax, color="#FF9800", edgecolor="black")
    ax.set_xlabel("Fare Range ($)")
    ax.set_ylabel("Mean Absolute Error ($)")
    ax.set_title("MAE by Fare Range")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "error_by_fare_range.png", dpi=150)
    plt.close(fig)

    logger.info("Evaluation plots saved to %s", output_dir)
    return metrics
