"""
Data drift detection module.

Uses Population Stability Index (PSI) to detect distribution shifts
between training and inference feature distributions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Results from drift detection analysis."""

    feature_psi: dict[str, float] = field(default_factory=dict)
    drifted_features: list[str] = field(default_factory=list)
    overall_drift: bool = False
    threshold: float = 0.2

    def summary(self) -> str:
        lines = [
            "Drift Detection Report",
            f"  Threshold: {self.threshold}",
            f"  Overall drift detected: {self.overall_drift}",
            f"  Features analyzed: {len(self.feature_psi)}",
            f"  Features with drift: {len(self.drifted_features)}",
        ]
        if self.drifted_features:
            lines.append("  Drifted features:")
            for feat in self.drifted_features:
                psi = self.feature_psi[feat]
                lines.append(f"    - {feat}: PSI={psi:.4f}")
        return "\n".join(lines)


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Compute the Population Stability Index (PSI).

    PSI interpretation:
      < 0.1  → No significant shift
      0.1–0.2 → Moderate shift (monitor)
      > 0.2  → Significant shift (retrain)

    Args:
        expected: Training distribution values.
        actual: Inference distribution values.
        bins: Number of bins for histogram.

    Returns:
        PSI value.
    """
    eps = 1e-6

    # Create bin edges from the expected distribution
    bin_edges = np.histogram_bin_edges(expected, bins=bins)

    # Compute proportions in each bin
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = (expected_counts + eps) / (expected_counts.sum() + eps * bins)
    actual_pct = (actual_counts + eps) / (actual_counts.sum() + eps * bins)

    # PSI formula
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def detect_drift(
    train_df: pd.DataFrame,
    inference_df: pd.DataFrame,
    features: list[str] | None = None,
    threshold: float = 0.2,
    bins: int = 10,
) -> DriftReport:
    """
    Detect data drift between training and inference datasets.

    Args:
        train_df: Training data features.
        inference_df: Inference data features.
        features: Specific features to check. If None, checks all numeric columns.
        threshold: PSI threshold for flagging drift (default: 0.2).
        bins: Number of bins for PSI calculation.

    Returns:
        DriftReport with per-feature PSI scores and overall drift flag.
    """
    report = DriftReport(threshold=threshold)

    if features is None:
        features = train_df.select_dtypes(include=[np.number]).columns.tolist()

    common_features = [f for f in features if f in train_df.columns and f in inference_df.columns]

    for feat in common_features:
        train_vals = train_df[feat].dropna().values
        inference_vals = inference_df[feat].dropna().values

        if len(train_vals) == 0 or len(inference_vals) == 0:
            logger.warning("Skipping feature '%s': empty after dropping NaN.", feat)
            continue

        psi = compute_psi(train_vals, inference_vals, bins=bins)
        report.feature_psi[feat] = psi

        if psi > threshold:
            report.drifted_features.append(feat)

    report.overall_drift = len(report.drifted_features) > 0

    logger.info(report.summary())
    return report
