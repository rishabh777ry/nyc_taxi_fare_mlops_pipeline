"""
Data validation module for NYC Taxi Fare MLOps pipeline.

Performs schema validation, null checks, and outlier detection.
Halts the pipeline on critical failures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import (
    MAX_FARE_AMOUNT,
    MAX_NULL_FRACTION,
    MAX_TRIP_DISTANCE,
    MIN_FARE_AMOUNT,
    NYC_LAT_MAX,
    NYC_LAT_MIN,
    NYC_LON_MAX,
    NYC_LON_MIN,
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when data validation fails critically."""


@dataclass
class ValidationReport:
    """Stores results from the validation pipeline."""

    total_rows: int = 0
    rows_after_cleaning: int = 0
    null_counts: dict = field(default_factory=dict)
    outliers_removed: dict = field(default_factory=dict)
    schema_valid: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [
            "Validation Report",
            f"  Total rows: {self.total_rows:,}",
            f"  Rows after cleaning: {self.rows_after_cleaning:,}",
            f"  Rows removed: {self.total_rows - self.rows_after_cleaning:,}",
            f"  Schema valid: {self.schema_valid}",
            f"  Warnings: {len(self.warnings)}",
            f"  Errors: {len(self.errors)}",
        ]
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        if self.errors:
            lines.append("  Errors:")
            for e in self.errors:
                lines.append(f"    - {e}")
        return "\n".join(lines)


# ─── Required columns (minimum viable set) ───────────────────────
REQUIRED_COLUMNS = {
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "fare_amount",
}

# Columns we need for lat/lon or fallback location IDs
LATLON_COLUMNS = {
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
}
LOCATION_ID_COLUMNS = {"PULocationID", "DOLocationID"}


def validate_schema(df: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    """
    Validate that the DataFrame contains the required columns.

    The NYC TLC schema changed over time — newer data uses LocationIDs
    instead of lat/lon.  We accept either and normalize later.
    """
    present = set(df.columns)
    missing_core = REQUIRED_COLUMNS - present
    if missing_core:
        report.errors.append(f"Missing required columns: {missing_core}")
        report.schema_valid = False
        return df

    has_latlon = LATLON_COLUMNS.issubset(present)
    has_locid = LOCATION_ID_COLUMNS.issubset(present)

    if not has_latlon and not has_locid:
        report.warnings.append(
            "Neither lat/lon nor LocationID columns found. "
            "Distance features will be limited to trip_distance."
        )

    report.schema_valid = True
    logger.info("Schema validation passed. Columns: %d", len(df.columns))
    return df


def handle_nulls(df: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    """
    Check for and handle null/missing values.

    Strategy:
    - Numeric columns: fill with median.
    - Critical columns with >30% nulls: flag as error.
    - Drop rows where target (fare_amount) is null.
    """
    critical_cols = ["fare_amount", "trip_distance", "tpep_pickup_datetime"]

    for col in df.columns:
        null_frac = df[col].isnull().mean()
        if null_frac > 0:
            report.null_counts[col] = {
                "count": int(df[col].isnull().sum()),
                "fraction": round(float(null_frac), 4),
            }

        if col in critical_cols and null_frac > MAX_NULL_FRACTION:
            report.errors.append(
                f"Column '{col}' has {null_frac:.1%} nulls (threshold: {MAX_NULL_FRACTION:.0%})."
            )

    # Drop rows with null fare_amount (target)
    before = len(df)
    df = df.dropna(subset=["fare_amount"])
    dropped = before - len(df)
    if dropped:
        report.warnings.append(f"Dropped {dropped:,} rows with null fare_amount.")

    # Fill numeric nulls with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            report.warnings.append(f"Filled {col} nulls with median ({median_val:.2f}).")

    return df


def detect_outliers(df: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    """
    Remove outliers based on business rules for taxi fares.

    Filters:
    - fare_amount must be in [MIN_FARE, MAX_FARE].
    - trip_distance must be in [0, MAX_DISTANCE].
    - If lat/lon present, must be within NYC bounding box.
    """
    before = len(df)

    # ─── Fare amount ──────────────────────────────────────────
    fare_mask = (df["fare_amount"] >= MIN_FARE_AMOUNT) & (df["fare_amount"] <= MAX_FARE_AMOUNT)
    fare_outliers = (~fare_mask).sum()
    if fare_outliers:
        report.outliers_removed["fare_amount"] = int(fare_outliers)
    df = df[fare_mask]

    # ─── Trip distance ────────────────────────────────────────
    dist_mask = (df["trip_distance"] >= 0) & (df["trip_distance"] <= MAX_TRIP_DISTANCE)
    dist_outliers = (~dist_mask).sum()
    if dist_outliers:
        report.outliers_removed["trip_distance"] = int(dist_outliers)
    df = df[dist_mask]

    # ─── Lat/lon bounding box (if columns exist) ─────────────
    if "pickup_latitude" in df.columns and "pickup_longitude" in df.columns:
        geo_mask = (
            (df["pickup_latitude"].between(NYC_LAT_MIN, NYC_LAT_MAX))
            & (df["pickup_longitude"].between(NYC_LON_MIN, NYC_LON_MAX))
            & (df["dropoff_latitude"].between(NYC_LAT_MIN, NYC_LAT_MAX))
            & (df["dropoff_longitude"].between(NYC_LON_MIN, NYC_LON_MAX))
        )
        geo_outliers = (~geo_mask).sum()
        if geo_outliers:
            report.outliers_removed["coordinates"] = int(geo_outliers)
        df = df[geo_mask]

    # ─── Zero/negative passenger count ────────────────────────
    if "passenger_count" in df.columns:
        pax_mask = df["passenger_count"] > 0
        pax_outliers = (~pax_mask).sum()
        if pax_outliers:
            report.outliers_removed["passenger_count"] = int(pax_outliers)
        df = df[pax_mask]

    total_removed = before - len(df)
    report.warnings.append(f"Outlier detection removed {total_removed:,} rows total.")
    logger.info("Outlier detection complete. Removed %d / %d rows.", total_removed, before)
    return df


def validate(df: pd.DataFrame, fail_on_error: bool = True) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Run the full validation pipeline.

    Args:
        df: Raw DataFrame to validate.
        fail_on_error: If True, raise ValidationError on critical failures.

    Returns:
        Tuple of (cleaned DataFrame, ValidationReport).

    Raises:
        ValidationError: If critical issues are found and fail_on_error is True.
    """
    report = ValidationReport(total_rows=len(df))
    logger.info("Starting validation on %d rows ...", len(df))

    # 1. Schema validation
    df = validate_schema(df, report)
    if report.errors and fail_on_error:
        raise ValidationError(report.summary())

    # 2. Handle nulls
    df = handle_nulls(df, report)
    if report.errors and fail_on_error:
        raise ValidationError(report.summary())

    # 3. Outlier detection
    df = detect_outliers(df, report)

    report.rows_after_cleaning = len(df)
    logger.info("Validation complete.\n%s", report.summary())

    if report.errors and fail_on_error:
        raise ValidationError(report.summary())

    return df, report
