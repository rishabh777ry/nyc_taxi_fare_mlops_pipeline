"""
Unit tests for data validation module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation.validate import (
    ValidationError,
    ValidationReport,
    detect_outliers,
    handle_nulls,
    validate,
    validate_schema,
)


class TestValidateSchema:
    """Tests for schema validation."""

    def test_valid_schema_passes(self, sample_raw_df):
        """Schema validation should pass for a complete DataFrame."""
        report = ValidationReport()
        df = validate_schema(sample_raw_df, report)
        assert report.schema_valid is True
        assert len(report.errors) == 0

    def test_missing_required_columns_fails(self, sample_df_bad_schema):
        """Should report error when required columns are missing."""
        report = ValidationReport()
        validate_schema(sample_df_bad_schema, report)
        assert report.schema_valid is False
        assert len(report.errors) > 0
        assert "Missing required columns" in report.errors[0]

    def test_schema_without_latlon_warns(self):
        """Should warn when neither lat/lon nor LocationID columns exist."""
        df = pd.DataFrame({
            "tpep_pickup_datetime": ["2023-01-01"],
            "tpep_dropoff_datetime": ["2023-01-01"],
            "trip_distance": [3.0],
            "fare_amount": [15.0],
        })
        report = ValidationReport()
        validate_schema(df, report)
        assert report.schema_valid is True
        assert any("lat/lon" in w.lower() or "LocationID" in w for w in report.warnings)


class TestHandleNulls:
    """Tests for null value handling."""

    def test_drops_null_fare_rows(self, sample_df_with_nulls):
        """Should drop rows where fare_amount is null."""
        report = ValidationReport()
        df = handle_nulls(sample_df_with_nulls, report)
        assert df["fare_amount"].isnull().sum() == 0
        assert any("null fare_amount" in w.lower() for w in report.warnings)

    def test_fills_numeric_nulls_with_median(self, sample_df_with_nulls):
        """Should fill numeric null values with column median."""
        report = ValidationReport()
        df = handle_nulls(sample_df_with_nulls, report)
        # trip_distance and passenger_count had nulls
        assert df["trip_distance"].isnull().sum() == 0
        assert df["passenger_count"].isnull().sum() == 0

    def test_records_null_counts(self, sample_df_with_nulls):
        """Should record null counts in the report."""
        report = ValidationReport()
        handle_nulls(sample_df_with_nulls, report)
        assert len(report.null_counts) > 0


class TestDetectOutliers:
    """Tests for outlier detection."""

    def test_removes_extreme_fares(self, sample_df_with_outliers):
        """Should remove rows with fares outside valid range."""
        report = ValidationReport()
        df = detect_outliers(sample_df_with_outliers, report)
        assert df["fare_amount"].max() <= 500.0
        assert df["fare_amount"].min() >= 0.0

    def test_removes_extreme_distance(self, sample_df_with_outliers):
        """Should remove rows with trip distance > 200 miles."""
        report = ValidationReport()
        df = detect_outliers(sample_df_with_outliers, report)
        assert df["trip_distance"].max() <= 200.0

    def test_removes_zero_passengers(self, sample_df_with_outliers):
        """Should remove rows with 0 passengers."""
        report = ValidationReport()
        df = detect_outliers(sample_df_with_outliers, report)
        assert (df["passenger_count"] > 0).all()

    def test_removes_invalid_coordinates(self, sample_df_with_outliers):
        """Should remove rows with coordinates outside NYC bounding box."""
        report = ValidationReport()
        df = detect_outliers(sample_df_with_outliers, report)
        if "pickup_latitude" in df.columns:
            assert df["pickup_latitude"].between(40.4774, 40.9176).all()

    def test_reports_outlier_counts(self, sample_df_with_outliers):
        """Should report number of outliers removed per category."""
        report = ValidationReport()
        detect_outliers(sample_df_with_outliers, report)
        assert len(report.outliers_removed) > 0


class TestValidatePipeline:
    """Tests for the full validation pipeline."""

    def test_valid_data_passes(self, sample_raw_df):
        """Full pipeline should pass on valid data."""
        df, report = validate(sample_raw_df, fail_on_error=True)
        assert report.is_valid
        assert len(df) > 0
        assert report.rows_after_cleaning <= report.total_rows

    def test_invalid_schema_raises(self, sample_df_bad_schema):
        """Should raise ValidationError on missing required columns."""
        with pytest.raises(ValidationError):
            validate(sample_df_bad_schema, fail_on_error=True)

    def test_invalid_schema_no_raise(self, sample_df_bad_schema):
        """Should not raise when fail_on_error=False."""
        df, report = validate(sample_df_bad_schema, fail_on_error=False)
        assert not report.is_valid
        assert len(report.errors) > 0

    def test_report_summary_string(self, sample_raw_df):
        """Report summary should be a non-empty string."""
        _, report = validate(sample_raw_df)
        summary = report.summary()
        assert isinstance(summary, str)
        assert "Validation Report" in summary
