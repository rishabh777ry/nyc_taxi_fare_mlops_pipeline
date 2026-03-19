"""
Data ingestion module for NYC Taxi Fare MLOps pipeline.

Downloads NYC TLC Yellow Taxi trip data (Parquet) and stores it
locally or in MinIO for downstream processing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from src.config import (
    DEFAULT_MONTH,
    DEFAULT_YEAR,
    NYC_TLC_BASE_URL,
    RAW_DATA_DIR,
)

logger = logging.getLogger(__name__)

# ─── Chunk size for streaming downloads (10 MB) ──────────────────
DOWNLOAD_CHUNK_SIZE = 10 * 1024 * 1024

# ─── Chunk size for reading large Parquet files ───────────────────
READ_CHUNK_SIZE = 500_000  # rows


def build_dataset_url(year: int = DEFAULT_YEAR, month: int = DEFAULT_MONTH) -> str:
    """Build the NYC TLC dataset URL for a given year/month."""
    return NYC_TLC_BASE_URL.format(year=year, month=month)


def download_dataset(
    year: int = DEFAULT_YEAR,
    month: int = DEFAULT_MONTH,
    output_dir: Path = RAW_DATA_DIR,
    force: bool = False,
) -> Path:
    """
    Download NYC TLC Yellow Taxi trip data as Parquet.

    Args:
        year: Dataset year (default: 2023).
        month: Dataset month (default: 1).
        output_dir: Directory to save the file.
        force: Re-download even if file exists.

    Returns:
        Path to the downloaded Parquet file.

    Raises:
        requests.HTTPError: If the download fails.
        ConnectionError: If the server is unreachable.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"yellow_tripdata_{year}-{month:02d}.parquet"
    filepath = output_dir / filename

    if filepath.exists() and not force:
        logger.info("Dataset already exists at %s, skipping download.", filepath)
        return filepath

    url = build_dataset_url(year, month)
    logger.info("Downloading dataset from %s ...", url)

    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
    except requests.ConnectionError as exc:
        raise ConnectionError(
            f"Failed to connect to NYC TLC data server: {url}"
        ) from exc

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = (downloaded / total_size) * 100
                    logger.info("Download progress: %.1f%%", pct)

    logger.info("Dataset saved to %s (%.2f MB)", filepath, filepath.stat().st_size / 1e6)
    return filepath


def load_dataset(
    filepath: Path,
    chunk_size: int | None = None,
    sample_fraction: float | None = None,
) -> pd.DataFrame:
    """
    Load a Parquet dataset into a Pandas DataFrame.

    Supports chunked loading for large datasets and optional sampling
    to reduce memory usage during development.

    Args:
        filepath: Path to the Parquet file.
        chunk_size: If set, read in batches of this many rows (memory-safe).
        sample_fraction: If set (0.0–1.0), randomly sample this fraction.

    Returns:
        DataFrame with the loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If sample_fraction is out of range.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    if sample_fraction is not None and not (0.0 < sample_fraction <= 1.0):
        raise ValueError(f"sample_fraction must be in (0, 1], got {sample_fraction}")

    logger.info("Loading dataset from %s ...", filepath)

    if chunk_size:
        # Chunked reading for memory-constrained environments
        chunks = []
        pf = pd.read_parquet(filepath)
        total_rows = len(pf)
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            chunks.append(pf.iloc[start:end])
            logger.info("Loaded rows %d–%d / %d", start, end, total_rows)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_parquet(filepath)

    if sample_fraction is not None:
        original_len = len(df)
        df = df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)
        logger.info("Sampled %d → %d rows (%.1f%%)", original_len, len(df), sample_fraction * 100)

    logger.info("Loaded %d rows, %d columns.", len(df), len(df.columns))
    return df


def ingest(
    year: int = DEFAULT_YEAR,
    month: int = DEFAULT_MONTH,
    sample_fraction: float | None = 0.1,
) -> pd.DataFrame:
    """
    Full ingestion pipeline: download + load + optional sample.

    Args:
        year: Dataset year.
        month: Dataset month.
        sample_fraction: Fraction to sample (None = full dataset).

    Returns:
        Loaded DataFrame.
    """
    filepath = download_dataset(year=year, month=month)
    df = load_dataset(filepath, sample_fraction=sample_fraction)
    return df
