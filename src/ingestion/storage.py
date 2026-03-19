"""
MinIO / S3-compatible object storage helpers.

Used for uploading raw data and model artifacts to the artifact store.
"""

from __future__ import annotations

import logging
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from src.config import MINIO_ACCESS_KEY, MINIO_BUCKET, MINIO_ENDPOINT, MINIO_SECRET_KEY

logger = logging.getLogger(__name__)


def get_s3_client():
    """
    Create a boto3 S3 client configured for MinIO.

    Returns:
        boto3 S3 client.

    Raises:
        ConnectionError: If MinIO is not reachable.
    """
    try:
        client = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            region_name="us-east-1",
        )
        return client
    except NoCredentialsError as exc:
        raise ConnectionError("MinIO credentials not configured.") from exc


def ensure_bucket_exists(client=None, bucket_name: str = MINIO_BUCKET) -> None:
    """Create the bucket if it doesn't already exist."""
    if client is None:
        client = get_s3_client()
    try:
        client.head_bucket(Bucket=bucket_name)
        logger.info("Bucket '%s' already exists.", bucket_name)
    except ClientError:
        logger.info("Creating bucket '%s' ...", bucket_name)
        client.create_bucket(Bucket=bucket_name)


def upload_file(
    local_path: Path,
    remote_key: str,
    bucket_name: str = MINIO_BUCKET,
    client=None,
) -> str:
    """
    Upload a local file to MinIO.

    Args:
        local_path: Path to the local file.
        remote_key: S3 key (path within the bucket).
        bucket_name: Target bucket name.
        client: Optional pre-existing S3 client.

    Returns:
        The S3 URI of the uploaded file.

    Raises:
        FileNotFoundError: If local_path does not exist.
        ConnectionError: If MinIO is unreachable.
    """
    if not local_path.exists():
        raise FileNotFoundError(f"File not found: {local_path}")

    if client is None:
        client = get_s3_client()

    ensure_bucket_exists(client, bucket_name)

    logger.info("Uploading %s → s3://%s/%s ...", local_path, bucket_name, remote_key)
    client.upload_file(str(local_path), bucket_name, remote_key)
    uri = f"s3://{bucket_name}/{remote_key}"
    logger.info("Upload complete: %s", uri)
    return uri


def download_file(
    remote_key: str,
    local_path: Path,
    bucket_name: str = MINIO_BUCKET,
    client=None,
) -> Path:
    """
    Download a file from MinIO to local filesystem.

    Args:
        remote_key: S3 key to download.
        local_path: Local path to save the file.
        bucket_name: Source bucket name.
        client: Optional pre-existing S3 client.

    Returns:
        Path to the downloaded file.
    """
    if client is None:
        client = get_s3_client()

    local_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading s3://%s/%s → %s ...", bucket_name, remote_key, local_path)
    client.download_file(bucket_name, remote_key, str(local_path))
    logger.info("Download complete: %s", local_path)
    return local_path


def list_files(
    prefix: str = "",
    bucket_name: str = MINIO_BUCKET,
    client=None,
) -> list[str]:
    """List files in a MinIO bucket with an optional prefix filter."""
    if client is None:
        client = get_s3_client()

    response = client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    keys = [obj["Key"] for obj in response.get("Contents", [])]
    return keys
