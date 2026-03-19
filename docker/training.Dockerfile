# ─── Training Pipeline (Prefect) ──────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e "." 2>/dev/null || pip install --no-cache-dir \
    pandas numpy scikit-learn xgboost mlflow boto3 prefect \
    pydantic python-dotenv pyarrow requests matplotlib

# Application code
COPY src/ src/
COPY pipelines/ pipelines/

# Create data directories
RUN mkdir -p data/raw data/processed data/predictions

CMD ["python", "-m", "pipelines.training_pipeline"]
