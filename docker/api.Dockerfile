# ─── FastAPI Service ──────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e "." 2>/dev/null || pip install --no-cache-dir \
    fastapi uvicorn[standard] pandas numpy scikit-learn xgboost \
    mlflow boto3 prometheus-client pydantic python-dotenv pyarrow requests

# Application code
COPY src/ src/
COPY api/ api/

# Create data directories
RUN mkdir -p data/raw data/processed data/predictions

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
