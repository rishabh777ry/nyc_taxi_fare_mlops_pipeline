# ─── MLflow Tracking Server ───────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    mlflow==2.9.2 \
    psycopg2-binary \
    boto3

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}", \
     "--default-artifact-root", "s3://mlflow-artifacts/", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
