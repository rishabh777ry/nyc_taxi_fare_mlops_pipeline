# 🚕 NYC Taxi Fare Prediction — MLOps Pipeline

Production-grade end-to-end MLOps pipeline for predicting NYC taxi fares, built entirely with open-source tools.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATION (Prefect)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────┐ │
│  │ Ingest   │→ │ Validate │→ │ Feature  │→ │  Train   │→ │ Eval  │ │
│  │ (NYC TLC)│  │ (Schema, │  │  Eng.    │  │ (LR, RF, │  │       │ │
│  │          │  │  Outlier)│  │ (Havers.)│  │  XGBoost)│  │       │ │
│  └──────────┘  └──────────┘  └──────────┘  └────┬─────┘  └───────┘ │
└──────────────────────────────────────────────────┼──────────────────┘
                                                   │
┌──────────────────────────────────────────────────▼──────────────────┐
│  MLflow (Experiment Tracking + Model Registry)                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  PostgreSQL   │    │    MinIO      │    │   Model      │          │
│  │  (Backend)    │    │  (Artifacts)  │    │  Registry    │          │
│  └──────────────┘    └──────────────┘    └──────┬───────┘          │
└─────────────────────────────────────────────────┼──────────────────┘
                                                   │ Production Model
┌──────────────────────────────────────────────────▼──────────────────┐
│  SERVING                                                            │
│  ┌──────────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  FastAPI          │    │  Streamlit    │    │  Prometheus   │      │
│  │  POST /predict    │    │  Dashboard    │    │  + Grafana    │      │
│  │  GET  /health     │    │              │    │  Monitoring   │      │
│  └──────────────────┘    └──────────────┘    └──────────────┘      │
└────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
nyc-taxi-fare-mlops/
├── src/                      # Core ML source code
│   ├── config.py             # Shared configuration
│   ├── ingestion/            # Data download & loading
│   │   ├── ingest.py
│   │   └── storage.py        # MinIO helpers
│   ├── validation/           # Data quality checks
│   │   └── validate.py
│   ├── features/             # Feature engineering
│   │   └── engineer.py
│   ├── training/             # Model training & eval
│   │   ├── train.py
│   │   └── evaluate.py
│   └── inference/            # Model inference
│       └── predict.py
├── pipelines/                # Prefect orchestration
│   ├── training_pipeline.py
│   └── batch_inference_pipeline.py
├── api/                      # FastAPI service
│   └── main.py
├── dashboard/                # Streamlit dashboard
│   └── app.py
├── monitoring/               # Prometheus + Grafana
│   ├── prometheus.yml
│   ├── drift.py              # Data drift detection (PSI)
│   └── grafana/
│       ├── dashboards/
│       └── provisioning/
├── tests/                    # Unit tests
│   ├── conftest.py
│   ├── test_validation.py
│   ├── test_features.py
│   └── test_api.py
├── docker/                   # Dockerfiles
│   ├── api.Dockerfile
│   ├── mlflow.Dockerfile
│   ├── training.Dockerfile
│   └── dashboard.Dockerfile
├── docker-compose.yml        # Full stack orchestration
├── .github/workflows/ci.yml  # CI/CD pipeline
├── pyproject.toml
├── Makefile
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- ~4 GB RAM available

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/nyc-taxi-fare-mlops.git
cd nyc-taxi-fare-mlops

# Copy environment file
cp .env.example .env

# Start all services
docker-compose up --build -d

# Check service status
docker-compose ps
```

**Service URLs:**
| Service     | URL                      |
|-------------|--------------------------|
| FastAPI     | http://localhost:8000     |
| API Docs    | http://localhost:8000/docs|
| MLflow UI   | http://localhost:5000     |
| Grafana     | http://localhost:3000     |
| Streamlit   | http://localhost:8501     |
| MinIO UI    | http://localhost:9001     |
| Prometheus  | http://localhost:9090     |

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv venv && source venv/bin/activate

# Install dependencies
make setup

# Run the training pipeline
make train

# Start the API server
make serve

# Start the dashboard
make dashboard

# Run tests
make test
```

### Run Training Pipeline (Docker)

```bash
# Start training with Docker
docker-compose --profile training up training

# Or run locally
python -m pipelines.training_pipeline
```

---

## 📡 API Usage

### Predict Fare

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_latitude": 40.7128,
    "pickup_longitude": -74.0060,
    "dropoff_latitude": 40.7580,
    "dropoff_longitude": -73.9855,
    "pickup_datetime": "2023-01-15T14:30:00",
    "passenger_count": 2
  }'
```

**Response:**
```json
{
  "predicted_fare": 12.45,
  "pickup_datetime": "2023-01-15T14:30:00",
  "trip_distance_miles": 3.42,
  "model_version": "latest",
  "prediction_timestamp": "2023-01-20T10:30:00.000Z"
}
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Model Info

```bash
curl http://localhost:8000/model-info
```

### Prometheus Metrics

```bash
curl http://localhost:8000/metrics
```

---

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_validation.py -v
```

---

## 📊 Monitoring

- **Prometheus** scrapes FastAPI `/metrics` every 10s
- **Grafana** (admin/admin) shows:
  - Request rate
  - Latency percentiles (p50/p95/p99)
  - Error rate
  - Endpoint breakdown
- **Drift detection** uses PSI (Population Stability Index)

---

## ⚙️ Configuration

All configuration is in `src/config.py` and `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URL |
| `MINIO_ENDPOINT` | `http://localhost:9000` | MinIO endpoint |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `API_PORT` | `8000` | FastAPI port |

---

## 🔄 CI/CD

GitHub Actions runs on every push to `main`/`develop`:

1. **Lint** — Ruff check & format
2. **Test** — pytest with coverage
3. **Docker Build** — Build all 4 images
4. **Smoke Test** — docker-compose up & health check (main only)

---

## 📋 Tech Stack

| Component | Tool |
|-----------|------|
| ML Models | Scikit-learn, XGBoost |
| API | FastAPI |
| Orchestration | Prefect 2.x |
| Experiment Tracking | MLflow |
| Database | PostgreSQL |
| Artifact Store | MinIO |
| Monitoring | Prometheus + Grafana |
| Dashboard | Streamlit |
| CI/CD | GitHub Actions |
| Containerization | Docker + Docker Compose |

---

