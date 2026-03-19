.PHONY: setup train serve test docker-up docker-down lint clean

# ─── Local Development ────────────────────────────────────────────
setup:
	pip install -e ".[dev]"
	cp -n .env.example .env || true
	mkdir -p data/raw data/processed data/predictions

train:
	python -m pipelines.training_pipeline

predict:
	python -m pipelines.batch_inference_pipeline

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run dashboard/app.py --server.port 8501

# ─── Quality ──────────────────────────────────────────────────────
lint:
	ruff check src/ api/ pipelines/ tests/ dashboard/ monitoring/
	ruff format --check src/ api/ pipelines/ tests/ dashboard/ monitoring/

format:
	ruff format src/ api/ pipelines/ tests/ dashboard/ monitoring/
	ruff check --fix src/ api/ pipelines/ tests/ dashboard/ monitoring/

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov=api --cov-report=html

# ─── Docker ───────────────────────────────────────────────────────
docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down -v

docker-logs:
	docker-compose logs -f

docker-ps:
	docker-compose ps

# ─── Cleanup ──────────────────────────────────────────────────────
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache
	rm -rf dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
