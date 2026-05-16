.PHONY: setup mlflow-up mlflow-down preprocess-local train-local register-local test lint deploy-infra upload-data run-emr help

help:
	@echo "AWS EMR Credit Card Fraud Detection — MLOps POC"
	@echo ""
	@echo "Local Development (shift-left — test locally first):"
	@echo "  make setup              — Install Python dependencies"
	@echo "  make mlflow-up          — Start MLflow Docker stack (PostgreSQL + MinIO)"
	@echo "  make preprocess-local   — Run PySpark preprocessing locally"
	@echo "  make train-local        — Train model locally, log to MLflow"
	@echo "  make register-local     — Register best model in MLflow Registry"
	@echo "  make test               — Run pytest (no AWS needed)"
	@echo "  make lint               — Run flake8"
	@echo ""
	@echo "Cloud (only after local is green):"
	@echo "  make deploy-infra       — Deploy CDK stack (S3 + EMR Serverless)"
	@echo "  make upload-data        — Upload creditcard.csv to S3"
	@echo "  make run-emr            — Submit EMR Serverless job"
	@echo ""
	@echo "Teardown:"
	@echo "  make mlflow-down        — Stop MLflow Docker stack"

setup:
	python3 -m venv venv 2>/dev/null || true
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

mlflow-up:
	cd mlflow && docker compose up -d
	@echo "Waiting for MLflow to be ready..."
	@sleep 5
	@echo "✓ MLflow UI: http://localhost:5000"
	@echo "✓ MinIO console: http://localhost:9001 (minioadmin/minioadmin)"

mlflow-down:
	cd mlflow && docker compose down

preprocess-local:
	python src/preprocess.py --input data/creditcard.csv --output data/processed/

train-local:
	python src/train.py --data data/processed/

register-local:
	python scripts/register_model.py

test:
	pytest tests/ -v

lint:
	flake8 src/ scripts/ tests/

deploy-infra:
	cd infra && pip install -r requirements.txt && cdk deploy

upload-data:
	python scripts/upload_data.py --file data/creditcard.csv

run-emr:
	python scripts/submit_emr_job.py
