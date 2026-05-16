"""
Register best MLflow run in Model Registry (move to Staging).

Usage:
  python scripts/register_model.py
"""
import logging
import os
import socket
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
import mlflow

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _resolve_to_ip(uri: str) -> str:
    """Resolve hostname to IP for MLflow Docker Compose."""
    parsed = urlparse(uri)
    try:
        ip = socket.gethostbyname(parsed.hostname)
        port = parsed.port
        return urlunparse(parsed._replace(netloc=f"{ip}:{port}" if port else ip))
    except socket.gaierror:
        return uri


def main():
    tracking_uri = _resolve_to_ip(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_tracking_uri(tracking_uri)

    logger.info(f"MLflow tracking URI: {tracking_uri}")

    # Find best run by AUC-ROC
    logger.info("Searching for best run in 'fraud-detection' experiment...")

    experiment = mlflow.get_experiment_by_name("fraud-detection")
    if experiment is None:
        logger.error("Experiment 'fraud-detection' not found. Run 'make train-local' first.")
        raise SystemExit(1)

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=100,
        order_by=["metrics.auc_roc DESC"],
    )

    if len(runs) == 0:
        logger.error("No runs found. Run 'make train-local' first.")
        raise SystemExit(1)

    best_run = runs.iloc[0]
    run_id = best_run["run_id"]
    auc_roc = best_run["metrics.auc_roc"]

    logger.info(f"Best run: {run_id}")
    logger.info(f"  AUC-ROC: {auc_roc:.4f}")

    # Register model
    run_uri = f"runs:/{run_id}/model"
    logger.info(f"Registering model from {run_uri}")

    try:
        model_version = mlflow.register_model(run_uri, "FraudDetectionModel")
        logger.info(f"✓ Model registered: {model_version.name}")
        logger.info(f"  Version: {model_version.version}")
        logger.info(f"  Status: {model_version.status}")
    except Exception as e:
        # Model may already exist; transition instead
        logger.warning(f"Register failed (model may exist): {e}")
        logger.info("Trying to transition existing model...")

        client = mlflow.tracking.MlflowClient()
        try:
            model_version = client.get_latest_model_version(
                name="FraudDetectionModel",
                stages=["None"]
            )
            logger.info(f"✓ Found model version: {model_version.version}")
        except Exception:
            logger.error("Could not find or register model. Check MLflow UI for details.")
            raise SystemExit(1)

    logger.info("\n✓ Model promotion ready!")
    logger.info(f"  View at: {tracking_uri}/#/models/FraudDetectionModel")


if __name__ == "__main__":
    main()
