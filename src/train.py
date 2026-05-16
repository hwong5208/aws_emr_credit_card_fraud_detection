"""
Train credit card fraud detection model with MLflow tracking.
Connects to local MLflow (PostgreSQL backend, MinIO artifacts).

Usage:
  python src/train.py --data data/processed/
"""
import argparse
import logging
import os
import socket
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _resolve_to_ip(uri: str) -> str:
    """
    MLflow Docker Compose on localhost rejects Host headers that are hostnames.
    Resolving to IP makes it work.
    """
    parsed = urlparse(uri)
    try:
        ip = socket.gethostbyname(parsed.hostname)
        port = parsed.port
        return urlunparse(parsed._replace(netloc=f"{ip}:{port}" if port else ip))
    except socket.gaierror:
        return uri


def load_parquet_data(data_path: str):
    """Load train/test parquet files."""
    logger.info(f"Loading data from: {data_path}")

    train_df = pd.read_parquet(f"{data_path}/train")
    test_df = pd.read_parquet(f"{data_path}/test")

    logger.info(f"Train: {len(train_df)} rows, {train_df.shape[1]} features")
    logger.info(f"Test:  {len(test_df)} rows, {test_df.shape[1]} features")

    return train_df, test_df


def train_and_evaluate(train_df, test_df):
    """Train LogisticRegression and compute metrics."""
    logger.info("Extracting features and labels...")

    X_train = train_df.drop(columns=["Class"])
    y_train = train_df["Class"]

    X_test = test_df.drop(columns=["Class"])
    y_test = test_df["Class"]

    logger.info(f"Feature count: {X_train.shape[1]}")
    logger.info(f"Train fraud rate: {y_train.mean():.4f}")
    logger.info(f"Test fraud rate:  {y_test.mean():.4f}")

    # Train with class weighting for imbalanced data
    logger.info("Training LogisticRegression...")
    model = LogisticRegression(
        C=0.1,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "auc_roc": roc_auc_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }

    logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"F1:      {metrics['f1']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:  {metrics['recall']:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion matrix:\n{cm}")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white")
    plt.colorbar(im)
    plt.savefig("/tmp/confusion_matrix.png", dpi=100, bbox_inches="tight")
    plt.close()

    return model, metrics, y_test, y_proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to processed data (train/test parquet)")
    args = parser.parse_args()

    # Configure MLflow
    tracking_uri = _resolve_to_ip(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fraud-detection")

    logger.info(f"MLflow tracking URI: {tracking_uri}")

    # Load data
    train_df, test_df = load_parquet_data(args.data)

    # Train and evaluate
    model, metrics, y_test, y_proba = train_and_evaluate(train_df, test_df)

    # Log to MLflow
    with mlflow.start_run():
        # Params
        mlflow.log_params({
            "model": "LogisticRegression",
            "C": 0.1,
            "solver": "lbfgs",
            "class_weight": "balanced",
            "max_iter": 1000,
        })

        # Metrics
        mlflow.log_metrics(metrics)

        # Artifacts
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact("/tmp/confusion_matrix.png", artifact_path="plots")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"✓ Run logged. Run ID: {run_id}")
        logger.info(f"  View at: {tracking_uri}/#/experiments")


if __name__ == "__main__":
    main()
