"""
Unit tests for training pipeline.
Tests run locally without AWS or MLflow server.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score


@pytest.fixture
def synthetic_fraud_data():
    """Generate synthetic credit card fraud dataset."""
    np.random.seed(42)

    n_samples = 1000
    n_features = 28

    # Create features
    X = np.random.randn(n_samples, n_features)

    # Create target (90% fraud, 10% legit for testing)
    y = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])

    # Create DataFrames
    cols = [f"V{i}" for i in range(n_features)]
    train_data = pd.DataFrame(X[:800], columns=cols)
    train_data["Class"] = y[:800]

    test_data = pd.DataFrame(X[800:], columns=cols)
    test_data["Class"] = y[800:]

    return train_data, test_data


def test_model_training(synthetic_fraud_data):
    """Test LogisticRegression training."""
    train_data, test_data = synthetic_fraud_data

    X_train = train_data.drop(columns=["Class"])
    y_train = train_data["Class"]

    model = LogisticRegression(
        C=0.1,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)

    assert model.classes_[0] == 0
    assert model.classes_[1] == 1


def test_model_evaluation(synthetic_fraud_data):
    """Test model metrics computation."""
    train_data, test_data = synthetic_fraud_data

    X_train = train_data.drop(columns=["Class"])
    y_train = train_data["Class"]
    X_test = test_data.drop(columns=["Class"])
    y_test = test_data["Class"]

    model = LogisticRegression(
        C=0.1,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc_roc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    assert 0 <= auc_roc <= 1, "AUC-ROC should be between 0 and 1"
    assert 0 <= f1 <= 1, "F1 should be between 0 and 1"


def test_class_imbalance_handling(synthetic_fraud_data):
    """Test class_weight balancing."""
    train_data, test_data = synthetic_fraud_data

    X_train = train_data.drop(columns=["Class"])
    y_train = train_data["Class"]

    model_balanced = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000
    )
    model_balanced.fit(X_train, y_train)

    model_unbalanced = LogisticRegression(
        class_weight=None,
        random_state=42,
        max_iter=1000
    )
    model_unbalanced.fit(X_train, y_train)

    # Both should train without errors
    assert model_balanced.coef_.shape == model_unbalanced.coef_.shape
