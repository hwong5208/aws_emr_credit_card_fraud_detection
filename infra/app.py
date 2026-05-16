#!/usr/bin/env python3
"""CDK app for fraud detection MLOps infrastructure."""
import os
from aws_cdk import App
from fraud_detection.emr_stack import FraudDetectionStack

app = App()

stage = os.getenv("CDK_STAGE", "dev")

FraudDetectionStack(
    app,
    f"FraudDetection-{stage}",
    stage=stage,
    env={
        "account": os.getenv("CDK_DEFAULT_ACCOUNT"),
        "region": os.getenv("CDK_DEFAULT_REGION", "us-west-2"),
    },
)

app.synth()
