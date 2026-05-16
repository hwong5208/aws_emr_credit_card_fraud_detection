"""
Upload creditcard.csv to S3 (raw/ prefix).

Usage:
  python scripts/upload_data.py --file data/creditcard.csv
"""
import argparse
import logging
import os
import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REGION = "us-west-2"
STAGE = "dev"


def get_ssm(name):
    ssm = boto3.client("ssm", region_name=REGION)
    try:
        return ssm.get_parameter(Name=name)["Parameter"]["Value"]
    except ClientError as e:
        logger.error(f"Failed to get SSM parameter {name}: {e}")
        logger.error("Have you run 'make deploy-infra' yet?")
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to creditcard.csv")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        raise SystemExit(f"File not found: {args.file}")

    bucket = get_ssm(f"/fraud-mlops/{STAGE}/bucket-name")
    logger.info(f"Bucket: {bucket}")
    logger.info(f"Uploading: {args.file}")

    s3 = boto3.client("s3", region_name=REGION)
    s3_key = "raw/creditcard.csv"

    try:
        s3.upload_file(args.file, bucket, s3_key)
        logger.info(f"✓ Uploaded to s3://{bucket}/{s3_key}")
    except ClientError as e:
        logger.error(f"Upload failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
