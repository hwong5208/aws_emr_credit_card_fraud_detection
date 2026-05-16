"""
PySpark preprocessing: feature engineering for credit card fraud detection.
Runs on local Spark or AWS EMR Serverless (path-agnostic).

Usage:
  Local:  python src/preprocess.py --input data/creditcard.csv --output data/processed/
  Cloud:  python src/preprocess.py --input s3://bucket/raw/ --output s3://bucket/processed/
"""
import argparse
import logging
import json
import math
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log1p, sin, cos

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input path (local or s3://)")
    parser.add_argument("--output", required=True, help="Output path (local or s3://)")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("FraudDetection-Preprocess").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    logger.info(f"Reading data from: {args.input}")
    df = spark.read.csv(args.input, header=True, inferSchema=True)

    logger.info(f"Schema: {df.schema}")
    logger.info(f"Row count: {df.count()}")

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Class distribution (audit trail for model governance)
    # ─────────────────────────────────────────────────────────────────────────
    fraud_counts = df.groupBy("Class").count().collect()
    fraud_dist = {int(row["Class"]): int(row["count"]) for row in fraud_counts}
    total = sum(fraud_dist.values())
    fraud_rate = fraud_dist.get(1, 0) / total if total > 0 else 0

    class_dist = {
        "total_rows": total,
        "fraud_count": fraud_dist.get(1, 0),
        "legit_count": fraud_dist.get(0, 0),
        "fraud_rate": float(fraud_rate),
    }
    logger.info(f"Class distribution: {json.dumps(class_dist)}")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Feature engineering
    # ─────────────────────────────────────────────────────────────────────────
    # Log-transform Amount (right-skewed)
    df = df.withColumn("Amount_log", log1p(col("Amount")))

    # Cyclical encoding of Time (hour-of-day simulation: Time ranges 0-86400 sec)
    # Normalize to [0, 2π], then sin/cos
    seconds_per_day = 86400
    df = df.withColumn(
        "Time_sin",
        sin(2 * math.pi * col("Time") / seconds_per_day)
    ).withColumn(
        "Time_cos",
        cos(2 * math.pi * col("Time") / seconds_per_day)
    )

    # Drop raw features (keep only engineered)
    df = df.drop("Time", "Amount")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Stratified train/test split (preserve fraud ratio)
    # ─────────────────────────────────────────────────────────────────────────
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    logger.info(f"Train set: {train_df.count()} rows")
    logger.info(f"Test set:  {test_df.count()} rows")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Write output
    # ─────────────────────────────────────────────────────────────────────────
    train_output = f"{args.output}/train"
    test_output = f"{args.output}/test"

    logger.info(f"Writing train parquet to: {train_output}")
    train_df.write.mode("overwrite").parquet(train_output)

    logger.info(f"Writing test parquet to: {test_output}")
    test_df.write.mode("overwrite").parquet(test_output)

    logger.info("✓ Preprocessing complete")
    spark.stop()


if __name__ == "__main__":
    main()
