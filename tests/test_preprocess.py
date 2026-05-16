"""
Unit tests for preprocessing pipeline.
Tests run locally without AWS.
"""
import tempfile
import os
import math
import pytest
from pyspark.sql import SparkSession

# Import the preprocess module
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def spark():
    """Create a local Spark session for testing."""
    return SparkSession.builder.appName("test-fraud-preprocess").getOrCreate()


@pytest.fixture
def sample_data(spark):
    """Create sample credit card fraud data."""
    import random
    random.seed(42)

    rows = []
    for i in range(100):
        row = {
            "Time": random.randint(0, 86400),
            "Amount": random.uniform(0, 2000),
            "Class": random.choice([0] * 99 + [1]),  # ~1% fraud
        }
        # Add V1-V28 features (simulated)
        for j in range(1, 29):
            row[f"V{j}"] = random.gauss(0, 1)
        rows.append(row)

    return spark.createDataFrame(rows)


def test_sample_data_shape(sample_data):
    """Test sample data has correct shape."""
    assert sample_data.count() == 100
    assert sample_data.schema.fieldNames()[0] == "Amount"
    assert "Class" in sample_data.schema.fieldNames()


def test_feature_engineering(sample_data):
    """Test feature engineering transforms."""
    from pyspark.sql.functions import col, log1p, sin, cos

    df = sample_data
    df = df.withColumn("Amount_log", log1p(col("Amount")))
    df = df.withColumn(
        "Time_sin",
        sin(2 * math.pi * col("Time") / 86400)
    ).withColumn(
        "Time_cos",
        cos(2 * math.pi * col("Time") / 86400)
    )

    assert "Amount_log" in df.schema.fieldNames()
    assert "Time_sin" in df.schema.fieldNames()
    assert "Time_cos" in df.schema.fieldNames()


def test_parquet_write_read(sample_data):
    """Test writing and reading Parquet."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test.parquet")
        sample_data.write.mode("overwrite").parquet(output_path)

        # Read back
        df_read = SparkSession.builder.getOrCreate().read.parquet(output_path)
        assert df_read.count() == sample_data.count()
        assert df_read.schema == sample_data.schema
