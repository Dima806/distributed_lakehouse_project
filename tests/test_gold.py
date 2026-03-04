"""Tests for gold layer aggregations."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

_SILVER_SCHEMA = StructType(
    [
        StructField("event_id", LongType(), False),
        StructField("user_id", IntegerType(), True),
        StructField("event_timestamp", TimestampType(), True),
        StructField("event_type", StringType(), True),
        StructField("product_id", IntegerType(), True),
        StructField("price", DoubleType(), True),
        StructField("quantity", IntegerType(), True),
        StructField("region", StringType(), True),
        StructField("discount", DoubleType(), True),
    ]
)


@pytest.fixture()
def silver_delta(spark: SparkSession, tmp_path: Path) -> str:
    rng = np.random.default_rng(0)
    n = 100
    base = datetime(2024, 1, 1)
    user_ids = rng.integers(0, 100, n).tolist()
    event_types = rng.choice(["purchase", "view", "add_to_cart"], n).tolist()
    product_ids = rng.integers(0, 50, n).tolist()
    prices = [round(float(p), 2) for p in rng.uniform(1.0, 100.0, n)]
    quantities = rng.integers(1, 5, n).tolist()
    regions = rng.choice(["north", "south", "east"], n).tolist()
    rows = [
        (
            int(i),
            user_ids[i],
            base + timedelta(hours=i),
            event_types[i],
            product_ids[i],
            prices[i],
            quantities[i],
            regions[i],
            None,  # discount
        )
        for i in range(n)
    ]
    path = str(tmp_path / "silver" / "events")
    spark.createDataFrame(rows, _SILVER_SCHEMA).write.format("delta").mode(
        "overwrite"
    ).save(path)
    return path


def test_revenue_by_region_and_type(
    spark: SparkSession, silver_delta: str
) -> None:
    df = spark.read.format("delta").load(silver_delta)
    result = (
        df.withColumn("revenue", F.col("price") * F.col("quantity"))
        .groupBy("region", "event_type")
        .agg(
            F.sum("revenue").alias("total_revenue"),
            F.count("event_id").alias("event_count"),
        )
    )
    assert result.count() > 0
    assert result.filter(F.col("total_revenue") > 0).count() > 0


def test_daily_event_counts(spark: SparkSession, silver_delta: str) -> None:
    df = spark.read.format("delta").load(silver_delta)
    result = (
        df.withColumn("date", F.to_date("event_timestamp"))
        .groupBy("date", "region")
        .agg(F.count("event_id").alias("event_count"))
    )
    assert result.count() > 0
    assert result.filter(F.col("event_count") > 0).count() > 0


def test_top_products_limit(spark: SparkSession, silver_delta: str) -> None:
    df = spark.read.format("delta").load(silver_delta)
    result = (
        df.filter(F.col("event_type") == "purchase")
        .withColumn("revenue", F.col("price") * F.col("quantity"))
        .groupBy("product_id")
        .agg(F.sum("revenue").alias("total_revenue"))
        .orderBy(F.desc("total_revenue"))
        .limit(10)
    )
    assert result.count() <= 10
    assert result.count() > 0
