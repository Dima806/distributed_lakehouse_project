"""Tests for gold layer aggregations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


@pytest.fixture()
def silver_delta(spark: SparkSession, tmp_path: Path) -> str:
    rng = np.random.default_rng(0)
    n = 500
    df_pd = pd.DataFrame(
        {
            "event_id": np.arange(n, dtype=np.int64),
            "user_id": rng.integers(0, 100, n).astype(np.int32),
            "event_timestamp": pd.date_range(
                "2024-01-01", periods=n, freq="1h"
            ),
            "event_type": rng.choice(["purchase", "view", "add_to_cart"], n),
            "product_id": rng.integers(0, 50, n).astype(np.int32),
            "price": np.round(rng.uniform(1.0, 100.0, n), 2),
            "quantity": rng.integers(1, 5, n).astype(np.int32),
            "region": rng.choice(["north", "south", "east"], n),
            "discount": np.full(n, np.nan),
        }
    )
    path = str(tmp_path / "silver" / "events")
    spark.createDataFrame(df_pd).write.format("delta").mode("overwrite").save(
        path
    )
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
