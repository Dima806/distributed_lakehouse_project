"""Smoke tests for benchmark modules."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
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

_SCHEMA = StructType(
    [
        StructField("event_id", LongType(), False),
        StructField("user_id", IntegerType(), True),
        StructField("event_timestamp", TimestampType(), True),
        StructField("event_type", StringType(), True),
        StructField("product_id", IntegerType(), True),
        StructField("price", DoubleType(), True),
        StructField("quantity", IntegerType(), True),
        StructField("region", StringType(), True),
    ]
)


def _write_small_silver(
    spark: SparkSession, tmp_path: Path, n: int = 40
) -> str:
    rng = np.random.default_rng(0)
    base = datetime(2024, 1, 1)
    user_ids = rng.integers(0, 100, n).tolist()
    event_types = rng.choice(["purchase", "view"], n).tolist()
    product_ids = rng.integers(0, 50, n).tolist()
    prices = [round(float(p), 2) for p in rng.uniform(1.0, 100.0, n)]
    quantities = rng.integers(1, 5, n).tolist()
    regions = rng.choice(["north", "south"], n).tolist()
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
        )
        for i in range(n)
    ]
    path = str(tmp_path / "silver" / "events")
    spark.createDataFrame(rows, _SCHEMA).write.format("delta").mode(
        "overwrite"
    ).save(path)
    return path


def test_parquet_write_read_roundtrip(
    spark: SparkSession, tmp_path: Path
) -> None:
    silver_path = _write_small_silver(spark, tmp_path)
    out_path = str(tmp_path / "bench" / "parquet")

    df = spark.read.format("delta").load(silver_path)
    df.write.mode("overwrite").format("parquet").save(out_path)

    result = spark.read.format("parquet").load(out_path)
    assert result.count() == 40


def test_orc_write_read_roundtrip(spark: SparkSession, tmp_path: Path) -> None:
    silver_path = _write_small_silver(spark, tmp_path)
    out_path = str(tmp_path / "bench" / "orc")

    df = spark.read.format("delta").load(silver_path)
    df.write.mode("overwrite").format("orc").save(out_path)

    result = spark.read.format("orc").load(out_path)
    assert result.count() == 40


def test_filtered_scan_returns_subset(
    spark: SparkSession, tmp_path: Path
) -> None:
    silver_path = _write_small_silver(spark, tmp_path, n=80)
    out_path = str(tmp_path / "bench" / "parquet_filter")

    df = spark.read.format("delta").load(silver_path)
    df.write.mode("overwrite").format("parquet").save(out_path)

    total = spark.read.format("parquet").load(out_path).count()
    filtered = (
        spark.read.format("parquet")
        .load(out_path)
        .filter(F.col("region") == "north")
        .count()
    )
    assert filtered < total
    assert filtered > 0
