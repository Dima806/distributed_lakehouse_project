"""Tests for the silver Delta Lake layer."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from delta.tables import DeltaTable
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

_BRONZE_SCHEMA = StructType(
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

_N = 50  # small enough to be fast, large enough for meaningful tests


def _make_rows(n: int = _N) -> list:
    rng = np.random.default_rng(0)
    base = datetime(2024, 1, 1)
    user_ids = rng.integers(0, 1_000, n).tolist()
    event_types = rng.choice(["purchase", "view"], n).tolist()
    product_ids = rng.integers(0, 100, n).tolist()
    prices = [round(float(p), 2) for p in rng.uniform(1.0, 100.0, n)]
    quantities = rng.integers(1, 5, n).tolist()
    regions = rng.choice(["north", "south"], n).tolist()
    return [
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


@pytest.fixture(scope="module")
def bronze_rows() -> list:
    """Pre-computed rows shared across all silver tests in this module."""
    return _make_rows(_N)


def test_silver_write_and_read(
    spark: SparkSession, tmp_path: Path, bronze_rows: list
) -> None:
    path = str(tmp_path / "silver" / "events")
    spark.createDataFrame(bronze_rows, _BRONZE_SCHEMA).write.format(
        "delta"
    ).mode("overwrite").save(path)

    df = spark.read.format("delta").load(path)
    assert df.count() == _N


def test_silver_time_travel(
    spark: SparkSession, tmp_path: Path, bronze_rows: list
) -> None:
    path = str(tmp_path / "silver" / "events")
    bronze = spark.createDataFrame(bronze_rows, _BRONZE_SCHEMA)

    bronze.write.format("delta").mode("overwrite").save(path)  # version 0
    bronze.limit(25).write.format("delta").mode("append").save(
        path
    )  # version 1

    v0 = spark.read.format("delta").option("versionAsOf", 0).load(path)
    current = spark.read.format("delta").load(path)

    assert v0.count() == _N
    assert current.count() == _N + 25


def test_silver_schema_evolution(
    spark: SparkSession, tmp_path: Path, bronze_rows: list
) -> None:
    path = str(tmp_path / "silver" / "events")
    spark.createDataFrame(bronze_rows, _BRONZE_SCHEMA).write.format(
        "delta"
    ).mode("overwrite").save(path)

    df = spark.read.format("delta").load(path)
    df_evolved = df.withColumn("discount", F.lit(None).cast("double"))
    df_evolved.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).save(path)

    result = spark.read.format("delta").load(path)
    assert "discount" in {f.name for f in result.schema.fields}


def test_silver_upsert(
    spark: SparkSession, tmp_path: Path, bronze_rows: list
) -> None:
    path = str(tmp_path / "silver" / "events")
    bronze = spark.createDataFrame(bronze_rows, _BRONZE_SCHEMA)
    bronze.write.format("delta").mode("overwrite").save(path)

    late = bronze.limit(5).withColumn("price", F.col("price") * 0.5)
    delta_table = DeltaTable.forPath(spark, path)
    (
        delta_table.alias("t")
        .merge(late.alias("s"), "t.event_id = s.event_id")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )

    result = spark.read.format("delta").load(path)
    assert result.count() == _N  # no new rows — all matched


def test_silver_filters_invalid_rows(
    spark: SparkSession, tmp_path: Path, bronze_rows: list
) -> None:
    path = str(tmp_path / "silver" / "events")
    bronze = spark.createDataFrame(bronze_rows, _BRONZE_SCHEMA)

    # Add bad rows with price=0 and quantity=0
    bad = spark.createDataFrame(
        [(9999, 1, datetime(2024, 6, 1), "view", 1, 0.0, 0, "north")],
        _BRONZE_SCHEMA,
    )
    combined = bronze.union(bad)

    cleaned = combined.filter(F.col("price") > 0).filter(F.col("quantity") > 0)
    cleaned.write.format("delta").mode("overwrite").save(path)

    result = spark.read.format("delta").load(path)
    assert result.count() == _N
