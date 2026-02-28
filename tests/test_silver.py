"""Tests for the silver Delta Lake layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from delta.tables import DeltaTable
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def _make_bronze_df(spark: SparkSession, n: int = 200):
    rng = np.random.default_rng(0)
    df_pd = pd.DataFrame(
        {
            "event_id": np.arange(n, dtype=np.int64),
            "user_id": rng.integers(0, 1_000, n).astype(np.int32),
            "event_timestamp": pd.date_range(
                "2024-01-01", periods=n, freq="1h"
            ),
            "event_type": rng.choice(["purchase", "view"], n),
            "product_id": rng.integers(0, 100, n).astype(np.int32),
            "price": np.round(rng.uniform(1.0, 100.0, n), 2),
            "quantity": rng.integers(1, 5, n).astype(np.int32),
            "region": rng.choice(["north", "south"], n),
        }
    )
    return spark.createDataFrame(df_pd)


def test_silver_write_and_read(spark: SparkSession, tmp_path: Path) -> None:
    path = str(tmp_path / "silver" / "events")
    _make_bronze_df(spark).write.format("delta").mode("overwrite").save(path)

    df = spark.read.format("delta").load(path)
    assert df.count() == 200


def test_silver_time_travel(spark: SparkSession, tmp_path: Path) -> None:
    path = str(tmp_path / "silver" / "events")
    bronze = _make_bronze_df(spark)

    bronze.write.format("delta").mode("overwrite").save(path)  # version 0
    bronze.limit(50).write.format("delta").mode("append").save(
        path
    )  # version 1

    v0 = spark.read.format("delta").option("versionAsOf", 0).load(path)
    current = spark.read.format("delta").load(path)

    assert v0.count() == 200
    assert current.count() == 250


def test_silver_schema_evolution(spark: SparkSession, tmp_path: Path) -> None:
    path = str(tmp_path / "silver" / "events")
    _make_bronze_df(spark).write.format("delta").mode("overwrite").save(path)

    df = spark.read.format("delta").load(path)
    df_evolved = df.withColumn("discount", F.lit(None).cast("double"))
    df_evolved.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).save(path)

    result = spark.read.format("delta").load(path)
    assert "discount" in {f.name for f in result.schema.fields}


def test_silver_upsert(spark: SparkSession, tmp_path: Path) -> None:
    path = str(tmp_path / "silver" / "events")
    bronze = _make_bronze_df(spark)
    bronze.write.format("delta").mode("overwrite").save(path)

    late = bronze.limit(10).withColumn("price", F.col("price") * 0.5)
    delta_table = DeltaTable.forPath(spark, path)
    (
        delta_table.alias("t")
        .merge(late.alias("s"), "t.event_id = s.event_id")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )

    result = spark.read.format("delta").load(path)
    assert result.count() == 200  # no new rows — all matched


def test_silver_filters_invalid_rows(
    spark: SparkSession, tmp_path: Path
) -> None:
    path = str(tmp_path / "silver" / "events")
    bronze = _make_bronze_df(spark)

    # Add bad rows with price=0 and quantity=0
    bad = spark.createDataFrame(
        pd.DataFrame(
            {
                "event_id": [9999],
                "user_id": [1],
                "event_timestamp": pd.to_datetime(["2024-06-01"]),
                "event_type": ["view"],
                "product_id": [1],
                "price": [0.0],
                "quantity": [0],
                "region": ["north"],
            }
        )
    )
    combined = bronze.union(bad)

    cleaned = combined.filter(F.col("price") > 0).filter(F.col("quantity") > 0)
    cleaned.write.format("delta").mode("overwrite").save(path)

    result = spark.read.format("delta").load(path)
    assert result.count() == 200
