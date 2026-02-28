"""Smoke tests for benchmark modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def _write_small_silver(
    spark: SparkSession, tmp_path: Path, n: int = 100
) -> str:
    rng = np.random.default_rng(0)
    df_pd = pd.DataFrame(
        {
            "event_id": np.arange(n, dtype=np.int64),
            "user_id": rng.integers(0, 100, n).astype(np.int32),
            "event_timestamp": pd.date_range(
                "2024-01-01", periods=n, freq="1h"
            ),
            "event_type": rng.choice(["purchase", "view"], n),
            "product_id": rng.integers(0, 50, n).astype(np.int32),
            "price": np.round(rng.uniform(1.0, 100.0, n), 2),
            "quantity": rng.integers(1, 5, n).astype(np.int32),
            "region": rng.choice(["north", "south"], n),
        }
    )
    path = str(tmp_path / "silver" / "events")
    spark.createDataFrame(df_pd).write.format("delta").mode("overwrite").save(
        path
    )
    return path


def test_parquet_write_read_roundtrip(
    spark: SparkSession, tmp_path: Path
) -> None:
    silver_path = _write_small_silver(spark, tmp_path)
    out_path = str(tmp_path / "bench" / "parquet")

    df = spark.read.format("delta").load(silver_path)
    df.write.mode("overwrite").format("parquet").save(out_path)

    result = spark.read.format("parquet").load(out_path)
    assert result.count() == 100


def test_orc_write_read_roundtrip(spark: SparkSession, tmp_path: Path) -> None:
    silver_path = _write_small_silver(spark, tmp_path)
    out_path = str(tmp_path / "bench" / "orc")

    df = spark.read.format("delta").load(silver_path)
    df.write.mode("overwrite").format("orc").save(out_path)

    result = spark.read.format("orc").load(out_path)
    assert result.count() == 100


def test_filtered_scan_returns_subset(
    spark: SparkSession, tmp_path: Path
) -> None:
    silver_path = _write_small_silver(spark, tmp_path, n=200)
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
