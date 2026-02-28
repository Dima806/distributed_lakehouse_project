"""Tests for the bronze ingestion layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession


def _make_raw_events(path: Path, n: int = 200) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
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
            "region": rng.choice(["north", "south", "east"], n),
        }
    )
    path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path / "events_0000000000.parquet", index=False)


def test_bronze_events_row_count(spark: SparkSession, tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    bronze_dir = tmp_path / "bronze"
    _make_raw_events(raw_dir, n=200)

    df = spark.read.parquet(str(raw_dir / "events_0000000000.parquet"))
    df.write.mode("overwrite").partitionBy("region").parquet(
        str(bronze_dir / "events")
    )

    result = spark.read.parquet(str(bronze_dir / "events"))
    assert result.count() == 200


def test_bronze_events_partitioned_by_region(
    spark: SparkSession, tmp_path: Path
) -> None:
    raw_dir = tmp_path / "raw"
    bronze_dir = tmp_path / "bronze"
    _make_raw_events(raw_dir, n=200)

    df = spark.read.parquet(str(raw_dir / "events_0000000000.parquet"))
    df.write.mode("overwrite").partitionBy("region").parquet(
        str(bronze_dir / "events")
    )

    # Partition directories must exist
    partition_dirs = [
        p for p in (bronze_dir / "events").iterdir() if p.is_dir()
    ]
    assert len(partition_dirs) > 0
    assert all(d.name.startswith("region=") for d in partition_dirs)


def test_bronze_preserves_schema(spark: SparkSession, tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    bronze_dir = tmp_path / "bronze"
    _make_raw_events(raw_dir, n=100)

    df = spark.read.parquet(str(raw_dir / "events_0000000000.parquet"))
    df.write.mode("overwrite").partitionBy("region").parquet(
        str(bronze_dir / "events")
    )

    result = spark.read.parquet(str(bronze_dir / "events"))
    col_names = {f.name for f in result.schema.fields}
    assert {"event_id", "user_id", "price", "quantity"}.issubset(col_names)
