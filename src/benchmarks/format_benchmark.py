"""Columnar format benchmark: Parquet vs ORC vs Delta."""

from __future__ import annotations

import logging
import os
import time

from pyspark.sql import functions as F

from src.utils.config import cfg
from src.utils.spark_session import get_spark

logger = logging.getLogger(__name__)

SILVER_DIR: str = cfg.paths.silver
BENCH_DIR: str = cfg.paths.bench_formats


def _dir_size_mb(path: str) -> float:
    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(path)
        for f in files
    )
    return total / (1024 * 1024)


def run() -> dict[str, dict[str, float]]:
    spark = get_spark("Benchmark-Formats")
    events = spark.read.format("delta").load(f"{SILVER_DIR}/events")
    filter_region: str = cfg.benchmarks.filter_region
    formats: list[str] = cfg.benchmarks.formats

    results: dict[str, dict[str, float]] = {}

    for fmt in formats:
        path = f"{BENCH_DIR}/{fmt}"

        t0 = time.perf_counter()
        events.write.mode("overwrite").format(fmt).save(path)
        write_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        spark.read.format(fmt).load(path).agg(F.count("*")).collect()
        full_scan = time.perf_counter() - t0

        t0 = time.perf_counter()
        (
            spark.read.format(fmt)
            .load(path)
            .filter(F.col("region") == filter_region)
            .agg(F.count("*"))
            .collect()
        )
        filtered_scan = time.perf_counter() - t0

        size_mb = _dir_size_mb(path)
        results[fmt] = {
            "write_time": write_time,
            "full_scan": full_scan,
            "filtered_scan": filtered_scan,
            "size_mb": size_mb,
        }
        logger.info(
            "%s: write=%.2fs scan=%.2fs filter=%.2fs size=%.1fMB",
            fmt,
            write_time,
            full_scan,
            filtered_scan,
            size_mb,
        )

    # Delta already written during silver phase — reads only
    silver_events = f"{SILVER_DIR}/events"
    t0 = time.perf_counter()
    (
        spark.read.format("delta")
        .load(silver_events)
        .agg(F.count("*"))
        .collect()
    )
    delta_full = time.perf_counter() - t0

    t0 = time.perf_counter()
    (
        spark.read.format("delta")
        .load(silver_events)
        .filter(F.col("region") == filter_region)
        .agg(F.count("*"))
        .collect()
    )
    delta_filtered = time.perf_counter() - t0

    results["delta"] = {
        "write_time": 0.0,
        "full_scan": delta_full,
        "filtered_scan": delta_filtered,
        "size_mb": _dir_size_mb(silver_events),
    }
    logger.info(
        "delta: scan=%.2fs filter=%.2fs size=%.1fMB",
        delta_full,
        delta_filtered,
        results["delta"]["size_mb"],
    )

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
