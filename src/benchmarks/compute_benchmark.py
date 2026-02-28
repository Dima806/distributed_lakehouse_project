"""Spark vs Ray compute benchmark: equivalent aggregation workload."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import psutil
import ray

from src.utils.config import cfg

logger = logging.getLogger(__name__)

SILVER_DIR: str = cfg.paths.silver
RAW_DIR: Path = Path(cfg.paths.raw)


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def run_spark_aggregation() -> dict[str, float]:
    from pyspark.sql import functions as F

    from src.utils.spark_session import get_spark

    spark = get_spark("Benchmark-SparkAgg")
    events = spark.read.format("delta").load(f"{SILVER_DIR}/events")

    mem_before = _rss_mb()
    t0 = time.perf_counter()

    result = (
        events.withColumn("revenue", F.col("price") * F.col("quantity"))
        .groupBy("region", "event_type")
        .agg(
            F.count("*").alias("count"),
            F.sum("revenue").alias("total_revenue"),
            F.avg("price").alias("avg_price"),
        )
    )
    out = f"{cfg.paths.tmp}/bench_spark_agg"
    result.write.mode("overwrite").parquet(out)

    elapsed = time.perf_counter() - t0
    mem_delta = _rss_mb() - mem_before
    logger.info("Spark aggregation: %.2fs, Δmem=%.0fMB", elapsed, mem_delta)
    return {"time": elapsed, "memory_mb": mem_delta}


@ray.remote
def _agg_chunk(filepath: str) -> dict[str, float]:
    import pandas as pd

    df = pd.read_parquet(filepath)
    df["revenue"] = df["price"] * df["quantity"]
    agg = df.groupby(["region", "event_type"]).agg(
        count=("revenue", "count"),
        total_revenue=("revenue", "sum"),
        avg_price=("price", "mean"),
    )
    return {
        "total_revenue": float(agg["total_revenue"].sum()),
        "count": int(agg["count"].sum()),
    }


def run_ray_aggregation() -> dict[str, float]:
    ray.init(num_cpus=cfg.ray.num_cpus, ignore_reinit_error=True)

    files = sorted(RAW_DIR.resolve().glob("events_*.parquet"))
    if not files:
        logger.warning("No raw event files — run `make generate-data` first.")
        return {"time": 0.0, "memory_mb": 0.0}

    mem_before = _rss_mb()
    t0 = time.perf_counter()

    futures = [_agg_chunk.remote(str(f)) for f in files]
    results = ray.get(futures)

    elapsed = time.perf_counter() - t0
    mem_delta = _rss_mb() - mem_before

    total_revenue = sum(r["total_revenue"] for r in results)
    logger.info(
        "Ray aggregation: %.2fs, Δmem=%.0fMB, revenue=%.0f",
        elapsed,
        mem_delta,
        total_revenue,
    )
    return {"time": elapsed, "memory_mb": mem_delta}


def run() -> dict[str, dict[str, float]]:
    return {
        "spark": run_spark_aggregation(),
        "ray": run_ray_aggregation(),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for engine, m in run().items():
        print(
            f"{engine}: time={m['time']:.2f}s  memory={m['memory_mb']:.0f}MB"
        )
