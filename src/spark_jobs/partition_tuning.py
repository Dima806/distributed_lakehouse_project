"""Spark partition tuning: compare shuffle partition counts."""

from __future__ import annotations

import logging
import time

from pyspark.sql import functions as F

from src.utils.config import cfg
from src.utils.spark_session import get_spark

logger = logging.getLogger(__name__)

SILVER_DIR: str = cfg.paths.silver


def run_with_partitions(n_partitions: int) -> float:
    spark = get_spark("SparkJob-PartitionTuning")
    spark.conf.set("spark.sql.shuffle.partitions", str(n_partitions))

    events = spark.read.format("delta").load(f"{SILVER_DIR}/events")

    t0 = time.perf_counter()
    result = (
        events.withColumn("revenue", F.col("price") * F.col("quantity"))
        .groupBy("user_id", "region")
        .agg(F.sum("revenue").alias("user_revenue"))
        .orderBy(F.desc("user_revenue"))
    )
    out = f"{cfg.paths.tmp}/partition_{n_partitions}_result"
    result.write.mode("overwrite").parquet(out)

    elapsed = time.perf_counter() - t0
    logger.info("Partition count=%d: %.2fs", n_partitions, elapsed)
    return elapsed


def run() -> dict[str, float]:
    counts: list[int] = cfg.spark_jobs.partition_counts
    return {f"partitions_{n}": run_with_partitions(n) for n in counts}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
