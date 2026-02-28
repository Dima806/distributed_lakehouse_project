"""Spark aggregation job: group-by aggregations measured end-to-end."""

from __future__ import annotations

import logging
import time

from pyspark.sql import functions as F

from src.utils.config import cfg
from src.utils.spark_session import get_spark

logger = logging.getLogger(__name__)

SILVER_DIR: str = cfg.paths.silver


def run() -> float:
    spark = get_spark("SparkJob-Aggregations")
    events = spark.read.format("delta").load(f"{SILVER_DIR}/events")

    t0 = time.perf_counter()

    result = (
        events.withColumn("revenue", F.col("price") * F.col("quantity"))
        .groupBy("region", "event_type")
        .agg(
            F.count("*").alias("count"),
            F.sum("revenue").alias("total_revenue"),
            F.avg("price").alias("avg_price"),
            F.max("price").alias("max_price"),
        )
    )
    result.write.mode("overwrite").parquet(f"{cfg.paths.tmp}/agg_result")

    elapsed = time.perf_counter() - t0
    logger.info("Aggregation job: %.2fs", elapsed)
    return elapsed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
