"""Spark join job: broadcast join vs shuffle join comparison."""

from __future__ import annotations

import logging
import time

from pyspark.sql import functions as F

from src.utils.config import cfg
from src.utils.spark_session import get_spark

logger = logging.getLogger(__name__)

SILVER_DIR: str = cfg.paths.silver
BRONZE_DIR: str = cfg.paths.bronze


def run_broadcast_join() -> float:
    spark = get_spark("SparkJob-BroadcastJoin")
    events = spark.read.format("delta").load(f"{SILVER_DIR}/events")
    products = spark.read.parquet(f"{BRONZE_DIR}/products")

    t0 = time.perf_counter()
    result = (
        events.join(F.broadcast(products), on="product_id", how="left")
        .groupBy("category")
        .agg(F.sum(F.col("price") * F.col("quantity")).alias("revenue"))
    )
    out = f"{cfg.paths.tmp}/broadcast_join_result"
    result.write.mode("overwrite").parquet(out)

    elapsed = time.perf_counter() - t0
    logger.info("Broadcast join: %.2fs", elapsed)
    return elapsed


def run_shuffle_join() -> float:
    spark = get_spark("SparkJob-ShuffleJoin")
    events = spark.read.format("delta").load(f"{SILVER_DIR}/events")
    products = spark.read.parquet(f"{BRONZE_DIR}/products")

    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

    t0 = time.perf_counter()
    result = (
        events.join(products, on="product_id", how="left")
        .groupBy("category")
        .agg(F.sum(F.col("price") * F.col("quantity")).alias("revenue"))
    )
    out = f"{cfg.paths.tmp}/shuffle_join_result"
    result.write.mode("overwrite").parquet(out)

    elapsed = time.perf_counter() - t0
    logger.info("Shuffle join: %.2fs", elapsed)
    return elapsed


def run() -> dict[str, float]:
    return {
        "broadcast_join": run_broadcast_join(),
        "shuffle_join": run_shuffle_join(),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
