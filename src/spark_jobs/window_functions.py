"""Spark window function job: rolling revenue and daily rank by region."""

from __future__ import annotations

import logging
import time

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.utils.config import cfg
from src.utils.spark_session import get_spark

logger = logging.getLogger(__name__)

SILVER_DIR: str = cfg.paths.silver


def run() -> float:
    spark = get_spark("SparkJob-Windows")
    events = spark.read.format("delta").load(f"{SILVER_DIR}/events")

    t0 = time.perf_counter()

    daily = (
        events.withColumn("date", F.to_date("event_timestamp"))
        .withColumn("revenue", F.col("price") * F.col("quantity"))
        .groupBy("region", "date")
        .agg(F.sum("revenue").alias("daily_revenue"))
    )

    lookback = cfg.gold.rolling_window_days - 1
    window_7d = (
        Window.partitionBy("region")
        .orderBy(F.col("date").cast("long"))
        .rowsBetween(-lookback, 0)
    )
    rank_window = Window.partitionBy("date").orderBy(F.desc("daily_revenue"))

    result = daily.withColumn(
        "rolling_7d_revenue", F.sum("daily_revenue").over(window_7d)
    ).withColumn("rank_in_day", F.rank().over(rank_window))

    out = f"{cfg.paths.tmp}/window_result"
    result.write.mode("overwrite").parquet(out)

    elapsed = time.perf_counter() - t0
    logger.info("Window functions job: %.2fs", elapsed)
    return elapsed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
