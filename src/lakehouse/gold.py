"""Gold layer: aggregated analytics tables written as Delta."""

from __future__ import annotations

import logging

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.utils.config import cfg
from src.utils.spark_session import get_spark

logger = logging.getLogger(__name__)

SILVER_DIR: str = cfg.paths.silver
BRONZE_DIR: str = cfg.paths.bronze
GOLD_DIR: str = cfg.paths.gold


def revenue_by_region_and_type() -> None:
    spark = get_spark("Gold-Revenue")
    events = spark.read.format("delta").load(f"{SILVER_DIR}/events")

    df = (
        events.withColumn("revenue", F.col("price") * F.col("quantity"))
        .groupBy("region", "event_type")
        .agg(
            F.sum("revenue").alias("total_revenue"),
            F.count("event_id").alias("event_count"),
        )
        .orderBy("region", "event_type")
    )

    df.write.format("delta").mode("overwrite").save(
        f"{GOLD_DIR}/revenue_by_region_type"
    )
    logger.info("Written gold: revenue_by_region_type (%d rows)", df.count())


def daily_event_counts() -> None:
    spark = get_spark("Gold-DailyCounts")
    events = spark.read.format("delta").load(f"{SILVER_DIR}/events")

    df = (
        events.withColumn("date", F.to_date("event_timestamp"))
        .groupBy("date", "region")
        .agg(F.count("event_id").alias("event_count"))
        .orderBy("date", "region")
    )

    df.write.format("delta").mode("overwrite").save(
        f"{GOLD_DIR}/daily_event_counts"
    )
    logger.info("Written gold: daily_event_counts (%d rows)", df.count())


def top_products_by_revenue() -> None:
    spark = get_spark("Gold-TopProducts")
    events = spark.read.format("delta").load(f"{SILVER_DIR}/events")
    products = spark.read.parquet(f"{BRONZE_DIR}/products")

    revenue_per_product = (
        events.filter(F.col("event_type") == "purchase")
        .withColumn("revenue", F.col("price") * F.col("quantity"))
        .groupBy("product_id")
        .agg(F.sum("revenue").alias("total_revenue"))
    )

    df = (
        revenue_per_product.join(products, on="product_id", how="left")
        .orderBy(F.desc("total_revenue"))
        .limit(cfg.gold.top_products_limit)
    )

    df.write.format("delta").mode("overwrite").save(f"{GOLD_DIR}/top_products")
    logger.info("Written gold: top_products (top 10)")


def rolling_revenue_by_region() -> None:
    """7-day rolling revenue window per region."""
    spark = get_spark("Gold-RollingRevenue")
    events = spark.read.format("delta").load(f"{SILVER_DIR}/events")

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

    df = daily.withColumn(
        "rolling_7d_revenue", F.sum("daily_revenue").over(window_7d)
    )
    df.write.format("delta").mode("overwrite").save(
        f"{GOLD_DIR}/rolling_revenue"
    )
    logger.info("Written gold: rolling_revenue")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    revenue_by_region_and_type()
    daily_event_counts()
    top_products_by_revenue()
    rolling_revenue_by_region()
