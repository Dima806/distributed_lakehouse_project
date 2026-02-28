"""Silver layer: Delta Lake ACID transactions, schema evolution, upserts."""

from __future__ import annotations

import logging

from delta.tables import DeltaTable
from pyspark.sql import functions as F

from src.utils.config import cfg
from src.utils.spark_session import get_spark

logger = logging.getLogger(__name__)

BRONZE_DIR: str = cfg.paths.bronze
SILVER_DIR: str = cfg.paths.silver


def write_initial_silver() -> int:
    """Clean bronze events and write as Delta table (version 0)."""
    spark = get_spark("Silver-Init")

    df = (
        spark.read.parquet(f"{BRONZE_DIR}/events")
        .dropna(
            subset=["event_id", "user_id", "event_timestamp", "event_type"]
        )
        .filter(F.col("price") > 0)
        .filter(F.col("quantity") > 0)
        .withColumn(
            "event_timestamp", F.col("event_timestamp").cast("timestamp")
        )
        .withColumn("price", F.col("price").cast("double"))
        .withColumn("quantity", F.col("quantity").cast("int"))
    )

    df.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).partitionBy("region").save(f"{SILVER_DIR}/events")
    row_count = df.count()
    logger.info(
        "Written silver events (version 0): %d rows → %s/events",
        row_count,
        SILVER_DIR,
    )
    return row_count


def evolve_schema_add_discount() -> None:
    """Schema evolution: add nullable discount column."""
    spark = get_spark("Silver-Evolve")

    df = spark.read.format("delta").load(f"{SILVER_DIR}/events")
    df_with_discount = df.withColumn("discount", F.lit(None).cast("double"))

    df_with_discount.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).save(f"{SILVER_DIR}/events")
    logger.info("Schema evolved: added 'discount' column")


def upsert_late_arriving() -> None:
    """ACID upsert: merge a late-arriving discounted batch into silver."""
    spark = get_spark("Silver-Upsert")

    existing = spark.read.format("delta").load(f"{SILVER_DIR}/events")
    late_df = (
        existing.limit(cfg.silver.late_arriving_limit)
        .withColumn(
            "price", F.col("price") * cfg.silver.late_discount_multiplier
        )
        .withColumn("discount", F.lit(cfg.silver.late_discount_value))
    )

    delta_table = DeltaTable.forPath(spark, f"{SILVER_DIR}/events")
    (
        delta_table.alias("target")
        .merge(late_df.alias("source"), "target.event_id = source.event_id")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
    logger.info("Upsert complete: merged 1,000 late-arriving records")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    write_initial_silver()
    evolve_schema_add_discount()
    upsert_late_arriving()
