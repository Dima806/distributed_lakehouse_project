"""Bronze layer: raw Parquet ingestion, partitioned by region."""

from __future__ import annotations

import logging

from src.utils.config import cfg
from src.utils.spark_session import get_spark

logger = logging.getLogger(__name__)

RAW_DIR: str = cfg.paths.raw
BRONZE_DIR: str = cfg.paths.bronze


def ingest_events() -> int:
    spark = get_spark("Bronze-Events")
    logger.info("Reading raw events from %s", RAW_DIR)

    df = spark.read.parquet(f"{RAW_DIR}/events_*.parquet")
    row_count = df.count()
    logger.info("Loaded %d rows", row_count)

    df.write.mode("overwrite").partitionBy("region").parquet(
        f"{BRONZE_DIR}/events"
    )
    logger.info("Written bronze events → %s/events", BRONZE_DIR)
    return row_count


def ingest_dimensions() -> None:
    spark = get_spark("Bronze-Dims")
    for table in ("users", "products"):
        df = spark.read.parquet(f"{RAW_DIR}/{table}.parquet")
        df.write.mode("overwrite").parquet(f"{BRONZE_DIR}/{table}")
        logger.info("Written bronze %s", table)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_events()
    ingest_dimensions()
