"""Time travel: query the silver Delta table at historical versions."""

from __future__ import annotations

import logging

from src.utils.config import cfg
from src.utils.spark_session import get_spark

logger = logging.getLogger(__name__)

SILVER_DIR: str = cfg.paths.silver


def show_version_diff() -> None:
    spark = get_spark("TimeTravel")

    v0 = (
        spark.read.format("delta")
        .option("versionAsOf", 0)
        .load(f"{SILVER_DIR}/events")
    )
    current = spark.read.format("delta").load(f"{SILVER_DIR}/events")

    print("\n=== Time Travel: Version 0 ===")
    print(f"  Schema  : {[f.name for f in v0.schema.fields]}")
    print(f"  Row count: {v0.count():,}")

    print("\n=== Time Travel: Current Version ===")
    print(f"  Schema  : {[f.name for f in current.schema.fields]}")
    print(f"  Row count: {current.count():,}")

    print("\n=== Delta Table History ===")
    spark.sql(f"DESCRIBE HISTORY delta.`{SILVER_DIR}/events`").show(
        truncate=False
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    show_version_diff()
