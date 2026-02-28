"""Shared Spark session factory."""

from __future__ import annotations

from pyspark.sql import SparkSession

from src.utils.config import cfg


def get_spark(app_name: str = "LakehouseLab") -> SparkSession:
    from delta import configure_spark_with_delta_pip

    builder = (
        SparkSession.builder.master(cfg.spark.master)
        .appName(app_name)
        .config("spark.executor.memory", cfg.spark.executor_memory)
        .config("spark.driver.memory", cfg.spark.driver_memory)
        .config(
            "spark.sql.shuffle.partitions", str(cfg.spark.shuffle_partitions)
        )
        .config(
            "spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension"
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()
