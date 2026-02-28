"""Spark jobs orchestrator: runs all jobs and prints a timing summary."""

from __future__ import annotations

import logging

from src.spark_jobs import (
    aggregations,
    joins,
    partition_tuning,
    window_functions,
)

logger = logging.getLogger(__name__)


def main() -> dict[str, float]:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    print("\n" + "=" * 60)
    print("SPARK JOBS BENCHMARK")
    print("=" * 60)

    results: dict[str, float] = {}
    results["aggregations"] = aggregations.run()
    results.update(joins.run())
    results["window_functions"] = window_functions.run()
    results.update(partition_tuning.run())

    print("\n--- Spark Timing Summary ---")
    for task, elapsed in results.items():
        print(f"  {task:<35} {elapsed:.2f}s")
    print()
    return results


if __name__ == "__main__":
    main()
