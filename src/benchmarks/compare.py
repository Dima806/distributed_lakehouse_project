"""Unified benchmark runner: formats + Spark vs Ray compute."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.benchmarks import compute_benchmark, format_benchmark
from src.utils.config import cfg

logger = logging.getLogger(__name__)

RESULTS_PATH: Path = Path(cfg.paths.benchmark_results)


def _print_format_table(results: dict[str, dict[str, float]]) -> None:
    print("\n=== Columnar Format Benchmark ===")
    header = (
        f"{'Format':<10} {'Write(s)':<10}"
        f" {'Scan(s)':<10} {'Filter(s)':<12} {'Size(MB)':<10}"
    )
    print(header)
    print("-" * 55)
    for fmt, m in results.items():
        print(
            f"{fmt:<10} {m['write_time']:<10.2f}"
            f" {m['full_scan']:<10.2f}"
            f" {m['filtered_scan']:<12.2f}"
            f" {m['size_mb']:<10.1f}"
        )


def _print_compute_table(results: dict[str, dict[str, float]]) -> None:
    print("\n=== Spark vs Ray Compute Benchmark ===")
    print(f"{'Engine':<10} {'Task':<25} {'Time(s)':<10} {'ΔMem(MB)':<12}")
    print("-" * 60)
    for engine, m in results.items():
        print(
            f"{engine:<10} {'aggregation':<25}"
            f" {m['time']:<10.2f} {m['memory_mb']:<12.0f}"
        )


def main() -> None:
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("DISTRIBUTED LAKEHOUSE BENCHMARK")
    print("=" * 60)

    fmt_results = format_benchmark.run()
    compute_results = compute_benchmark.run()

    _print_format_table(fmt_results)
    _print_compute_table(compute_results)

    RESULTS_PATH.write_text(
        json.dumps(
            {
                "format_benchmark": fmt_results,
                "compute_benchmark": compute_results,
            },
            indent=2,
        )
    )
    logger.info("Results written to %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
