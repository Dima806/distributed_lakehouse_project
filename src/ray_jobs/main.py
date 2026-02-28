"""Ray jobs orchestrator: runs all jobs and prints a timing summary."""

from __future__ import annotations

import logging

import ray

from src.ray_jobs import feature_engineering, model_scoring, simulation
from src.utils.config import cfg

logger = logging.getLogger(__name__)


def main() -> dict[str, float]:
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    ray.init(num_cpus=cfg.ray.num_cpus, ignore_reinit_error=True)

    print("\n" + "=" * 60)
    print("RAY JOBS BENCHMARK")
    print("=" * 60)

    results: dict[str, float] = {}
    results["feature_engineering"] = feature_engineering.run()
    results["simulation"] = simulation.run()
    results["model_scoring"] = model_scoring.run()

    print("\n--- Ray Timing Summary ---")
    for task, elapsed in results.items():
        print(f"  {task:<35} {elapsed:.2f}s")
    print()

    ray.shutdown()
    return results


if __name__ == "__main__":
    main()
