"""Ray model scoring: one stateful actor per region (actor pattern)."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ray

from src.utils.config import cfg

logger = logging.getLogger(__name__)

REGIONS: list[str] = cfg.data_gen.regions
RAW_DIR: Path = Path(cfg.paths.raw)


@ray.remote
class ScoringActor:
    """Stateful actor: per-region pricing model (dummy linear weights)."""

    def __init__(self, region: str) -> None:
        self.region = region
        rng = np.random.default_rng(hash(region) % 2**32)
        self.coef_price = rng.uniform(0.8, 1.2)
        self.coef_qty = rng.uniform(0.5, 1.5)
        self.bias = rng.uniform(-10.0, 10.0)

    def score_batch(
        self, prices: list[float], quantities: list[int]
    ) -> list[float]:
        p = np.array(prices, dtype=np.float64)
        q = np.array(quantities, dtype=np.float64)
        return (self.coef_price * p + self.coef_qty * q + self.bias).tolist()

    def region_name(self) -> str:
        return self.region


def run() -> float:
    ray.init(num_cpus=cfg.ray.num_cpus, ignore_reinit_error=True)

    actors = {r: ScoringActor.remote(r) for r in REGIONS}

    files = sorted(RAW_DIR.glob("events_*.parquet"))
    if not files:
        logger.warning(
            "No event files found — run `make generate-data` first."
        )
        return 0.0

    sample_size: int = cfg.model_scoring.sample_size
    df = pd.read_parquet(files[0]).head(sample_size)

    t0 = time.perf_counter()
    futures = []
    for region in REGIONS:
        subset = df[df["region"] == region]
        if subset.empty:
            continue
        fut = actors[region].score_batch.remote(
            subset["price"].tolist(), subset["quantity"].tolist()
        )
        futures.append((region, fut))

    scores = {region: ray.get(fut) for region, fut in futures}
    elapsed = time.perf_counter() - t0

    for region, s in scores.items():
        logger.info(
            "Region %s: scored %d records, mean=%.2f",
            region,
            len(s),
            float(np.mean(s)),
        )

    logger.info("Model scoring (Ray actor pattern): %.2fs", elapsed)
    return elapsed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
