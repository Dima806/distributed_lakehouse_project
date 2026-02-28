"""Ray feature engineering: per-region features (task pattern)."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import ray

from src.utils.config import cfg

logger = logging.getLogger(__name__)

REGIONS: list[str] = cfg.data_gen.regions
RAW_DIR: Path = Path(cfg.paths.raw)


@ray.remote
def engineer_region_features(region: str, raw_dir: str) -> dict[str, object]:
    """Compute per-region feature statistics from raw Parquet files."""
    files = list(Path(raw_dir).glob("events_*.parquet"))
    if not files:
        return {
            "region": region,
            "mean_price": 0.0,
            "total_revenue": 0.0,
            "event_count": 0,
        }

    chunks = [
        pd.read_parquet(f, filters=[("region", "=", region)]) for f in files
    ]
    df = pd.concat([c for c in chunks if not c.empty], ignore_index=True)

    if df.empty:
        return {
            "region": region,
            "mean_price": 0.0,
            "total_revenue": 0.0,
            "event_count": 0,
        }

    revenue = df["price"] * df["quantity"]
    return {
        "region": region,
        "event_count": len(df),
        "mean_price": float(df["price"].mean()),
        "price_std": float(df["price"].std()),
        "total_revenue": float(revenue.sum()),
        "avg_quantity": float(df["quantity"].mean()),
    }


def run() -> float:
    ray.init(num_cpus=cfg.ray.num_cpus, ignore_reinit_error=True)

    t0 = time.perf_counter()
    futures = [
        engineer_region_features.remote(r, str(RAW_DIR.resolve()))
        for r in REGIONS
    ]
    results = ray.get(futures)
    elapsed = time.perf_counter() - t0

    print("\nRegion Features:")
    for r in results:
        print(
            f"  {r['region']:<10}"
            f" count={r['event_count']:>10,}"
            f"  revenue={r['total_revenue']:>14,.0f}"
            f"  avg_price={r['mean_price']:>7.2f}"
        )

    logger.info("Feature engineering (Ray): %.2fs", elapsed)
    return elapsed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
