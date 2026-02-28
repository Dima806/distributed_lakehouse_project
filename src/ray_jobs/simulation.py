"""Ray Monte Carlo simulation: parallel GBM price paths (task pattern)."""

from __future__ import annotations

import logging
import time

import numpy as np
import ray

from src.utils.config import cfg

logger = logging.getLogger(__name__)

N_STEPS: int = cfg.simulation.n_steps


@ray.remote
def simulate_price_path(
    product_id: int, base_price: float, seed: int
) -> dict[str, object]:
    """Simulate a price path via Geometric Brownian Motion."""
    rng = np.random.default_rng(seed)
    mu = 0.05 / N_STEPS
    sigma = 0.20 / np.sqrt(N_STEPS)

    returns = rng.normal(mu, sigma, size=N_STEPS)
    prices = base_price * np.exp(np.cumsum(returns))

    return {
        "product_id": product_id,
        "base_price": base_price,
        "final_price": float(prices[-1]),
        "max_price": float(prices.max()),
        "min_price": float(prices.min()),
        "volatility": float(prices.std()),
    }


def run(n_products: int = cfg.simulation.n_products) -> float:
    ray.init(num_cpus=cfg.ray.num_cpus, ignore_reinit_error=True)

    rng = np.random.default_rng(cfg.simulation.seed)
    base_prices = rng.uniform(
        cfg.simulation.price_min,
        cfg.simulation.price_max,
        size=n_products,
    )

    t0 = time.perf_counter()
    futures = [
        simulate_price_path.remote(i, float(base_prices[i]), seed=i)
        for i in range(n_products)
    ]
    results = ray.get(futures)
    elapsed = time.perf_counter() - t0

    avg_final = np.mean([r["final_price"] for r in results])
    logger.info(
        "Monte Carlo (%d products, Ray): %.2fs | avg final: %.2f",
        n_products,
        elapsed,
        avg_final,
    )
    return elapsed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
