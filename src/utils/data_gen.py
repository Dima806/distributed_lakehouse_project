"""Synthetic e-commerce event dataset generator."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import cfg

logger = logging.getLogger(__name__)

SEED: int = cfg.data_gen.seed
REGIONS: list[str] = cfg.data_gen.regions
EVENT_TYPES: list[str] = cfg.data_gen.event_types
N_USERS: int = cfg.data_gen.n_users
N_PRODUCTS: int = cfg.data_gen.n_products
RAW_DIR: Path = Path(cfg.paths.raw)

_START_TS = pd.Timestamp(cfg.data_gen.events.start_date).timestamp()
_END_TS = pd.Timestamp(cfg.data_gen.events.end_date).timestamp()
_TS_RANGE = _END_TS - _START_TS


def generate_events(
    n_rows: int = cfg.data_gen.n_rows,
    chunk_size: int = cfg.data_gen.chunk_size,
) -> None:
    """Generate events table and write to Parquet in chunks."""
    rng = np.random.default_rng(SEED)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    price_min: float = cfg.data_gen.events.price_min
    price_max: float = cfg.data_gen.events.price_max
    qty_min: int = cfg.data_gen.events.quantity_min
    qty_max: int = cfg.data_gen.events.quantity_max

    for chunk_start in range(0, n_rows, chunk_size):
        chunk_n = min(chunk_size, n_rows - chunk_start)
        df = pd.DataFrame(
            {
                "event_id": np.arange(
                    chunk_start, chunk_start + chunk_n, dtype=np.int64
                ),
                "user_id": rng.integers(0, N_USERS, size=chunk_n).astype(
                    np.int32
                ),
                "event_timestamp": pd.to_datetime(
                    _START_TS + rng.uniform(0, _TS_RANGE, size=chunk_n),
                    unit="s",
                ).astype("datetime64[us]"),
                "event_type": pd.Categorical(
                    rng.choice(EVENT_TYPES, size=chunk_n),
                    categories=EVENT_TYPES,
                ),
                "product_id": rng.integers(0, N_PRODUCTS, size=chunk_n).astype(
                    np.int32
                ),
                "price": np.round(
                    rng.uniform(price_min, price_max, size=chunk_n), 2
                ),
                "quantity": rng.integers(
                    qty_min, qty_max, size=chunk_n
                ).astype(np.int8),
                "region": pd.Categorical(
                    rng.choice(REGIONS, size=chunk_n), categories=REGIONS
                ),
            }
        )
        out_path = RAW_DIR / f"events_{chunk_start:010d}.parquet"
        df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info(
            "Written chunk %d → %s (%d rows)",
            chunk_start // chunk_size,
            out_path,
            chunk_n,
        )


def generate_users() -> None:
    """Generate users dimension table."""
    rng = np.random.default_rng(SEED + 1)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    signup_start = pd.Timestamp(cfg.data_gen.users.signup_start).timestamp()
    signup_range = _START_TS - signup_start
    age_min: int = cfg.data_gen.users.age_min
    age_max: int = cfg.data_gen.users.age_max

    df = pd.DataFrame(
        {
            "user_id": np.arange(N_USERS, dtype=np.int32),
            "age": rng.integers(age_min, age_max, size=N_USERS).astype(
                np.int8
            ),
            "region": pd.Categorical(
                rng.choice(REGIONS, size=N_USERS), categories=REGIONS
            ),
            "signup_date": pd.to_datetime(
                signup_start + rng.uniform(0, signup_range, size=N_USERS),
                unit="s",
            )
            .astype("datetime64[us]")
            .normalize(),
        }
    )
    df.to_parquet(RAW_DIR / "users.parquet", index=False, engine="pyarrow")
    logger.info("Written users.parquet (%d rows)", N_USERS)


def generate_products() -> None:
    """Generate products dimension table."""
    rng = np.random.default_rng(SEED + 2)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    categories: list[str] = cfg.data_gen.product_categories
    price_min: float = cfg.data_gen.products.price_min
    price_max: float = cfg.data_gen.products.price_max
    weight_min: float = cfg.data_gen.products.weight_min
    weight_max: float = cfg.data_gen.products.weight_max

    df = pd.DataFrame(
        {
            "product_id": np.arange(N_PRODUCTS, dtype=np.int32),
            "category": pd.Categorical(
                rng.choice(categories, size=N_PRODUCTS), categories=categories
            ),
            "base_price": np.round(
                rng.uniform(price_min, price_max, size=N_PRODUCTS), 2
            ),
            "weight_kg": np.round(
                rng.uniform(weight_min, weight_max, size=N_PRODUCTS), 2
            ),
        }
    )
    df.to_parquet(RAW_DIR / "products.parquet", index=False, engine="pyarrow")
    logger.info("Written products.parquet (%d rows)", N_PRODUCTS)


def generate_all(n_rows: int = cfg.data_gen.n_rows) -> None:
    """Generate the full synthetic dataset."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Generating %d event rows...", n_rows)
    generate_events(n_rows)
    generate_users()
    generate_products()
    logger.info("Dataset generation complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=cfg.data_gen.n_rows)
    args = parser.parse_args()
    generate_all(args.rows)
