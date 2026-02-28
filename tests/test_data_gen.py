"""Tests for synthetic dataset generator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.utils import data_gen


def test_generate_events_row_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data_gen, "RAW_DIR", tmp_path)
    data_gen.generate_events(n_rows=1_000, chunk_size=500)

    files = sorted(tmp_path.glob("events_*.parquet"))
    assert len(files) == 2

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    assert len(df) == 1_000


def test_generate_events_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data_gen, "RAW_DIR", tmp_path)
    data_gen.generate_events(n_rows=100, chunk_size=100)

    df = pd.read_parquet(tmp_path / "events_0000000000.parquet")
    expected = {
        "event_id",
        "user_id",
        "event_timestamp",
        "event_type",
        "product_id",
        "price",
        "quantity",
        "region",
    }
    assert expected.issubset(set(df.columns))


def test_generate_events_no_nulls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data_gen, "RAW_DIR", tmp_path)
    data_gen.generate_events(n_rows=100, chunk_size=100)

    df = pd.read_parquet(tmp_path / "events_0000000000.parquet")
    assert df.isnull().sum().sum() == 0


def test_generate_users(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data_gen, "RAW_DIR", tmp_path)
    data_gen.generate_users()

    df = pd.read_parquet(tmp_path / "users.parquet")
    assert len(df) == data_gen.N_USERS
    assert set(df.columns) >= {"user_id", "age", "region", "signup_date"}


def test_generate_products(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data_gen, "RAW_DIR", tmp_path)
    data_gen.generate_products()

    df = pd.read_parquet(tmp_path / "products.parquet")
    assert len(df) == data_gen.N_PRODUCTS
    assert set(df.columns) >= {
        "product_id",
        "category",
        "base_price",
        "weight_kg",
    }


def test_generate_events_reproducible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data_gen, "RAW_DIR", tmp_path)
    data_gen.generate_events(n_rows=50, chunk_size=50)
    df1 = pd.read_parquet(tmp_path / "events_0000000000.parquet")

    # Re-generate into the same path (overwrite) — must produce identical data
    data_gen.generate_events(n_rows=50, chunk_size=50)
    df2 = pd.read_parquet(tmp_path / "events_0000000000.parquet")

    pd.testing.assert_frame_equal(df1, df2)
