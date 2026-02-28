"""Config loader: reads config/config.yaml → SimpleNamespace tree."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

_CONFIG_PATH = Path(__file__).parents[2] / "config" / "config.yaml"


def _to_ns(value: object) -> object:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_ns(item) for item in value]
    return value


def load(path: Path = _CONFIG_PATH) -> SimpleNamespace:
    with open(path) as f:
        return _to_ns(yaml.safe_load(f))


# Module-level singleton: from src.utils.config import cfg
cfg: SimpleNamespace = load()
