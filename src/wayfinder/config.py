"""Repository paths and the pipeline configuration (configs/default.yaml)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Config(dict):
    """The YAML config as a nested dict, plus helpers that resolve paths against the repo root."""

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key])
        return p if p.is_absolute() else ROOT / p


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        with open(path, encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = _merge(cfg, overrides)
    return Config(cfg)
