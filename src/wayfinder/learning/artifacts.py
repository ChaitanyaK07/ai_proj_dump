"""File locations of trained artefacts and helpers to load them."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wayfinder.config import Config

EMBEDDINGS_FILE = "node2vec.npy"
EMBEDDINGS_META = "node2vec.json"
TUNED_FILE = "tuned_classical.json"
HPARAMS_FILE = "best_hparams.json"


def checkpoint_path(cfg: Config, seed: int) -> Path:
    return cfg.path("models_dir") / f"heuristic_seed{seed}.pt"


def embeddings_path(cfg: Config) -> Path:
    return cfg.path("processed_dir") / EMBEDDINGS_FILE


def load_embeddings(cfg: Config, n_nodes: int) -> np.ndarray:
    emb = np.load(embeddings_path(cfg))
    if emb.shape[0] != n_nodes:
        raise ValueError(f"embeddings have {emb.shape[0]} rows but the graph has {n_nodes} nodes")
    return emb


def load_models(cfg: Config, rg, seeds=None) -> dict:
    """Numpy inference models for every trained seed found (or the requested ones)."""
    from wayfinder.learning.inference import HeuristicModel

    seeds = cfg["training"]["seeds"] if seeds is None else seeds
    found = {s: checkpoint_path(cfg, s) for s in seeds if checkpoint_path(cfg, s).exists()}
    if not found:
        return {}
    emb = load_embeddings(cfg, rg.n)
    return {s: HeuristicModel.from_checkpoint(p, emb, rg.x, rg.y) for s, p in found.items()}


def load_tuned(cfg: Config):
    from wayfinder.search.heuristics import TunedClassicalHeuristic

    p = cfg.path("models_dir") / TUNED_FILE
    return TunedClassicalHeuristic.from_dict(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else None
