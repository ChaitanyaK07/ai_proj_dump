"""Weight-vector sampling, the frozen train / held-out split, the λ family and the study profiles."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wayfinder import WEIGHT_KEYS


def sample_weight_vectors(n: int, rng: np.random.Generator, zero_prob: float = 0.3) -> np.ndarray:
    """Each slider uniform in [0, 1], switched off with probability ``zero_prob`` (sparse preferences)."""
    W = rng.uniform(0.0, 1.0, size=(n, len(WEIGHT_KEYS)))
    W[rng.uniform(size=W.shape) < zero_prob] = 0.0
    return np.round(W, 3)


def make_weight_sets(n_train: int, n_heldout: int, seed: int, zero_prob: float = 0.3) -> dict[str, np.ndarray]:
    """40 training and 10 held-out weight vectors, all distinct, from one fixed seed."""
    rng = np.random.default_rng(seed)
    seen: set[tuple] = set()
    rows = []
    while len(rows) < n_train + n_heldout:
        for w in sample_weight_vectors(n_train + n_heldout, rng, zero_prob):
            key = tuple(w.tolist())
            if key not in seen:
                seen.add(key)
                rows.append(w)
            if len(rows) == n_train + n_heldout:
                break
    W = np.array(rows)
    return {"train": W[:n_train], "heldout": W[n_train:]}


def lambda_weights(lam: float) -> np.ndarray:
    """Problem-statement family: share ``lam`` of the weight spread over the four non-time sliders."""
    return np.array([1.0 - lam, lam / 4, lam / 4, lam / 4, lam / 4])


# Three stated preference profiles for the user study (and presets in the web UI).
PROFILES = {
    "shade": {"weights": {"time": 0.2, "canopy": 0.8},
              "scenario": "It is 2 pm in May and 41 °C. You would like as much tree cover along the way as possible."},
    "quiet": {"weights": {"time": 0.2, "traffic": 0.8},
              "scenario": "You are walking with a young child and want to stay away from busy roads."},
    "easy": {"weights": {"time": 0.2, "climb": 0.8, "junction": 0.4},
             "scenario": "You are recovering from a knee injury: you want to avoid uphill stretches, stairs "
                         "and awkward crossings."},
}
FASTEST = {"time": 1.0}


def save_weight_sets(sets: dict[str, np.ndarray], path: str | Path, meta: dict) -> None:
    out = {"keys": list(WEIGHT_KEYS), **meta,
           "train": sets["train"].tolist(), "heldout": sets["heldout"].tolist()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=1), encoding="utf-8")


def load_weight_sets(path: str | Path) -> dict[str, np.ndarray]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if tuple(d["keys"]) != WEIGHT_KEYS:
        raise ValueError(f"weight file {path} uses keys {d['keys']}, expected {WEIGHT_KEYS}")
    return {"train": np.array(d["train"], dtype=float), "heldout": np.array(d["heldout"], dtype=float)}
