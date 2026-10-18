"""The frozen benchmark: 200 queries with fixed seeds, generated once and never regenerated.

Destinations never coincide with a label target (no target leaks from training into evaluation). Half
of the queries use one of the 40 training weight vectors, half one of the 10 held-out vectors.
A SHA-256 of the CSV is stored next to it and checked on every load.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from wayfinder import WEIGHT_KEYS
from wayfinder.config import Config

QUERIES_FILE = "queries.csv"
HASH_FILE = "queries.sha256"
CRLF, LF = b"\r\n", b"\n"


def _sha(path: Path) -> str:
    """Hash with line endings normalised, so a git checkout on another OS does not break the check."""
    return hashlib.sha256(path.read_bytes().replace(CRLF, LF)).hexdigest()


def make_benchmark(cfg: Config, rg, splits: dict, force: bool = False) -> pd.DataFrame:
    bc = cfg["benchmark"]
    out = cfg.path("benchmark_dir")
    qpath = out / QUERIES_FILE
    if qpath.exists() and not force:
        raise FileExistsError(f"{qpath} is frozen; it must never be regenerated (use --force only "
                              "if you really mean to start the evaluation over)")
    rng = np.random.default_rng(bc["seed"])
    excluded = set(np.concatenate(list(splits["targets"].values())).tolist())
    n = int(bc["n_queries"])
    n_held = int(round(n * bc["heldout_share"]))
    rows, seen = [], set()
    while len(rows) < n:
        s, t = (int(i) for i in rng.choice(rg.n, size=2, replace=False))
        if t in excluded or (s, t) in seen:
            continue
        d = float(np.hypot(rg.x[s] - rg.x[t], rg.y[s] - rg.y[t]))
        if d < bc["min_euclid_m"]:
            continue
        seen.add((s, t))
        rows.append((s, t, d))
    sets = ["train"] * (n - n_held) + ["heldout"] * n_held
    rng.shuffle(sets)
    recs = []
    for qid, ((s, t, d), ws) in enumerate(zip(rows, sets)):
        W = splits["weights"][ws]
        wi = int(rng.integers(len(W)))
        recs.append({"qid": qid, "source": s, "target": t, "source_osmid": str(rg.osmid[s]),
                     "target_osmid": str(rg.osmid[t]), "euclid_m": round(d, 2), "weight_set": ws,
                     "weight_idx": wi, **{f"w_{k}": float(v) for k, v in zip(WEIGHT_KEYS, W[wi])}})
    df = pd.DataFrame(recs)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(qpath, index=False, lineterminator="\n")
    (out / HASH_FILE).write_text(_sha(qpath) + "\n", encoding="utf-8")
    return df


def load_benchmark(cfg: Config, rg=None) -> pd.DataFrame:
    out = cfg.path("benchmark_dir")
    qpath = out / QUERIES_FILE
    expected = (out / HASH_FILE).read_text(encoding="utf-8").strip()
    if _sha(qpath) != expected:
        raise ValueError(f"{qpath} has changed since it was frozen (hash mismatch)")
    df = pd.read_csv(qpath, dtype={"source_osmid": str, "target_osmid": str})
    if rg is not None:
        bad = [(s, o) for s, o in zip(df["source"], df["source_osmid"]) if str(rg.osmid[s]) != o]
        bad += [(t, o) for t, o in zip(df["target"], df["target_osmid"]) if str(rg.osmid[t]) != o]
        if bad:
            raise ValueError("benchmark node indices do not match the loaded graph")
    return df


def query_weights(row) -> np.ndarray:
    return np.array([row[f"w_{k}"] for k in WEIGHT_KEYS], dtype=float)
