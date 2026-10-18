"""Frozen experimental splits: weight vectors and label target nodes (train / val / test by target)."""
from __future__ import annotations

import json

import numpy as np

from wayfinder.config import Config
from wayfinder.learning.weights import load_weight_sets, make_weight_sets, save_weight_sets

WEIGHTS_FILE = "weights.json"
TARGETS_FILE = "targets.json"


def make_splits(cfg: Config, n_nodes: int, osmid: np.ndarray, force: bool = False) -> dict:
    sc = cfg["splits"]
    out_dir = cfg.path("splits_dir")
    wpath, tpath = out_dir / WEIGHTS_FILE, out_dir / TARGETS_FILE
    if (wpath.exists() or tpath.exists()) and not force:
        raise FileExistsError(f"splits already frozen in {out_dir}; pass --force to regenerate")

    sets = make_weight_sets(sc["n_train_weights"], sc["n_heldout_weights"], sc["seed"], sc["zero_prob"])
    save_weight_sets(sets, wpath, {"seed": sc["seed"], "zero_prob": sc["zero_prob"]})

    rng = np.random.default_rng(sc["seed"] + 1)
    targets = rng.choice(n_nodes, size=sc["n_label_targets"], replace=False)
    fr = sc["target_split"]
    n_tr = int(round(fr[0] * len(targets)))
    n_va = int(round(fr[1] * len(targets)))
    split = {"train": targets[:n_tr], "val": targets[n_tr:n_tr + n_va], "test": targets[n_tr + n_va:]}
    payload = {"seed": sc["seed"] + 1,
               **{k: [int(i) for i in v] for k, v in split.items()},
               **{f"{k}_osmid": [str(osmid[i]) for i in v] for k, v in split.items()}}
    tpath.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return {"weights": sets, "targets": split}


def load_splits(cfg: Config) -> dict:
    out_dir = cfg.path("splits_dir")
    sets = load_weight_sets(out_dir / WEIGHTS_FILE)
    d = json.loads((out_dir / TARGETS_FILE).read_text(encoding="utf-8"))
    targets = {k: np.array(d[k], dtype=np.int64) for k in ("train", "val", "test")}
    return {"weights": sets, "targets": targets, "targets_osmid": {k: d[f"{k}_osmid"] for k in targets}}


def check_splits_match_graph(splits: dict, osmid: np.ndarray) -> None:
    """Node indices are only meaningful for the graph they were drawn from; verify by OSM id."""
    for k, idx in splits["targets"].items():
        if len(idx) and (idx.max() >= len(osmid) or
                         [str(osmid[i]) for i in idx] != list(splits["targets_osmid"][k])):
            raise ValueError("data/splits does not match the loaded graph; regenerate the splits")
