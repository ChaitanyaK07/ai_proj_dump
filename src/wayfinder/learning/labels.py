"""Label pipeline: exact cost-to-go labels from backward Dijkstra at scale.

One backward run per (weight vector, target) gives the optimal cost ``H[v]`` from *every* node to the
target. We keep ``nodes_per_run`` uniformly sampled nodes per run (all labels are exact, sampling only
bounds the file size). Two label sets are written, each as a folder with one compressed shard per
weight vector (a few MB each, so the dataset can be published on GitHub) and an ``index.json``:

* ``train_weights/`` — 40 training weight vectors × all 200 label targets (8,000 runs);
  the train / val / test membership of every sample is decided by its target node.
* ``heldout_weights/`` — 10 held-out weight vectors × the val + test targets, used only to
  measure generalisation to unseen preferences.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from wayfinder.network import Network
from wayfinder.search.backward import BackwardSolver

TRAIN_DIR = "train_weights"
HELDOUT_DIR = "heldout_weights"
INDEX_FILE = "index.json"
COLUMNS = ("node", "target", "widx", "h_star", "h_adm")


def generate_labels(net: Network, W: np.ndarray, targets: np.ndarray, nodes_per_run: int,
                    seed: int, log_every: int = 500) -> dict[str, np.ndarray]:
    rg, cm = net.rg, net.cm
    solver = BackwardSolver(net.tg)
    rng = np.random.default_rng(seed)
    k = min(nodes_per_run, rg.n - 1)
    n_runs = len(W) * len(targets)
    node = np.empty(n_runs * k, dtype=np.int32)
    target = np.empty(n_runs * k, dtype=np.int32)
    widx = np.empty(n_runs * k, dtype=np.int16)
    h_star = np.empty(n_runs * k, dtype=np.float64)
    h_adm = np.empty(n_runs * k, dtype=np.float64)
    violations = 0
    t0 = time.perf_counter()
    run = 0
    for wi, w in enumerate(W):
        ec = cm.edge_cost(w)
        tc = cm.transition_cost(w, ec)
        G = solver.reversed_graph(tc)
        rho_min = cm.rho_min(w)
        for t in targets:
            t = int(t)
            _, H = solver.costs(tc, ec, t, G)
            if not np.all(np.isfinite(H)):
                raise RuntimeError("unreachable nodes: the graph must be strongly connected")
            ha = rho_min * cm.distance_to(t)
            violations += int(np.sum(ha > H * (1 + 1e-12) + 1e-9))
            pick = rng.choice(rg.n - 1, size=k, replace=False)
            pick[pick >= t] += 1                         # never the target itself
            sl = slice(run * k, (run + 1) * k)
            node[sl], target[sl], widx[sl] = pick, t, wi
            h_star[sl], h_adm[sl] = H[pick], ha[pick]
            run += 1
            if log_every and run % log_every == 0:
                el = time.perf_counter() - t0
                print(f"  {run}/{n_runs} runs, {el:.0f} s elapsed, ~{el / run * (n_runs - run):.0f} s left",
                      flush=True)
    return {"node": node, "target": target, "widx": widx, "h_star": h_star, "h_adm": h_adm,
            "weights": W.astype(np.float64), "admissibility_violations": np.array(violations),
            "seconds": np.array(time.perf_counter() - t0)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_labels(labels: dict, out_dir: Path, meta: dict) -> None:
    """One ``wNN.npz`` shard per weight vector plus ``index.json`` (weights, metadata, checksums)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("w*.npz"):
        old.unlink()
    shards = []
    for wi in range(len(labels["weights"])):
        rows = labels["widx"] == wi
        name = f"w{wi:02d}.npz"
        np.savez_compressed(out_dir / name, **{c: labels[c][rows] for c in COLUMNS})
        shards.append({"file": name, "widx": wi, "samples": int(rows.sum()), "sha256": _sha256(out_dir / name)})
    index = {**meta, "columns": list(COLUMNS), "weights": labels["weights"].tolist(),
             "admissibility_violations": int(labels["admissibility_violations"]),
             "seconds": float(labels["seconds"]), "samples": int(len(labels["node"])), "shards": shards}
    (out_dir / INDEX_FILE).write_text(json.dumps(index, indent=1), encoding="utf-8")


def load_labels(path: Path, verify: bool = True) -> dict:
    """Load a label folder (shards, in weight order) or a single legacy ``.npz`` file."""
    path = Path(path)
    if path.is_file():
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        d["meta"] = json.loads(str(d["meta"]))
        return d
    index = json.loads((path / INDEX_FILE).read_text(encoding="utf-8"))
    parts = {c: [] for c in COLUMNS}
    for sh in index["shards"]:
        f = path / sh["file"]
        if verify and _sha256(f) != sh["sha256"]:
            raise ValueError(f"label shard {f} is corrupted or was changed (checksum mismatch)")
        with np.load(f, allow_pickle=False) as z:
            for c in COLUMNS:
                parts[c].append(z[c])
    d = {c: np.concatenate(v) for c, v in parts.items()}
    d["weights"] = np.array(index["weights"], dtype=np.float64)
    d["admissibility_violations"] = np.array(index["admissibility_violations"])
    d["seconds"] = np.array(index["seconds"])
    d["meta"] = {k: v for k, v in index.items() if k not in ("weights", "shards", "columns")}
    return d


def split_mask(labels: dict, targets: np.ndarray) -> np.ndarray:
    return np.isin(labels["target"], targets)


DATASHEET = """# Label dataset — Routes That Fit You

Exact optimal cost-to-go labels for the Jubilee Hills / Banjara Hills walking graph
(`data/jubilee_3500.graphml`, {n} nodes, {m} directed edges, largest strongly connected component).

* **Generator:** `python -m wayfinder labels` (turn-aware backward Dijkstra, `wayfinder.search.backward`).
* **Cost model:** `wayfinder.cost` — `rho(e,w) = 0.1 + Σ_k w_k f_k(e)`, turn cost `w_junction · 30 m · penalty`.
* **Weights:** `data/splits/weights.json` (40 train, 10 held-out); keys `{keys}`.
* **Targets:** `data/splits/targets.json` (200 nodes; train / val / test split by target).

| folder | weight vectors | targets | runs | samples | shards |
|---|---|---|---|---|---|
| `{train_file}/` | 40 train | all 200 | {train_runs} | {train_samples} | `w00.npz` … `w39.npz` |
| `{heldout_file}/` | 10 held-out | val + test | {heldout_runs} | {heldout_samples} | `w00.npz` … `w09.npz` |

Each shard holds the samples of one weight vector (`wNN` = row NN of that folder's `weights`), with
columns `node`, `target` (node indices into `data/processed/graph.npz`), `widx`, `h_star` (exact optimal
cost from `node` to `target`) and `h_adm` (the admissible heuristic). `index.json` stores the weight
vectors, a SHA-256 per shard (checked on load), and `admissibility_violations` — the number of samples
with `h_adm > h_star` found while generating, which must be 0.

Load in Python:

```python
from wayfinder.learning.labels import load_labels
labels = load_labels("data/labels/{train_file}")   # dict of numpy arrays, 8,000,000 rows
```

Licences: derived from OpenStreetMap (ODbL), ESA WorldCover 2021 (CC BY 4.0), Copernicus GLO-30 DEM.
"""
