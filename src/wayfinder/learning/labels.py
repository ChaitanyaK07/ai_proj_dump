"""Label pipeline: exact cost-to-go labels from backward Dijkstra at scale.

One backward run per (weight vector, target) gives the optimal cost ``H[v]`` from *every* node to the
target. We keep ``nodes_per_run`` uniformly sampled nodes per run (all labels are exact, sampling only
bounds the file size). Two label sets are written:

* ``labels_train_weights.npz`` — 40 training weight vectors × all 200 label targets (8,000 runs);
  the train / val / test membership of every sample is decided by its target node.
* ``labels_heldout_weights.npz`` — 10 held-out weight vectors × the val + test targets, used only to
  measure generalisation to unseen preferences.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from wayfinder.network import Network
from wayfinder.search.backward import BackwardSolver

TRAIN_FILE = "labels_train_weights.npz"
HELDOUT_FILE = "labels_heldout_weights.npz"


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


def save_labels(labels: dict, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **labels, meta=np.array(json.dumps(meta)))


def load_labels(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
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

| file | weight vectors | targets | runs | samples |
|---|---|---|---|---|
| `{train_file}` | 40 train | all 200 | {train_runs} | {train_samples} |
| `{heldout_file}` | 10 held-out | val + test | {heldout_runs} | {heldout_samples} |

Columns (one row per sample): `node`, `target` (node indices into `data/processed/graph.npz`), `widx`
(row of `weights`), `h_star` (exact optimal cost from `node` to `target`), `h_adm` (the admissible
heuristic). `h_star / h_adm >= 1` always holds; the number of violations found while generating is
stored as `admissibility_violations` (must be 0).

Licences: derived from OpenStreetMap (ODbL), ESA WorldCover 2021 (CC BY 4.0), Copernicus GLO-30 DEM.
"""
