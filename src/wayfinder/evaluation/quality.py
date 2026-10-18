"""Heuristic-quality analysis against exact cost-to-go.

For every benchmark query (target t, weights w) one backward run gives ``h*`` at every node. For each
heuristic we report, over sampled nodes v != t:

* admissibility violations: share of nodes with ``h(v) > h*(v)`` and the mean overestimate among them;
* informedness: ``h(v) / h*(v)`` (1 is perfect, the admissible heuristic is always <= 1);

and over every edge ``e = (u, v)`` with ``u != v``:

* consistency violations: share of edges with ``h(u) > cost(e) + h(v)`` (turn cost 0, the strictest case).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wayfinder.evaluation.benchmark import query_weights
from wayfinder.network import Network
from wayfinder.search.backward import BackwardSolver

TOL = 1e-9


def heuristic_node_values(net: Network, name: str, t: int, w, models: dict, tuned) -> np.ndarray:
    from wayfinder.learning.inference import LearnedHeuristic

    if name == "h_adm":
        return net.cm.h_adm(t, w)
    if name == "h_tuned":
        return tuned.heuristic(net.cm, t, w).values
    if name.startswith("h_learn_s"):
        seed = int(name.rsplit("_s", 1)[1])
        return LearnedHeuristic(models[seed], net.cm, t, w, mode="full").values
    raise KeyError(name)


def quality_analysis(net: Network, queries: pd.DataFrame, models: dict, tuned, nodes_per_query: int,
                     seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per query × heuristic rows, a sample of h/h* ratios for histograms)."""
    rg, cm = net.rg, net.cm
    solver = BackwardSolver(net.tg)
    rng = np.random.default_rng(seed)
    names = ["h_adm"] + (["h_tuned"] if tuned is not None else []) + [f"h_learn_s{s}" for s in models]
    proper = rg.tail != rg.head
    eu, ev = rg.tail[proper], rg.head[proper]
    rows, samples = [], []
    for q in queries.itertuples(index=False):
        qd = q._asdict()
        t, w = int(qd["target"]), query_weights(qd)
        ec = cm.edge_cost(w)
        _, H = solver.costs(cm.transition_cost(w, ec), ec, t)
        others = np.delete(np.arange(rg.n), t)
        pick = rng.choice(others, size=min(nodes_per_query, len(others)), replace=False)
        hs = H[pick]
        for name in names:
            h = heuristic_node_values(net, name, t, w, models, tuned)
            ratio = h[pick] / hs
            over = h[pick] > hs * (1 + TOL) + TOL
            cons = h[eu] > ec[proper] + h[ev] + TOL * (1 + ec[proper])
            rows.append({"qid": qd["qid"], "weight_set": qd["weight_set"], "heuristic": name,
                         "admissibility_violation_rate": float(over.mean()),
                         "mean_overestimate_pct": float(100 * (ratio[over] - 1).mean()) if over.any() else 0.0,
                         "max_overestimate_pct": float(100 * (ratio.max() - 1)) if over.any() else 0.0,
                         "consistency_violation_rate": float(cons.mean()),
                         "informedness_mean": float(ratio.mean()),
                         "informedness_median": float(np.median(ratio)),
                         "informedness_at_source": float(h[int(qd["source"])] / H[int(qd["source"])])})
            sub = rng.choice(len(ratio), size=min(200, len(ratio)), replace=False)
            samples.append(pd.DataFrame({"heuristic": name, "weight_set": qd["weight_set"], "ratio": ratio[sub]}))
    return pd.DataFrame(rows), pd.concat(samples, ignore_index=True)


def summarise_quality(per_query: pd.DataFrame) -> pd.DataFrame:
    agg = {c: "mean" for c in ("admissibility_violation_rate", "mean_overestimate_pct", "consistency_violation_rate",
                               "informedness_mean", "informedness_median", "informedness_at_source")}
    agg["max_overestimate_pct"] = "max"
    pooled = per_query.groupby("heuristic", sort=False).agg(agg).reset_index()
    pooled.insert(1, "weight_set", "all")
    split = per_query.groupby(["heuristic", "weight_set"], sort=False).agg(agg).reset_index()
    return pd.concat([pooled, split], ignore_index=True)
