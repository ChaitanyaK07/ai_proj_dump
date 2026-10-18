"""Evaluation harness: run every search condition on every query against exact ground truth.

For each query the exact optimum ``C*`` comes from one backward Dijkstra run (which also gives the
perfect heuristic). Every condition records nodes expanded and generated, re-expansions, latency and
the suboptimality ratio ``cost / C*``. Latency = pricing the graph for ``w`` + preparing the heuristic
+ the search itself, measured with ``time.perf_counter`` in a single process.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from wayfinder.network import Network
from wayfinder.search import (BackwardSolver, NodeHeuristic, SearchResult, SearchSpace, admissible_heuristic,
                              astar, dijkstra, focal_search, perfect_heuristic, weighted_astar)
from wayfinder.search.backward import focal_floor

SUBOPT_TOL = 1e-9


@dataclass
class QueryContext:
    qid: int
    source: int
    target: int
    w: np.ndarray
    weight_set: str
    space: SearchSpace
    space_ms: float
    h_adm: NodeHeuristic
    h_adm_ms: float
    D: np.ndarray          # exact cost-to-go per state
    H: np.ndarray          # exact cost-to-go per node
    optimal: float
    solver: BackwardSolver | None = None
    cache: dict = field(default_factory=dict)

    def floor(self, W: float) -> int:
        """Expansions any Focal Search with bound W needs on this query (see search.backward.focal_floor)."""
        if "g_star" not in self.cache:
            self.cache["g_star"] = self.solver.forward_state_costs(self.space.trans_cost, self.space.edge_cost,
                                                                   self.source)
            self.cache["h_state"] = self.h_adm.values[self.space.tg.rg.head]
            self.cache["goal"] = self.space.tg.rg.head == self.target
        return focal_floor(self.cache["g_star"], self.cache["h_state"], self.optimal, W, self.cache["goal"])


def make_context(net: Network, solver: BackwardSolver, qid: int, source: int, target: int, w,
                 weight_set: str = "") -> QueryContext:
    t0 = time.perf_counter()
    space = SearchSpace.build(net.cm, w)
    space_ms = 1000 * (time.perf_counter() - t0)
    t0 = time.perf_counter()
    h = admissible_heuristic(net.cm, target, space.w)
    h_ms = 1000 * (time.perf_counter() - t0)
    D, H = solver.costs(np.asarray(space.trans_cost), np.asarray(space.edge_cost), target)
    return QueryContext(qid, source, target, space.w, weight_set, space, space_ms, h, h_ms, D, H,
                        float(H[source]), solver)


@dataclass
class Condition:
    name: str
    family: str
    W: float | None
    seed: int | None
    run: Callable[[QueryContext], tuple[SearchResult, float]]   # -> (result, heuristic prep ms)

    @property
    def group(self) -> str:
        return self.name if self.seed is None else self.name.rsplit("_s", 1)[0]


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, 1000 * (time.perf_counter() - t0)


def build_conditions(net: Network, cfg: dict, models: dict | None = None, tuned=None,
                     include_perfect: bool = True) -> list[Condition]:
    from wayfinder.learning.inference import LearnedHeuristic

    ev = cfg["evaluation"]
    mode, key = ev["inference_mode"], ev["focal_key"]
    cm = net.cm
    conds = [
        Condition("dijkstra", "dijkstra", None, None,
                  lambda c: (dijkstra(c.space, c.source, c.target), 0.0)),
        Condition("astar_hadm", "astar_hadm", 1.0, None,
                  lambda c: (astar(c.space, c.source, c.target, c.h_adm, name="astar_hadm"), c.h_adm_ms)),
    ]
    for W in ev["wastar_W"]:
        conds.append(Condition(f"wastar_hadm_W{W:g}", "wastar_hadm", W, None,
                               lambda c, W=W: (weighted_astar(c.space, c.source, c.target, c.h_adm, W),
                                               c.h_adm_ms)))
    if include_perfect:
        conds.append(Condition("astar_hstar", "astar_hstar", 1.0, None,
                               lambda c: (astar(c.space, c.source, c.target, perfect_heuristic(c.D),
                                                name="astar_hstar"), 0.0)))
        for W in ev["focal_W"]:
            conds.append(Condition(f"focal_hstar_W{W:g}", "focal_hstar", W, None,
                                   lambda c, W=W: (focal_search(c.space, c.source, c.target, c.h_adm,
                                                                NodeHeuristic(c.H, cm.tg.head_list), W, key),
                                                   c.h_adm_ms)))
    if tuned is not None:
        for W in ev["focal_W"]:
            def run_tuned(c, W=W):
                h, ms = _timed(lambda: tuned.heuristic(cm, c.target, c.w))
                return focal_search(c.space, c.source, c.target, c.h_adm, h, W, key), c.h_adm_ms + ms
            conds.append(Condition(f"focal_tuned_W{W:g}", "focal_tuned", W, None, run_tuned))
    for seed, model in (models or {}).items():
        def run_learned_astar(c, model=model):
            h, ms = _timed(lambda: LearnedHeuristic(model, cm, c.target, c.w, mode))
            return astar(c.space, c.source, c.target, h, name="astar_learned"), ms
        conds.append(Condition(f"astar_learned_s{seed}", "astar_learned", None, seed, run_learned_astar))
        for W in ev["focal_W"]:
            def run_focal(c, W=W, model=model):
                h, ms = _timed(lambda: LearnedHeuristic(model, cm, c.target, c.w, mode))
                return focal_search(c.space, c.source, c.target, c.h_adm, h, W, key), c.h_adm_ms + ms
            conds.append(Condition(f"focal_learned_W{W:g}_s{seed}", "focal_learned", W, seed, run_focal))
    return conds


def run_condition(cond: Condition, ctx: QueryContext) -> dict:
    res, prep_ms = cond.run(ctx)
    subopt = res.cost / ctx.optimal if ctx.optimal > 0 else 1.0
    bound_ok = None
    if cond.W is not None and cond.family not in ("astar_learned",):
        bound_ok = bool(res.cost <= cond.W * ctx.optimal * (1 + SUBOPT_TOL) + SUBOPT_TOL)
    floor = ctx.floor(cond.W) if cond.family.startswith("focal") and ctx.solver is not None else None
    return {"qid": ctx.qid, "weight_set": ctx.weight_set, "condition": cond.name, "group": cond.group,
            "family": cond.family, "W": cond.W, "seed": cond.seed, "found": res.found, "cost": res.cost,
            "optimal": ctx.optimal, "suboptimality": subopt, "bound_ok": bound_ok,
            "expanded": res.expanded, "generated": res.generated, "reexpanded": res.reexpanded,
            "focal_floor": floor, "search_ms": res.runtime_ms, "prep_ms": prep_ms, "space_ms": ctx.space_ms,
            "latency_ms": ctx.space_ms + prep_ms + res.runtime_ms,
            "length_m": float(ctx.space.tg.rg.length[res.path].sum()) if res.path else 0.0}


def run_benchmark(net: Network, queries: pd.DataFrame, conditions: list[Condition],
                  verbose: bool = True) -> pd.DataFrame:
    from wayfinder.evaluation.benchmark import query_weights

    solver = BackwardSolver(net.tg)
    rows = []
    t0 = time.perf_counter()
    for i, q in enumerate(queries.itertuples(index=False)):
        qd = q._asdict()
        ctx = make_context(net, solver, int(qd["qid"]), int(qd["source"]), int(qd["target"]),
                           query_weights(qd), qd["weight_set"])
        for cond in conditions:
            rows.append(run_condition(cond, ctx))
        if verbose and (i + 1) % 20 == 0:
            el = time.perf_counter() - t0
            left = el / (i + 1) * (len(queries) - i - 1)
            print(f"  {i + 1}/{len(queries)} queries, {el:.0f} s elapsed, ~{left:.0f} s left", flush=True)
    return pd.DataFrame(rows)


def _q(s: pd.Series, p: float) -> float:
    return float(np.percentile(s, p)) if len(s) else float("nan")


def summarise(raw: pd.DataFrame, by: str = "group") -> pd.DataFrame:
    """Median / IQR summary per condition group (learned seeds pooled; per-seed medians listed)."""
    out = []
    for g, d in raw.groupby(by, sort=False):
        row = {by: g, "family": d["family"].iloc[0], "W": d["W"].iloc[0], "runs": len(d),
               "found_rate": float(d["found"].mean())}
        for col in ("expanded", "generated", "latency_ms"):
            row[f"{col}_median"] = _q(d[col], 50)
            row[f"{col}_q25"] = _q(d[col], 25)
            row[f"{col}_q75"] = _q(d[col], 75)
        row["latency_ms_p95"] = _q(d["latency_ms"], 95)
        row["expanded_mean"] = float(d["expanded"].mean())
        fl = d["focal_floor"].dropna()
        row["focal_floor_median"] = _q(fl, 50) if len(fl) else None
        row["reexpanded_mean"] = float(d["reexpanded"].mean())
        row["subopt_mean"] = float(d["suboptimality"].mean())
        row["subopt_p95"] = _q(d["suboptimality"], 95)
        row["subopt_max"] = float(d["suboptimality"].max())
        row["optimal_rate"] = float((d["suboptimality"] <= 1 + 1e-9).mean())
        b = d["bound_ok"].dropna()
        row["bound_violations"] = int((~b.astype(bool)).sum()) if len(b) else None
        seeds = d["seed"].dropna()
        if len(seeds):
            meds = d.groupby("seed")["expanded"].median()
            row["n_seeds"] = int(len(meds))
            row["seed_medians_expanded"] = ";".join(f"{int(s)}:{v:g}" for s, v in meds.items())
        out.append(row)
    return pd.DataFrame(out)
