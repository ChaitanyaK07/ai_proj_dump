"""The experiments of the evaluation plan, each writing the result file its claim traces to."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from wayfinder.evaluation.harness import make_context, run_condition, summarise
from wayfinder.learning.weights import lambda_weights
from wayfinder.network import Network
from wayfinder.search import BackwardSolver


def problem_curve(net: Network, queries: pd.DataFrame, lambdas, conditions, verbose: bool = True) -> pd.DataFrame:
    """Node expansions as weight moves from walking time to the non-distance sliders.

    Uses the 200 frozen OD pairs with the λ family ``w = (1-λ, λ/4, λ/4, λ/4, λ/4)``.
    """
    solver = BackwardSolver(net.tg)
    rows = []
    t0 = time.perf_counter()
    for lam in lambdas:
        w = lambda_weights(lam)
        rho_min = net.cm.rho_min(w)
        for q in queries.itertuples(index=False):
            ctx = make_context(net, solver, int(q.qid), int(q.source), int(q.target), w, "lambda")
            inform = float(ctx.h_adm.values[ctx.source] / ctx.optimal)
            for cond in conditions:
                r = run_condition(cond, ctx)
                r.update({"lambda": lam, "rho_min": rho_min, "informedness_hadm_source": inform})
                rows.append(r)
        if verbose:
            print(f"  lambda {lam}: done ({time.perf_counter() - t0:.0f} s)", flush=True)
    return pd.DataFrame(rows)


def summarise_problem_curve(raw: pd.DataFrame) -> pd.DataFrame:
    piv_mean = raw.pivot_table(index="lambda", columns="group", values="expanded", aggfunc="mean")
    piv_med = raw.pivot_table(index="lambda", columns="group", values="expanded", aggfunc="median")
    out = pd.DataFrame(index=piv_mean.index)
    out["rho_min"] = raw.groupby("lambda")["rho_min"].first()
    for g in piv_mean.columns:
        out[f"{g}_mean"] = piv_mean[g]
        out[f"{g}_median"] = piv_med[g]
    if "dijkstra" in piv_mean.columns:
        for g in piv_mean.columns:
            if g != "dijkstra":
                out[f"{g}/dijkstra"] = piv_mean[g] / piv_mean["dijkstra"]
    out["informedness_hadm_source"] = raw.groupby("lambda")["informedness_hadm_source"].mean()
    return out.reset_index()


def bound_sweep(raw: pd.DataFrame) -> pd.DataFrame:
    """Guaranteed factor W vs the suboptimality actually achieved, for every bounded condition."""
    d = raw[raw["family"].isin(["focal_learned", "focal_tuned", "focal_hstar", "wastar_hadm", "astar_hadm"])]
    rows = []
    seed_key = pd.to_numeric(d["seed"], errors="coerce").fillna(-1).astype(int)
    for (fam, W, seed), g in d.groupby(["family", "W", seed_key], sort=True):
        rows.append({"family": fam, "W": W, "seed": None if seed == -1 else int(seed), "queries": len(g),
                     "achieved_mean": float(g["suboptimality"].mean()),
                     "achieved_p95": float(np.percentile(g["suboptimality"], 95)),
                     "achieved_max": float(g["suboptimality"].max()),
                     "optimal_rate": float((g["suboptimality"] <= 1 + 1e-9).mean()),
                     "bound_violations": int((~g["bound_ok"].astype(bool)).sum()),
                     "expanded_median": float(g["expanded"].median()),
                     "expanded_mean": float(g["expanded"].mean()),
                     "focal_floor_median": float(g["focal_floor"].median()) if g["focal_floor"].notna().any()
                     else None})
    return pd.DataFrame(rows)


def generalisation_search(raw: pd.DataFrame) -> pd.DataFrame:
    """Benchmark summary split by whether the query's weight vector was seen in training."""
    parts = []
    for ws, d in raw.groupby("weight_set"):
        s = summarise(d)
        s.insert(1, "weight_set", ws)
        parts.append(s)
    return pd.concat(parts, ignore_index=True)


def go_no_go(summary: pd.DataFrame, headline_W: float) -> dict:
    """Week-4 gate: does learned guidance expand fewer nodes than the classical baseline?

    GO requires the learned guide to beat both A*(h_adm) and the tuned classical guide at the same
    bound W (the proposal's fallback), so the learned network has to earn its place. Focal Search with
    the perfect guide h* is reported alongside: it shows how much any guide could gain.
    """
    def med(group):
        hit = summary[summary["group"] == group]
        return float(hit["expanded_median"].iloc[0]) if len(hit) else None

    def beats(a, b):
        return None if a is None or b is None else bool(a < b)

    W = f"{headline_W:g}"
    learned, tuned = med(f"focal_learned_W{W}"), med(f"focal_tuned_W{W}")
    baseline, wastar = med("astar_hadm"), med(f"wastar_hadm_W{W}")
    perfect = med(f"focal_hstar_W{W}")
    go = bool(beats(learned, baseline)) and beats(learned, tuned) is not False
    verdict = {
        "headline_W": headline_W,
        "median_expanded": {"dijkstra": med("dijkstra"), "astar_hadm": baseline, f"wastar_hadm_W{W}": wastar,
                            f"focal_tuned_W{W}": tuned, f"focal_learned_W{W}": learned,
                            f"focal_hstar_W{W} (perfect guide)": perfect,
                            "astar_learned (unbounded)": med("astar_learned")},
        "learned_beats_astar_hadm": beats(learned, baseline),
        "learned_beats_tuned_classical_same_W": beats(learned, tuned),
        "learned_beats_wastar_same_W": beats(learned, wastar),
        "verdict": "GO" if go else "NO-GO",
        "headline": (f"Focal Search guided by the learned heuristic (W={W})" if go else
                     f"Fallback: Focal Search guided by the tuned classical heuristic (W={W}); the learned "
                     "guide is far more informed (heuristic_quality.csv), but Focal Search cannot exploit it: "
                     "even the perfect guide h* barely helps, because FOCAL is gated by the weak admissible "
                     "OPEN heuristic (see focal_hstar and focal_floor)"),
    }
    if learned is not None and baseline:
        verdict["expansion_reduction_vs_astar_hadm_pct"] = 100 * (1 - learned / baseline)
    if perfect is not None and baseline:
        verdict["perfect_guide_reduction_vs_astar_hadm_pct"] = 100 * (1 - perfect / baseline)
    return verdict


def write_json(obj, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
