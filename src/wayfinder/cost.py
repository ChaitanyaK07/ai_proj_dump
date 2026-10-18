"""The cost model and the admissible heuristic (frozen interfaces).

    rho(e, w)          = BASE + sum_k w_k * f_k(e)                 k = time, traffic, canopy, climb
    c(e_in, e_out, w)  = length(e_out) * rho(e_out, w) + w_junction * TURN_M * turn_penalty(e_in, e_out)
    h_adm(v, t, w)     = rho_min(w) * s * haversine(v, t)          rho_min(w) = min_e rho(e, w)

Why ``h_adm`` is admissible *and* consistent. ``s`` is the smallest ratio ``length(e) / haversine(u, v)``
over all edges (times 1 - 1e-9 for rounding), so ``rho_min * s * haversine(u, v) <= rho(e) * length(e)``
for every edge. Haversine is a metric, so for any edge ``e = (u, v)`` and any incoming state:

    h_adm(u) <= rho_min * s * (haversine(u, v) + haversine(v, t))
             <= length(e) * rho(e, w) + h_adm(v)
             <= c(e_in, e, w) + h_adm(v)            (turn penalties are >= 0)

and ``h_adm(t) = 0``. Consistency implies admissibility. ``s`` is 1 whenever OSMnx edge lengths are
great-circle sums, which they are; the factor only guards against projection round-off.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from wayfinder import PER_METRE_KEYS, WEIGHT_KEYS, geo
from wayfinder.graph.features import edge_speed, per_metre_features
from wayfinder.graph.road_graph import RoadGraph
from wayfinder.graph.turn_graph import TurnGraph

BASE = 0.1          # cost per metre floor: keeps rho_min > 0 for every weight vector
TURN_M = 30.0       # a penalty-1 turn costs as much as 30 m of walking at w_junction = 1

_PM_INDEX = [WEIGHT_KEYS.index(k) for k in PER_METRE_KEYS]   # positions of per-metre weights in w
_J_INDEX = WEIGHT_KEYS.index("junction")


def as_weight_array(w: Mapping[str, float] | Sequence[float] | np.ndarray) -> np.ndarray:
    """Weight vector as a float array in WEIGHT_KEYS order. Missing dict keys are 0."""
    if isinstance(w, Mapping):
        unknown = set(w) - set(WEIGHT_KEYS)
        if unknown:
            raise ValueError(f"unknown weight keys {sorted(unknown)}; expected {WEIGHT_KEYS}")
        arr = np.array([float(w.get(k, 0.0)) for k in WEIGHT_KEYS])
    else:
        arr = np.asarray(w, dtype=float).reshape(-1)
        if arr.shape != (len(WEIGHT_KEYS),):
            raise ValueError(f"weight vector must have {len(WEIGHT_KEYS)} entries {WEIGHT_KEYS}")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("weights must be finite and non-negative")
    return arr


def weight_dict(w) -> dict[str, float]:
    arr = as_weight_array(w)
    return {k: float(v) for k, v in zip(WEIGHT_KEYS, arr)}


class CostModel:
    def __init__(self, rg: RoadGraph, tg: TurnGraph | None = None, F: np.ndarray | None = None):
        self.rg = rg
        self.tg = tg if tg is not None else TurnGraph(rg)
        self.F = per_metre_features(rg) if F is None else F
        self.length = rg.length
        self.speed = np.array([edge_speed(h) for h in rg.highway])
        hav = geo.haversine_m(rg.lat[rg.tail], rg.lon[rg.tail], rg.lat[rg.head], rg.lon[rg.head])
        ok = hav > 0
        ratio = float(np.min(rg.length[ok] / hav[ok])) if ok.any() else 1.0
        self.hav_scale = min(1.0, ratio) * (1.0 - 1e-9)

    # ------------------------------------------------------------------ per weight vector
    def rho(self, w) -> np.ndarray:
        wa = as_weight_array(w)
        return BASE + self.F @ wa[_PM_INDEX]

    def rho_min(self, w) -> float:
        return float(self.rho(w).min())

    def edge_cost(self, w) -> np.ndarray:
        return self.length * self.rho(w)

    def turn_scale(self, w) -> float:
        return float(as_weight_array(w)[_J_INDEX]) * TURN_M

    def transition_cost(self, w, edge_cost: np.ndarray | None = None) -> np.ndarray:
        """Cost of every transition ``t_from[i] -> t_to[i]`` (aligned with TurnGraph arrays)."""
        ec = self.edge_cost(w) if edge_cost is None else edge_cost
        return ec[self.tg.t_to] + self.turn_scale(w) * self.tg.t_pen

    # ------------------------------------------------------------------ heuristic
    def distance_to(self, t: int) -> np.ndarray:
        """Scaled haversine distance from every node to node ``t`` (metres)."""
        rg = self.rg
        return self.hav_scale * geo.haversine_m(rg.lat, rg.lon, rg.lat[t], rg.lon[t])

    def h_adm(self, t: int, w) -> np.ndarray:
        """Admissible, consistent heuristic for every node towards target ``t``."""
        return self.rho_min(w) * self.distance_to(t)

    # ------------------------------------------------------------------ paths
    def path_cost(self, path: Sequence[int], w) -> float:
        """Cost of a route given as a list of edge ids (first edge leaves the origin, no turn cost)."""
        if not len(path):
            return 0.0
        ec = self.edge_cost(w)
        k = self.turn_scale(w)
        total = float(ec[path[0]])
        for a, b in zip(path[:-1], path[1:]):
            total += float(ec[b]) + k * float(self.tg.t_pen[self.tg.transition_index(int(a), int(b))])
        return total

    def route_metrics(self, path: Sequence[int]) -> dict[str, float]:
        """Interpretable exposures of a route (length-weighted means of the raw features)."""
        if not len(path):
            return {"length_m": 0.0, "walk_time_s": 0.0, "traffic": 0.0, "canopy_share": 0.0,
                    "climb_m": 0.0, "stairs_m": 0.0, "turn_load": 0.0, "turn_load_per_km": 0.0, "edges": 0}
        p = np.asarray(path, dtype=np.int64)
        L = self.length[p]
        total = float(L.sum())
        turn_load = float(sum(self.tg.t_pen[self.tg.transition_index(int(a), int(b))]
                              for a, b in zip(p[:-1], p[1:])))
        rise = np.clip(self.rg.grade[p], 0, None) * L
        return {
            "length_m": total,
            "walk_time_s": float((L / self.speed[p]).sum()),
            "traffic": float((L * self.F[p, 1]).sum() / total),
            "canopy_share": float((L * self.rg.canopy[p]).sum() / total),
            "climb_m": float(rise[~self.rg.steps[p]].sum()),
            "stairs_m": float(L[self.rg.steps[p]].sum()),
            "turn_load": turn_load,
            "turn_load_per_km": turn_load / (total / 1000.0) if total > 0 else 0.0,
            "edges": int(len(p)),
        }
