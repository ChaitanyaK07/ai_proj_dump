"""Routing facade used by the web service: snap points, price the graph, search, describe the route."""
from __future__ import annotations

import threading
import time
from collections import OrderedDict

from wayfinder.cost import as_weight_array, weight_dict
from wayfinder.learning.weights import FASTEST
from wayfinder.network import Network
from wayfinder.search import SearchSpace, admissible_heuristic, astar, dijkstra, focal_search, weighted_astar

ALGORITHMS = ("astar", "focal", "wastar", "dijkstra")
MAX_SNAP_M = 300.0


class RoutePlanner:
    def __init__(self, net: Network, models: dict | None = None, tuned=None, inference_mode: str = "tile",
                 focal_key: str = "f", cache_size: int = 64):
        self.net = net
        self.models = models or {}
        self.model = self.models[min(self.models)] if self.models else None
        self.tuned = tuned
        self.inference_mode = inference_mode
        self.focal_key = focal_key
        self._spaces: OrderedDict = OrderedDict()
        self._lock = threading.Lock()       # FastAPI serves requests from a thread pool
        self._cache_size = cache_size

    @classmethod
    def from_config(cls, cfg) -> "RoutePlanner":
        from wayfinder.learning.artifacts import load_models, load_tuned
        from wayfinder.network import load_network

        net = load_network(cfg)
        return cls(net, load_models(cfg, net.rg), load_tuned(cfg), cfg["evaluation"]["inference_mode"],
                   cfg["evaluation"]["focal_key"])

    @property
    def guide_name(self) -> str:
        return "learned" if self.model is not None else ("tuned classical" if self.tuned is not None else "h_adm")

    def space(self, w) -> tuple[SearchSpace, float]:
        key = tuple(float(v) for v in as_weight_array(w))      # exact values: never route on rounded weights
        with self._lock:
            if key in self._spaces:
                self._spaces.move_to_end(key)
                return self._spaces[key], 0.0
        t0 = time.perf_counter()
        sp = SearchSpace.build(self.net.cm, key)                # read-only once built; safe to share
        ms = 1000 * (time.perf_counter() - t0)
        with self._lock:
            self._spaces[key] = sp
            self._spaces.move_to_end(key)
            while len(self._spaces) > self._cache_size:
                self._spaces.popitem(last=False)
        return sp, ms

    def snap(self, lat: float, lon: float) -> dict:
        v, d = self.net.rg.nearest_node(lat, lon)
        if d > MAX_SNAP_M:
            raise ValueError(f"point ({lat:.5f}, {lon:.5f}) is {d:.0f} m from the walking network; "
                             "pick a point inside the study area")
        rg = self.net.rg
        return {"node": v, "osmid": str(rg.osmid[v]), "lat": float(rg.lat[v]), "lon": float(rg.lon[v]),
                "snap_m": d}

    def _guide(self, target: int, w):
        if self.model is not None:
            from wayfinder.learning.inference import LearnedHeuristic
            return LearnedHeuristic(self.model, self.net.cm, target, w, self.inference_mode)
        if self.tuned is not None:
            return self.tuned.heuristic(self.net.cm, target, w)
        return admissible_heuristic(self.net.cm, target, w)

    def search(self, s: int, t: int, w, algorithm: str = "astar", W: float = 1.5):
        if algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {ALGORITHMS}")
        if W < 1.0:
            raise ValueError("W must be >= 1")
        sp, space_ms = self.space(w)
        t0 = time.perf_counter()
        if algorithm == "dijkstra":
            res, bound = dijkstra(sp, s, t), 1.0
        else:
            h = admissible_heuristic(self.net.cm, t, sp.w)
            if algorithm == "astar":
                res, bound = astar(sp, s, t, h), 1.0
            elif algorithm == "wastar":
                res, bound = weighted_astar(sp, s, t, h, W), W
            else:
                res, bound = focal_search(sp, s, t, h, self._guide(t, sp.w), W, self.focal_key), W
        latency = space_ms + 1000 * (time.perf_counter() - t0)
        return res, bound, latency

    def route(self, origin: tuple[float, float], destination: tuple[float, float], weights,
              algorithm: str = "astar", W: float = 1.5, compare_fastest: bool = True, verify: bool = False) -> dict:
        w = as_weight_array(weights)
        o, d = self.snap(*origin), self.snap(*destination)
        res, bound, latency = self.search(o["node"], d["node"], w, algorithm, W)
        rg, cm = self.net.rg, self.net.cm
        out = {
            "weights": weight_dict(w), "origin": o, "destination": d,
            "route": {"found": res.found, "cost": res.cost, "latlon": rg.path_latlon(res.path),
                      "metrics": cm.route_metrics(res.path)},
            "search": {"algorithm": algorithm, "guide": self.guide_name if algorithm == "focal" else None,
                       "W": bound, "expanded": res.expanded, "generated": res.generated,
                       "reexpanded": res.reexpanded, "search_ms": res.runtime_ms, "latency_ms": latency},
        }
        if verify:
            sp, _ = self.space(w)
            opt = dijkstra(sp, o["node"], d["node"])
            out["optimal"] = {"cost": opt.cost, "suboptimality": res.cost / opt.cost if opt.cost > 0 else 1.0,
                              "dijkstra_expanded": opt.expanded}
        if compare_fastest:
            sp, _ = self.space(FASTEST)
            fr = dijkstra(sp, o["node"], d["node"])
            out["fastest"] = {"latlon": rg.path_latlon(fr.path), "metrics": cm.route_metrics(fr.path)}
        return out
