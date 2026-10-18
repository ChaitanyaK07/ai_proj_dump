"""The weighted search space and the result record shared by every search algorithm."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wayfinder.cost import CostModel, as_weight_array
from wayfinder.graph.turn_graph import TurnGraph

INF = float("inf")


@dataclass
class SearchSpace:
    """Turn-aware state graph with every edge and transition priced under one weight vector."""
    tg: TurnGraph
    w: np.ndarray
    edge_cost: list[float]        # cost of the first edge out of the origin (no turn)
    trans_cost: list[float]       # aligned with tg.t_to

    @classmethod
    def build(cls, cm: CostModel, w) -> "SearchSpace":
        wa = as_weight_array(w)
        ec = cm.edge_cost(wa)
        tc = cm.transition_cost(wa, ec)
        return cls(tg=cm.tg, w=wa, edge_cost=ec.tolist(), trans_cost=tc.tolist())

    def start_states(self, source: int) -> list[int]:
        a, b = self.tg.out_ptr_list[source], self.tg.out_ptr_list[source + 1]
        return self.tg.out_idx_list[a:b]


@dataclass
class SearchResult:
    algorithm: str
    found: bool
    cost: float
    path: list[int]               # edge ids from origin to destination
    expanded: int                 # states taken off OPEN and expanded (the origin counts as one)
    generated: int                # states inserted into OPEN (new or with an improved g)
    reexpanded: int = 0           # expansions of a state that had been expanded before
    runtime_ms: float = 0.0
    extra: dict = field(default_factory=dict)

    def nodes(self, tg: TurnGraph, source: int) -> list[int]:
        return [source] + [tg.head_list[e] for e in self.path]


def reconstruct(parent: dict, step_cost: dict, goal: int) -> tuple[list[int], float]:
    """Follow parent pointers back to the origin; returns the path and its exact cost."""
    path = []
    e = goal
    while e != -1:
        path.append(e)
        e = parent[e]
    path.reverse()
    cost = 0.0
    for e in path:
        cost += step_cost[e]
    return path, cost
