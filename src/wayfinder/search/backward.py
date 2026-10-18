"""Backward Dijkstra: the exact optimal cost-to-go from every state and every node to one target.

One run per (weight vector, target) gives

* ``D[e]`` — optimal cost from state ``e`` (having just walked edge ``e``) to the target, which is the
  perfect heuristic ``h*`` for the turn-aware search; ``D[e] = 0`` when ``head[e]`` is the target;
* ``H[v] = min_{e2 in out(v)} (cost(e2) + D[e2])`` with ``H[t] = 0`` — the optimal cost of a route that
  *starts* at ``v``. These are the training labels of the learned heuristic. Since turn penalties are
  non-negative, ``H[head[e]] <= D[e]``, so a node-level heuristic trained on ``H`` targets a lower
  bound of the state cost-to-go.

``backward_state_costs`` is the readable reference implementation; ``BackwardSolver`` runs the same
search through ``scipy.sparse.csgraph.dijkstra`` (C speed) for the 8,000-run label sweep. The test
suite checks the two agree.
"""
from __future__ import annotations

import heapq

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as cs_dijkstra

from wayfinder.graph.turn_graph import TurnGraph

INF = float("inf")


def goal_states(tg: TurnGraph, target: int) -> np.ndarray:
    return np.asarray(tg.rg.in_edges(target), dtype=np.int64)


def backward_state_costs(tg: TurnGraph, trans_cost, target: int) -> np.ndarray:
    """Reference pure-Python backward Dijkstra over the reversed transition graph."""
    rev_ptr, rev_idx = _reverse_csr(tg)
    t_from = tg.t_from.tolist()
    tc = list(trans_cost)
    rp, ri = rev_ptr.tolist(), rev_idx.tolist()
    D = [INF] * tg.m
    heap = []
    for e in goal_states(tg, target).tolist():
        D[e] = 0.0
        heap.append((0.0, e))
    heapq.heapify(heap)
    while heap:
        d, e2 = heapq.heappop(heap)
        if d > D[e2]:
            continue
        for j in range(rp[e2], rp[e2 + 1]):
            i = ri[j]
            e = t_from[i]
            nd = d + tc[i]
            if nd < D[e]:
                D[e] = nd
                heapq.heappush(heap, (nd, e))
    return np.array(D)


def node_costs(tg: TurnGraph, D: np.ndarray, edge_cost: np.ndarray, target: int) -> np.ndarray:
    """H[v]: optimal cost from v as an origin (first edge carries no turn cost)."""
    rg = tg.rg
    vals = edge_cost + D                                     # per out-edge e2 of its tail
    H = np.full(rg.n, np.inf)
    np.minimum.at(H, rg.tail, vals)
    H[target] = 0.0
    return H


def _reverse_csr(tg: TurnGraph) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(tg.t_to, kind="stable").astype(np.int64)
    counts = np.bincount(tg.t_to, minlength=tg.m)
    ptr = np.zeros(tg.m + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    return ptr, order


class BackwardSolver:
    """Fast backward Dijkstra over the reversed state graph using scipy's C implementation."""

    def __init__(self, tg: TurnGraph):
        self.tg = tg
        m = tg.m
        # reversed graph: an arc e2 -> e for every transition e -> e2
        marker = csr_matrix((np.arange(1, tg.n_transitions + 1, dtype=float), (tg.t_to, tg.t_from)),
                            shape=(m, m))
        marker.sort_indices()
        self._indptr = marker.indptr.copy()
        self._indices = marker.indices.copy()
        self._perm = (marker.data - 1).astype(np.int64)      # stored position -> transition index
        if len(self._perm) != tg.n_transitions:
            raise RuntimeError("duplicate transitions in the turn graph")

    def reversed_graph(self, trans_cost: np.ndarray) -> csr_matrix:
        data = np.asarray(trans_cost, dtype=float)[self._perm]
        if np.any(data <= 0):
            raise ValueError("transition costs must be strictly positive")
        return csr_matrix((data, self._indices, self._indptr), shape=(self.tg.m, self.tg.m))

    def state_costs(self, trans_cost, target: int, graph: csr_matrix | None = None) -> np.ndarray:
        G = graph if graph is not None else self.reversed_graph(trans_cost)
        goals = goal_states(self.tg, target)
        return cs_dijkstra(G, directed=True, indices=goals, min_only=True)

    def costs(self, trans_cost, edge_cost, target: int, graph: csr_matrix | None = None):
        """(D over states, H over nodes) for one target."""
        D = self.state_costs(trans_cost, target, graph)
        return D, node_costs(self.tg, D, np.asarray(edge_cost, dtype=float), target)

    def forward_state_costs(self, trans_cost, edge_cost, source: int) -> np.ndarray:
        """g*(e): optimal cost from the origin to every state (forward search, virtual origin state)."""
        tg = self.tg
        starts = np.asarray(tg.rg.out_edges(source), dtype=np.int64)
        m = tg.m
        rows = np.concatenate([tg.t_from, np.full(len(starts), m)])
        cols = np.concatenate([tg.t_to, starts])
        data = np.concatenate([np.asarray(trans_cost, dtype=float), np.asarray(edge_cost, dtype=float)[starts]])
        G = csr_matrix((data, (rows, cols)), shape=(m + 1, m + 1))
        return cs_dijkstra(G, directed=True, indices=m)[:m]


def focal_floor(g_star: np.ndarray, h_state: np.ndarray, optimal: float, W: float,
                goal_mask: np.ndarray | None = None) -> int:
    """States every Focal Search with bound W must expand, whatever its guide (origin included).

    Focal stops only once ``f_min >= C*/W``. With an admissible, consistent OPEN heuristic and re-opening,
    a state with ``g* + h < C*/W`` keeps a state of ``f < C*/W`` on OPEN until it is expanded, so all
    such states are expanded before termination. Goal states are never expanded (the search stops on
    them) and a relative tolerance keeps rounding ties between forward and backward sums out.
    """
    below = g_star + h_state < (optimal / W) * (1 - 1e-9)
    if goal_mask is not None:
        below &= ~goal_mask
    return int(np.sum(below)) + 1
