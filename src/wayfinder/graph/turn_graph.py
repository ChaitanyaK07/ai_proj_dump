"""Turn-aware state space.

A search state is the directed edge just traversed, i.e. the pair (node, incoming edge), so that the
junction load of every turn can be priced. Transition ``e -> e2`` exists whenever ``head[e] == tail[e2]``
and carries a weight-independent turn penalty; its cost under weight vector ``w`` is
``length(e2) * rho(e2, w) + w_junction * 30 m * penalty(e, e2)``.
"""
from __future__ import annotations

import numpy as np

from wayfinder.graph.features import turn_penalty
from wayfinder.graph.road_graph import RoadGraph


class TurnGraph:
    def __init__(self, rg: RoadGraph):
        self.rg = rg
        self.m = rg.m
        frm, to, pen = [], [], []
        for e in range(rg.m):
            v = int(rg.head[e])
            deg = int(rg.degree[v])
            be = float(rg.b_end[e])
            for e2 in rg.out_edges(v):
                e2 = int(e2)
                frm.append(e)
                to.append(e2)
                pen.append(turn_penalty(be, float(rg.b_start[e2]), deg))
        self.t_from = np.array(frm, dtype=np.int64)
        self.t_to = np.array(to, dtype=np.int64)
        self.t_pen = np.array(pen, dtype=float)
        self.n_transitions = len(frm)
        # transitions are generated grouped by source state, so ptr is a running count
        counts = np.bincount(self.t_from, minlength=rg.m)
        self.t_ptr = np.zeros(rg.m + 1, dtype=np.int64)
        np.cumsum(counts, out=self.t_ptr[1:])
        # python lists for the inner loops of the searches
        self.ptr_list = self.t_ptr.tolist()
        self.to_list = self.t_to.tolist()
        self.head_list = rg.head.tolist()
        self.out_ptr_list = rg.out_ptr.tolist()
        self.out_idx_list = rg.out_idx.tolist()

    def successors(self, e: int):
        """(next state, turn penalty) pairs of state ``e``."""
        a, b = self.t_ptr[e], self.t_ptr[e + 1]
        return list(zip(self.t_to[a:b].tolist(), self.t_pen[a:b].tolist()))

    def transition_index(self, e: int, e2: int) -> int:
        a, b = self.t_ptr[e], self.t_ptr[e + 1]
        hits = np.nonzero(self.t_to[a:b] == e2)[0]
        if not len(hits):
            raise KeyError(f"no transition {e} -> {e2}")
        return int(a + hits[0])
