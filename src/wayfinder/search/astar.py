"""Dijkstra, A* and weighted A* over the turn-aware state space, with node accounting.

All three are one best-first search ordered by ``f = g + weight * h``:

* Dijkstra       ``h = 0``
* A*             ``weight = 1`` (optimal for an admissible h; no re-expansions for a consistent h)
* weighted A*    ``weight = W > 1`` with a consistent h: cost <= W * optimal without re-expansions

Ties on ``f`` are broken towards larger ``g`` (deeper states), then FIFO. The origin is a virtual state
with no incoming edge and counts as one expansion. A query is solved when a state whose head is the
destination is expanded.
"""
from __future__ import annotations

import heapq
import time

from wayfinder.search.heuristics import Heuristic, ZeroHeuristic
from wayfinder.search.space import INF, SearchResult, SearchSpace, reconstruct


def astar(space: SearchSpace, source: int, target: int, h: Heuristic, weight: float = 1.0,
          reopen: bool = False, name: str | None = None) -> SearchResult:
    t0 = time.perf_counter()
    name = name or ("astar" if weight == 1.0 else f"wastar_{weight:g}")
    if source == target:
        return SearchResult(name, True, 0.0, [], 0, 0, runtime_ms=1000 * (time.perf_counter() - t0))

    tg = space.tg
    head, ptr, to = tg.head_list, tg.ptr_list, tg.to_list
    tc, ec = space.trans_cost, space.edge_cost
    g: dict[int, float] = {}
    parent: dict[int, int] = {}
    step: dict[int, float] = {}
    closed: set[int] = set()
    ever_expanded: set[int] = set()
    heap: list = []
    push, pop = heapq.heappush, heapq.heappop
    counter = 0
    expanded, generated, reexpanded = 1, 0, 0

    starts = [e for e in space.start_states(source)]
    for e, he in zip(starts, h.evaluate(starts)):
        ge = ec[e]
        if ge < g.get(e, INF):
            g[e], parent[e], step[e] = ge, -1, ge
            push(heap, (ge + weight * he, -ge, counter, e))
            counter += 1
            generated += 1

    while heap:
        _, neg_g, _, e = pop(heap)
        ge = -neg_g
        if ge > g[e] or e in closed:
            continue                                    # stale entry
        if head[e] == target:
            path, cost = reconstruct(parent, step, e)
            return SearchResult(name, True, cost, path, expanded, generated, reexpanded,
                                1000 * (time.perf_counter() - t0))
        closed.add(e)
        expanded += 1
        if e in ever_expanded:
            reexpanded += 1
        else:
            ever_expanded.add(e)

        cand, cand_g = [], []
        for i in range(ptr[e], ptr[e + 1]):
            e2 = to[i]
            n2 = ge + tc[i]
            if n2 < g.get(e2, INF):
                if e2 in closed:
                    if not reopen:
                        continue
                    closed.discard(e2)
                g[e2], parent[e2], step[e2] = n2, e, tc[i]
                cand.append(e2)
                cand_g.append(n2)
        if cand:
            for e2, n2, h2 in zip(cand, cand_g, h.evaluate(cand)):
                push(heap, (n2 + weight * h2, -n2, counter, e2))
                counter += 1
            generated += len(cand)

    return SearchResult(name, False, INF, [], expanded, generated, reexpanded, 1000 * (time.perf_counter() - t0))


def dijkstra(space: SearchSpace, source: int, target: int) -> SearchResult:
    return astar(space, source, target, ZeroHeuristic(), 1.0, name="dijkstra")


def weighted_astar(space: SearchSpace, source: int, target: int, h: Heuristic, W: float) -> SearchResult:
    return astar(space, source, target, h, weight=W, name=f"wastar_{W:g}")
