"""Focal Search (Pearl & Kim, 1982) with a safe OPEN ordering and a free-to-be-inadmissible guide.

* OPEN is ordered by ``f = g + h_open`` with ``h_open`` admissible (``h_adm``).
* FOCAL holds every OPEN state with ``f <= W * f_min``.
* The state expanded next is the FOCAL state with the smallest focal key, computed from ``h_focal``
  (the learned heuristic): ``g + h_focal`` (``focal_key="f"``) or ``h_focal`` alone (``"h"``).

Bound. Re-opening is on by default, so at any time some state on an optimal path is in OPEN with its
optimal g, hence ``f_min <= C*``. A goal is only expanded from FOCAL, where ``g = f <= W * f_min``
(``h_open`` is 0 at the goal), so the returned route costs at most ``W * C*``. The returned cost is
recomputed along the parent chain, which can only be lower than the goal's g.

Implementation. Three heaps with lazy deletion and per-state version stamps: ``open`` (by f, for
``f_min``), ``wait`` (OPEN states not yet in FOCAL, by f) and ``focal`` (by focal key). Whenever
``f_min`` moves, states with ``f <= W * f_min`` are moved from ``wait`` into ``focal``; a FOCAL entry
found above the current bound is moved back, so the bound is honoured even if ``f_min`` were to drop.
"""
from __future__ import annotations

import heapq
import time

from wayfinder.search.heuristics import Heuristic
from wayfinder.search.space import INF, SearchResult, SearchSpace, reconstruct


def focal_search(space: SearchSpace, source: int, target: int, h_open: Heuristic, h_focal: Heuristic,
                 W: float = 1.5, focal_key: str = "f", reopen: bool = True,
                 name: str | None = None) -> SearchResult:
    if W < 1.0:
        raise ValueError("Focal Search needs W >= 1")
    if focal_key not in ("f", "h"):
        raise ValueError("focal_key must be 'f' (g + h_focal) or 'h' (h_focal)")
    t0 = time.perf_counter()
    name = name or f"focal_{W:g}"
    if source == target:
        return SearchResult(name, True, 0.0, [], 0, 0, runtime_ms=1000 * (time.perf_counter() - t0))

    tg = space.tg
    head, ptr, to = tg.head_list, tg.ptr_list, tg.to_list
    tc, ec = space.trans_cost, space.edge_cost
    use_g = focal_key == "f"

    g: dict[int, float] = {}
    parent: dict[int, int] = {}
    step: dict[int, float] = {}
    version: dict[int, int] = {}
    in_open: set[int] = set()
    closed: set[int] = set()
    ever_expanded: set[int] = set()
    open_heap: list = []      # (f, -g, cnt, state, ver)
    wait_heap: list = []      # (f, cnt, state, ver)
    focal_heap: list = []     # (key, f, cnt, state, ver)
    push, pop = heapq.heappush, heapq.heappop
    cnt = 0
    expanded, generated, reexpanded = 1, 0, 0
    hf_of: dict[int, float] = {}

    def insert(states, gs):
        nonlocal cnt, generated
        hos = h_open.evaluate(states)
        hfs = h_focal.evaluate(states)
        for e, ge, ho, hf in zip(states, gs, hos, hfs):
            ver = version.get(e, 0) + 1
            version[e] = ver
            in_open.add(e)
            hf_of[e] = hf
            f = ge + ho
            push(open_heap, (f, -ge, cnt, e, ver))
            push(wait_heap, (f, cnt, e, ver))
            cnt += 1
        generated += len(states)

    starts, start_g = [], []
    for e in space.start_states(source):
        ge = ec[e]
        if ge < g.get(e, INF):
            g[e], parent[e], step[e] = ge, -1, ge
            starts.append(e)
            start_g.append(ge)
    insert(starts, start_g)

    while True:
        while open_heap and (open_heap[0][4] != version[open_heap[0][3]] or open_heap[0][3] not in in_open):
            pop(open_heap)
        if not open_heap:
            return SearchResult(name, False, INF, [], expanded, generated, reexpanded,
                                1000 * (time.perf_counter() - t0))
        f_min = open_heap[0][0]
        bound = W * f_min

        while wait_heap and wait_heap[0][0] <= bound:
            f, c, e, ver = pop(wait_heap)
            if ver == version[e] and e in in_open:
                key = g[e] + hf_of[e] if use_g else hf_of[e]
                push(focal_heap, (key, f, c, e, ver))

        e = -1
        while focal_heap:
            key, f, c, s, ver = pop(focal_heap)
            if ver != version[s] or s not in in_open:
                continue
            if f > bound:                       # bound tightened: back to waiting
                push(wait_heap, (f, c, s, ver))
                continue
            e = s
            break
        if e == -1:                             # cannot happen: the f_min state is always in FOCAL
            raise RuntimeError("FOCAL empty while OPEN is not")

        ge = g[e]
        if head[e] == target:
            path, cost = reconstruct(parent, step, e)
            return SearchResult(name, True, cost, path, expanded, generated, reexpanded,
                                1000 * (time.perf_counter() - t0),
                                extra={"bound": bound, "f_min": f_min, "goal_g": ge})
        in_open.discard(e)
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
            insert(cand, cand_g)
