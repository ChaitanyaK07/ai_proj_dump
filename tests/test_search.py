"""Regression tests of every search algorithm against exact optimal routes."""
import itertools

import networkx as nx
import numpy as np
import pytest

from conftest import WEIGHTS
from wayfinder.cost import BASE, as_weight_array
from wayfinder.search import (BackwardSolver, NodeHeuristic, SearchSpace, StateHeuristic, admissible_heuristic,
                              astar, backward_state_costs, dijkstra, focal_search, node_costs, perfect_heuristic,
                              weighted_astar)


def queries(rg, k=25, seed=0):
    rng = np.random.default_rng(seed)
    return [tuple(int(v) for v in rng.choice(rg.n, size=2, replace=False)) for _ in range(k)]


def reference_cost(net, sp, s, t):
    """Shortest path on an explicit networkx state graph with a virtual source and sink."""
    tg = net.tg
    G = nx.DiGraph()
    for i, (a, b) in enumerate(zip(tg.t_from.tolist(), tg.t_to.tolist())):
        G.add_edge(a, b, weight=sp.trans_cost[i])
    for e in sp.start_states(s):
        G.add_edge("S", e, weight=sp.edge_cost[e])
    for e in net.rg.in_edges(t):
        G.add_edge(int(e), "T", weight=0.0)
    return nx.dijkstra_path_length(G, "S", "T")


def test_weights_validation():
    assert as_weight_array({"time": 1}).tolist() == [1, 0, 0, 0, 0]
    with pytest.raises(ValueError):
        as_weight_array({"speed": 1})
    with pytest.raises(ValueError):
        as_weight_array([1, 2])
    with pytest.raises(ValueError):
        as_weight_array([-1, 0, 0, 0, 0])


@pytest.mark.parametrize("w", WEIGHTS)
def test_dijkstra_matches_networkx_reference(net, w):
    sp = SearchSpace.build(net.cm, w)
    for s, t in queries(net.rg, 10):
        r = dijkstra(sp, s, t)
        assert r.found
        assert r.cost == pytest.approx(reference_cost(net, sp, s, t), rel=1e-12)
        assert r.cost == pytest.approx(net.cm.path_cost(r.path, w), rel=1e-12)
        assert net.rg.tail[r.path[0]] == s and net.rg.head[r.path[-1]] == t


@pytest.mark.parametrize("w", WEIGHTS)
def test_backward_reference_equals_scipy_and_forward(net, w):
    cm = net.cm
    ec = cm.edge_cost(w)
    tc = cm.transition_cost(w, ec)
    sp = SearchSpace.build(cm, w)
    solver = BackwardSolver(net.tg)
    for t in range(0, net.rg.n, 13):
        D_ref = backward_state_costs(net.tg, tc, t)
        D, H = solver.costs(tc, ec, t)
        assert np.allclose(D, D_ref, rtol=1e-12)
        assert np.allclose(H, node_costs(net.tg, D_ref, ec, t), rtol=1e-12)
        for s in range(0, net.rg.n, 17):
            if s != t:
                assert dijkstra(sp, s, t).cost == pytest.approx(H[s], rel=1e-10)


@pytest.mark.parametrize("w", WEIGHTS)
def test_h_adm_admissible_and_consistent(net, w):
    cm = net.cm
    assert cm.rho(w).min() >= BASE
    ec = cm.edge_cost(w)
    solver = BackwardSolver(net.tg)
    proper = net.rg.tail != net.rg.head
    for t in range(0, net.rg.n, 7):
        h = cm.h_adm(t, w)
        D, H = solver.costs(cm.transition_cost(w, ec), ec, t)
        assert h[t] == 0
        assert np.all(h <= H + 1e-9)                                        # admissible (nodes)
        assert np.all(h[net.rg.head] <= D + 1e-9)                           # admissible (states)
        u, v = net.rg.tail[proper], net.rg.head[proper]
        assert np.all(h[u] <= ec[proper] + h[v] + 1e-9)                     # consistent


@pytest.mark.parametrize("w", WEIGHTS)
def test_astar_optimal_with_fewer_expansions(net, w):
    sp = SearchSpace.build(net.cm, w)
    for s, t in queries(net.rg):
        d = dijkstra(sp, s, t)
        a = astar(sp, s, t, admissible_heuristic(net.cm, t, w))
        assert a.cost == pytest.approx(d.cost, rel=1e-12)
        assert a.expanded <= d.expanded
        assert a.reexpanded == 0


def test_perfect_heuristic_expands_only_the_path(net):
    w = WEIGHTS[2]
    sp = SearchSpace.build(net.cm, w)
    solver = BackwardSolver(net.tg)
    for s, t in queries(net.rg, 10):
        D, _ = solver.costs(np.array(sp.trans_cost), np.array(sp.edge_cost), t)
        r = astar(sp, s, t, perfect_heuristic(D))
        assert r.cost == pytest.approx(dijkstra(sp, s, t).cost, rel=1e-12)
        assert r.expanded <= len(r.path) + 1


@pytest.mark.parametrize("W", [1.25, 1.5, 2.0, 3.0])
def test_weighted_astar_bound(net, W):
    for w in WEIGHTS:
        sp = SearchSpace.build(net.cm, w)
        for s, t in queries(net.rg, 15):
            opt = dijkstra(sp, s, t).cost
            r = weighted_astar(sp, s, t, admissible_heuristic(net.cm, t, w), W)
            assert r.cost <= W * opt * (1 + 1e-12)


class AdversarialHeuristic(StateHeuristic):
    """Random, wildly inadmissible guide: Focal Search must still honour its bound."""

    def __init__(self, m, seed, scale):
        super().__init__(np.random.default_rng(seed).uniform(0, scale, size=m), name="adversarial")


@pytest.mark.parametrize("W,key", list(itertools.product([1.0, 1.1, 1.5, 2.0], ["f", "h"])))
def test_focal_bound_with_adversarial_guide(net, W, key):
    for wi, w in enumerate(WEIGHTS):
        sp = SearchSpace.build(net.cm, w)
        for qi, (s, t) in enumerate(queries(net.rg, 12, seed=wi)):
            opt = dijkstra(sp, s, t).cost
            guide = AdversarialHeuristic(net.tg.m, seed=100 * wi + qi, scale=5 * opt)
            r = focal_search(sp, s, t, admissible_heuristic(net.cm, t, w), guide, W, key)
            assert r.found
            assert r.cost <= W * opt * (1 + 1e-12)
            assert r.cost == pytest.approx(net.cm.path_cost(r.path, w), rel=1e-10)
            assert r.extra["goal_g"] <= r.extra["bound"] * (1 + 1e-12)
            if W == 1.0:
                assert r.cost == pytest.approx(opt, rel=1e-12)


def test_focal_with_perfect_guide_and_without_reopening(net):
    w = WEIGHTS[5]
    sp = SearchSpace.build(net.cm, w)
    solver = BackwardSolver(net.tg)
    for s, t in queries(net.rg, 10):
        _, H = solver.costs(np.array(sp.trans_cost), np.array(sp.edge_cost), t)
        h = admissible_heuristic(net.cm, t, w)
        opt = H[s]
        r = focal_search(sp, s, t, h, NodeHeuristic(H, net.tg.head_list), 1.5)
        assert r.cost <= 1.5 * opt * (1 + 1e-12)
        r2 = focal_search(sp, s, t, h, NodeHeuristic(H, net.tg.head_list), 1.5, reopen=False)
        assert r2.found


@pytest.mark.parametrize("W", [1.0, 1.25, 2.0])
def test_focal_never_expands_fewer_than_its_floor(net, W):
    from wayfinder.search.backward import focal_floor
    solver = BackwardSolver(net.tg)
    for wi, w in enumerate(WEIGHTS):
        sp = SearchSpace.build(net.cm, w)
        for qi, (s, t) in enumerate(queries(net.rg, 8, seed=wi)):
            D, H = solver.costs(np.array(sp.trans_cost), np.array(sp.edge_cost), t)
            g_star = solver.forward_state_costs(sp.trans_cost, sp.edge_cost, s)
            assert np.allclose(g_star[net.rg.in_edges(t)].min(), H[s], rtol=1e-10)
            h = admissible_heuristic(net.cm, t, w)
            floor = focal_floor(g_star, h.values[net.rg.head], H[s], W, net.rg.head == t)
            for guide in (NodeHeuristic(H, net.tg.head_list), AdversarialHeuristic(net.tg.m, qi, 3 * H[s])):
                assert focal_search(sp, s, t, h, guide, W).expanded >= floor


def test_trivial_and_invalid_queries(net):
    sp = SearchSpace.build(net.cm, WEIGHTS[0])
    h = admissible_heuristic(net.cm, 3, WEIGHTS[0])
    assert dijkstra(sp, 3, 3).cost == 0 and dijkstra(sp, 3, 3).path == []
    assert focal_search(sp, 3, 3, h, h, 1.5).cost == 0
    with pytest.raises(ValueError):
        focal_search(sp, 0, 3, h, h, 0.9)
    with pytest.raises(ValueError):
        focal_search(sp, 0, 3, h, h, 1.5, focal_key="x")


def test_turn_penalties_change_routes(net):
    """Moving the junction slider must be able to change the chosen route."""
    sp0 = SearchSpace.build(net.cm, [1, 0, 0, 0, 0])
    sp1 = SearchSpace.build(net.cm, [1, 1, 0, 0, 0])
    changed = sum(dijkstra(sp0, s, t).path != dijkstra(sp1, s, t).path for s, t in queries(net.rg, 30))
    assert changed > 0
