"""Checks on the real Jubilee Hills graph (skipped until `python -m wayfinder preprocess` has run)."""
import numpy as np
import pytest

from wayfinder.config import load_config
from wayfinder.network import load_network, processed_graph_path
from wayfinder.search import BackwardSolver, SearchSpace, admissible_heuristic, astar, dijkstra, focal_search

cfg = load_config()
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not processed_graph_path(cfg).exists(), reason="processed graph not built")]


@pytest.fixture(scope="module")
def real():
    return load_network(cfg)


def test_graph_matches_proposal(real):
    assert real.rg.n == 13246 and real.rg.m == 34708
    assert real.rg.meta["scc_share"] == 1.0
    assert 0.999999 < real.cm.hav_scale <= 1.0


@pytest.mark.parametrize("w", [[1, 0, 0, 0, 0], [0.2, 0.8, 0.8, 0.8, 0.8], [0, 1, 0, 1, 0]])
def test_h_adm_consistent_on_every_edge(real, w):
    rg, cm = real.rg, real.cm
    ec = cm.edge_cost(w)
    proper = rg.tail != rg.head
    for t in (0, 5000, 13000):
        h = cm.h_adm(t, w)
        assert np.all(h[rg.tail[proper]] <= ec[proper] + h[rg.head[proper]] + 1e-9)


def test_searches_agree_and_focal_bound(real):
    w = [0.3, 0.5, 0.5, 0.5, 0.5]
    sp = SearchSpace.build(real.cm, w)
    solver = BackwardSolver(real.tg)
    rng = np.random.default_rng(0)
    for _ in range(3):
        s, t = (int(v) for v in rng.choice(real.rg.n, size=2, replace=False))
        _, H = solver.costs(np.array(sp.trans_cost), np.array(sp.edge_cost), t)
        d = dijkstra(sp, s, t)
        h = admissible_heuristic(real.cm, t, w)
        assert d.cost == pytest.approx(H[s], rel=1e-10)
        assert astar(sp, s, t, h).cost == pytest.approx(d.cost, rel=1e-12)
        assert focal_search(sp, s, t, h, h, 2.0).cost <= 2.0 * d.cost * (1 + 1e-12)
