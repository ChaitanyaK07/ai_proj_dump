"""Shared fixtures: a small synthetic street grid near the study area (the Week-1 fixture graph)."""
from __future__ import annotations

import numpy as np
import pytest

from wayfinder import geo
from wayfinder.config import load_config
from wayfinder.graph.road_graph import RoadGraph
from wayfinder.network import load_network

CLASSES = ["residential", "primary", "footway", "secondary", "service", "tertiary", "living_street"]


def make_grid_graph(k: int = 9, spacing: float = 80.0, seed: int = 0) -> RoadGraph:
    """k x k grid with two-way streets, a few one-way streets, stairs, diagonals and a self-loop."""
    rng = np.random.default_rng(seed)
    x0, y0 = geo.latlon_to_utm(17.4220, 78.4200)
    nodes = [(i, j) for i in range(k) for j in range(k)]
    idx = {p: n for n, p in enumerate(nodes)}
    x = np.array([float(x0) + i * spacing + rng.uniform(-10, 10) for i, j in nodes])
    y = np.array([float(y0) + j * spacing + rng.uniform(-10, 10) for i, j in nodes])
    lat, lon = geo.utm_to_latlon(x, y)
    elev = 520 + 25 * np.sin(x / 300.0) + 15 * np.cos(y / 250.0)

    pairs = []
    for (i, j) in nodes:
        for di, dj in ((1, 0), (0, 1)):
            if (i + di, j + dj) in idx:
                pairs.append((idx[(i, j)], idx[(i + di, j + dj)]))
    for _ in range(k):                                      # some diagonals
        i, j = rng.integers(0, k - 1, size=2)
        pairs.append((idx[(i, j)], idx[(i + 1, j + 1)]))

    tail, head, length, hw, steps, canopy, coords, offsets = [], [], [], [], [], [], [], [0]

    def add(u, v, cls, L):
        tail.append(u)
        head.append(v)
        length.append(L)
        hw.append(cls)
        steps.append(cls == "steps")
        canopy.append(float(rng.uniform(0, 1) ** 2))
        if u == v:
            pts = [(x[u], y[u]), (x[u] + 20, y[u] + 20), (x[u] + 20, y[u] - 20), (x[u], y[u])]
        else:
            mid = ((x[u] + x[v]) / 2 + rng.uniform(-5, 5), (y[u] + y[v]) / 2 + rng.uniform(-5, 5))
            pts = [(x[u], y[u]), mid, (x[v], y[v])]
        coords.extend(pts)
        offsets.append(len(coords))

    for n, (u, v) in enumerate(pairs):
        cls = "steps" if n % 17 == 5 else CLASSES[n % len(CLASSES)]
        L = float(geo.haversine_m(lat[u], lon[u], lat[v], lon[v])) * (1 + rng.uniform(0, 0.15))
        add(u, v, cls, L)
        if n % 11 != 3:                                     # a few one-way streets
            add(v, u, cls, L)
    add(idx[(0, 0)], idx[(0, 0)], "service", 70.0)          # cul-de-sac loop

    tail, head = np.array(tail), np.array(head)
    grade = np.where(tail != head, (elev[head] - elev[tail]) / np.array(length), 0.0)
    b0, b1 = [], []
    for e in range(len(tail)):
        pts = coords[offsets[e]:offsets[e + 1]]
        b0.append(geo.bearing(pts[0], pts[1]))
        b1.append(geo.bearing(pts[-2], pts[-1]))
    rg = RoadGraph(osmid=np.array([str(1000 + n) for n in range(len(nodes))]), x=x, y=y, lat=np.asarray(lat),
                   lon=np.asarray(lon), elevation=elev, tail=tail, head=head, length=np.array(length),
                   highway=np.array(hw), steps=np.array(steps), canopy=np.array(canopy), grade=grade,
                   name=np.array([f"Road {e % 7}" for e in range(len(tail))]),
                   geom_coords=np.array(coords), geom_offsets=np.array(offsets),
                   b_start=np.array(b0), b_end=np.array(b1), meta={"source": "fixture", "scc_share": 1.0})
    import networkx as nx
    G = nx.DiGraph(list(zip(tail.tolist(), head.tolist())))
    assert nx.is_strongly_connected(G), "fixture must be strongly connected"
    return rg


@pytest.fixture(scope="session")
def grid():
    return make_grid_graph()


@pytest.fixture(scope="session")
def net(grid):
    return load_network(load_config(), rg=grid)


@pytest.fixture
def tmp_cfg(tmp_path):
    return load_config(overrides={"paths": {
        "processed_dir": str(tmp_path / "processed"), "splits_dir": str(tmp_path / "splits"),
        "benchmark_dir": str(tmp_path / "benchmark"), "labels_dir": str(tmp_path / "labels"),
        "models_dir": str(tmp_path / "models"), "results_dir": str(tmp_path / "results"),
        "study_dir": str(tmp_path / "study")}})


WEIGHTS = [
    np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.2, 0.8, 0.0, 0.0, 0.0]),
    np.array([0.1, 0.3, 0.9, 0.5, 0.7]),
    np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.5, 1.0, 1.0, 1.0, 1.0]),
]
