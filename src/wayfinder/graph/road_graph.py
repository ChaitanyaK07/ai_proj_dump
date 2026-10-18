"""The walking network as compact numpy arrays.

``RoadGraph.from_graphml`` reads the OSMnx GraphML written by ``scripts/acquisition/build_dataset.py``
(projected to UTM 44N, with ``canopy`` and ``grade`` on every edge), keeps the largest strongly
connected component, orients every edge geometry from tail to head, and records start/end bearings.
Everything downstream (features, turn graph, search, learning) works on integer node and edge ids.
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wayfinder import geo


def highway_classes(value) -> list[str]:
    """OSM highway value(s) of an edge with any ``_link`` suffix removed."""
    if isinstance(value, str) and value.startswith("["):
        value = ast.literal_eval(value)
    vals = value if isinstance(value, (list, tuple)) else [value]
    return [str(v).replace("_link", "") for v in vals if v is not None and str(v) != "nan"]


def _first_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str) and value.startswith("["):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return str(value)


def _csr(keys: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Group item ids by ``keys`` (stable): returns ``ptr`` (n + 1) and ``idx``."""
    order = np.argsort(keys, kind="stable").astype(np.int64)
    counts = np.bincount(keys, minlength=n)
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    return ptr, order


@dataclass
class RoadGraph:
    # nodes
    osmid: np.ndarray          # (n,) str
    x: np.ndarray              # (n,) UTM easting, m
    y: np.ndarray              # (n,) UTM northing, m
    lat: np.ndarray            # (n,) degrees
    lon: np.ndarray            # (n,) degrees
    elevation: np.ndarray      # (n,) m (NaN if unknown)
    # edges
    tail: np.ndarray           # (m,) int
    head: np.ndarray           # (m,) int
    length: np.ndarray         # (m,) m
    highway: np.ndarray        # (m,) str, classes joined by "|"
    steps: np.ndarray          # (m,) bool, highway=steps
    canopy: np.ndarray         # (m,) tree-cover share in the 15 m buffer, [0, 1]
    grade: np.ndarray          # (m,) signed grade along tail -> head
    name: np.ndarray           # (m,) str street name ("" if none)
    geom_coords: np.ndarray    # (P, 2) UTM vertices of every edge, concatenated
    geom_offsets: np.ndarray   # (m + 1,) slice of geom_coords per edge
    b_start: np.ndarray        # (m,) bearing leaving the tail, degrees (NaN if degenerate)
    b_end: np.ndarray          # (m,) bearing arriving at the head, degrees (NaN if degenerate)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.n = len(self.x)
        self.m = len(self.tail)
        self.out_ptr, self.out_idx = _csr(self.tail, self.n)
        self.in_ptr, self.in_idx = _csr(self.head, self.n)
        self.degree = self._undirected_degree()
        self._kdtree = None

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_graphml(cls, path: str | Path, largest_scc: bool = True) -> "RoadGraph":
        import networkx as nx

        G = nx.read_graphml(str(path), force_multigraph=True)
        n_raw, m_raw = G.number_of_nodes(), G.number_of_edges()
        scc_share = 1.0
        if largest_scc:
            scc = max(nx.strongly_connected_components(G), key=len)
            scc_share = len(scc) / n_raw
            if len(scc) < n_raw:
                G = G.subgraph(scc).copy()

        nodes = list(G.nodes)
        index = {nid: i for i, nid in enumerate(nodes)}
        nd = G.nodes
        x = np.array([float(nd[v]["x"]) for v in nodes])
        y = np.array([float(nd[v]["y"]) for v in nodes])
        elev = np.array([float(nd[v].get("elevation", "nan")) for v in nodes])
        lat, lon = geo.utm_to_latlon(x, y)

        tail, head, length, hw, steps, canopy, grade, name = [], [], [], [], [], [], [], []
        coords, offsets, b0, b1 = [], [0], [], []
        for u, v, _k, d in G.edges(keys=True, data=True):
            iu, iv = index[u], index[v]
            pu, pv = (x[iu], y[iu]), (x[iv], y[iv])
            if "geometry" in d and d["geometry"]:
                pts = geo.parse_linestring(d["geometry"])
                if iu != iv and math.dist(pts[0], pu) > math.dist(pts[-1], pu):
                    pts = pts[::-1]
            else:
                pts = [pu, pv]
            cl = highway_classes(d.get("highway", "residential")) or ["residential"]
            L = float(d["length"])
            g = d.get("grade")
            if g is None or g == "" or str(g) == "nan":
                g = (elev[iv] - elev[iu]) / L if L > 0 else 0.0
            g = float(g)
            if not math.isfinite(g):
                g = 0.0
            tail.append(iu)
            head.append(iv)
            length.append(L)
            hw.append("|".join(cl))
            steps.append("steps" in cl)
            canopy.append(min(1.0, max(0.0, float(d.get("canopy", 0.0) or 0.0))))
            grade.append(g)
            name.append(_first_text(d.get("name")))
            coords.extend(pts)
            offsets.append(len(coords))
            b0.append(next((b for b in (geo.bearing(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
                            if not math.isnan(b)), math.nan))
            b1.append(next((b for b in (geo.bearing(pts[i - 1], pts[i]) for i in range(len(pts) - 1, 0, -1))
                            if not math.isnan(b)), math.nan))

        meta = {"source": str(Path(path).name), "crs": G.graph.get("crs", "EPSG:32644"),
                "raw_nodes": n_raw, "raw_edges": m_raw, "scc_share": scc_share}
        return cls(osmid=np.array(nodes, dtype=str), x=x, y=y, lat=np.asarray(lat), lon=np.asarray(lon),
                   elevation=elev, tail=np.array(tail, dtype=np.int64), head=np.array(head, dtype=np.int64),
                   length=np.array(length), highway=np.array(hw, dtype=str), steps=np.array(steps, dtype=bool),
                   canopy=np.array(canopy), grade=np.array(grade), name=np.array(name, dtype=str),
                   geom_coords=np.array(coords, dtype=float).reshape(-1, 2),
                   geom_offsets=np.array(offsets, dtype=np.int64),
                   b_start=np.array(b0, dtype=float), b_end=np.array(b1, dtype=float), meta=meta)

    # ------------------------------------------------------------------ persistence
    _ARRAYS = ("osmid", "x", "y", "lat", "lon", "elevation", "tail", "head", "length", "highway", "steps",
               "canopy", "grade", "name", "geom_coords", "geom_offsets", "b_start", "b_end")

    def save(self, path: str | Path) -> None:
        import json
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: getattr(self, k) for k in self._ARRAYS},
                            meta=np.array(json.dumps(self.meta)))

    @classmethod
    def load(cls, path: str | Path) -> "RoadGraph":
        import json
        with np.load(path, allow_pickle=False) as z:
            kw = {k: z[k] for k in cls._ARRAYS}
            meta = json.loads(str(z["meta"]))
        return cls(**kw, meta=meta)

    # ------------------------------------------------------------------ queries
    def _undirected_degree(self) -> np.ndarray:
        """Number of distinct neighbouring nodes (in either direction), self-loops excluded."""
        mask = self.tail != self.head
        a = np.concatenate([self.tail[mask], self.head[mask]])
        b = np.concatenate([self.head[mask], self.tail[mask]])
        pairs = np.unique(np.stack([a, b], axis=1), axis=0) if len(a) else np.zeros((0, 2), dtype=np.int64)
        return np.bincount(pairs[:, 0], minlength=self.n) if len(pairs) else np.zeros(self.n, dtype=np.int64)

    def out_edges(self, v: int) -> np.ndarray:
        return self.out_idx[self.out_ptr[v]:self.out_ptr[v + 1]]

    def in_edges(self, v: int) -> np.ndarray:
        return self.in_idx[self.in_ptr[v]:self.in_ptr[v + 1]]

    def edge_geometry(self, e: int) -> np.ndarray:
        return self.geom_coords[self.geom_offsets[e]:self.geom_offsets[e + 1]]

    def path_latlon(self, edges) -> list[list[float]]:
        """Route polyline as ``[[lat, lon], ...]`` following the full edge geometry."""
        pts: list[np.ndarray] = []
        for i, e in enumerate(edges):
            g = self.edge_geometry(int(e))
            pts.append(g if i == 0 else g[1:])
        if not pts:
            return []
        xy = np.vstack(pts)
        lat, lon = geo.utm_to_latlon(xy[:, 0], xy[:, 1])
        return [[float(a), float(b)] for a, b in zip(lat, lon)]

    def nearest_node(self, lat: float, lon: float) -> tuple[int, float]:
        """Closest graph node to a WGS84 point and its distance in metres."""
        from scipy.spatial import cKDTree
        if self._kdtree is None:
            self._kdtree = cKDTree(np.column_stack([self.x, self.y]))
        x, y = geo.latlon_to_utm(lat, lon)
        dist, i = self._kdtree.query([float(x), float(y)])
        return int(i), float(dist)

    def node_index(self, osmid: str) -> int:
        hits = np.nonzero(self.osmid == str(osmid))[0]
        if not len(hits):
            raise KeyError(f"OSM node {osmid} is not in the graph")
        return int(hits[0])

    def bounds_latlon(self) -> dict:
        return {"south": float(self.lat.min()), "west": float(self.lon.min()),
                "north": float(self.lat.max()), "east": float(self.lon.max())}
