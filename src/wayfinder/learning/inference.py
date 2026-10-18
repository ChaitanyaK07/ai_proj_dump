"""Search-time inference: a numpy port of the trained network with cached embedding projections.

The first layer is linear in its inputs, so ``W1 x = W1_v e_v + W1_t e_t + W1_w w + W1_o offset``.
``W1_v e_v`` and ``W1_t e_t`` are precomputed once for every node (the cached embeddings), the weight
and target terms once per query, and only the 3-dimensional offset term and the remaining layers are
evaluated per node. ``LearnedHeuristic`` evaluates in batches, with a per-node cache:

* ``mode="tile"`` (default) — on a cache miss, the whole 250 m spatial tile containing the node is
  inferred in one batch, so only the region the search actually touches is computed, in a few dozen calls;
* ``mode="lazy"`` — exactly the newly generated successors of each expansion;
* ``mode="full"`` — every node once, at query start.

All three give the same values up to float32 rounding.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from wayfinder.cost import CostModel, as_weight_array
from wayfinder.learning.model import N_WEIGHTS, OFFSET_SCALE_M, load_checkpoint
from wayfinder.search.heuristics import Heuristic

TILE_M = 250.0
MODES = ("tile", "lazy", "full")


class HeuristicModel:
    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray]], emb_std_mean: tuple[np.ndarray, np.ndarray],
                 emb: np.ndarray, x: np.ndarray, y: np.ndarray, name: str = "learned"):
        (W1, b1), rest = layers[0], layers[1:]
        mean, std = emb_std_mean
        E = ((np.asarray(emb, dtype=np.float32) - mean) / std).astype(np.float32)
        d = E.shape[1]
        self.Pv = E @ W1[:, :d].T                     # (n, h1) cached node projections
        self.Pt = E @ W1[:, d:2 * d].T                # (n, h1) cached target projections
        self.W1w = W1[:, 2 * d:2 * d + N_WEIGHTS]     # (h1, 5)
        self.W1o = W1[:, 2 * d + N_WEIGHTS:]          # (h1, 3)
        self.b1 = b1
        self.rest = rest
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.name = name
        # spatial tiles for tile-batched inference
        cx = np.floor((self.x - self.x.min()) / TILE_M).astype(np.int64)
        cy = np.floor((self.y - self.y.min()) / TILE_M).astype(np.int64)
        _, tile = np.unique(cx * (cy.max() + 1) + cy, return_inverse=True)
        self.tile_of = tile.tolist()
        order = np.argsort(tile, kind="stable")
        bounds = np.searchsorted(tile[order], np.arange(tile.max() + 2))
        self.tile_members = [order[bounds[i]:bounds[i + 1]] for i in range(tile.max() + 1)]

    @classmethod
    def from_checkpoint(cls, path: str | Path, emb: np.ndarray, x: np.ndarray, y: np.ndarray) -> "HeuristicModel":
        net, ck = load_checkpoint(path)
        linears = [m for m in net.mlp if hasattr(m, "weight")]
        layers = [(m.weight.detach().numpy().astype(np.float32), m.bias.detach().numpy().astype(np.float32))
                  for m in linears]
        return cls(layers, (ck["emb_mean"], ck["emb_std"]), emb, x, y, name=Path(path).stem)

    def query_constant(self, target: int, w) -> np.ndarray:
        wa = as_weight_array(w).astype(np.float32)
        return self.Pt[target] + self.W1w @ wa + self.b1

    def ratio(self, nodes: np.ndarray, target: int, w, const: np.ndarray | None = None) -> np.ndarray:
        """r_hat for a batch of nodes towards ``target`` under ``w``."""
        nodes = np.asarray(nodes, dtype=np.int64)
        c = self.query_constant(target, w) if const is None else const
        dx = ((self.x[target] - self.x[nodes]) / OFFSET_SCALE_M).astype(np.float32)
        dy = ((self.y[target] - self.y[nodes]) / OFFSET_SCALE_M).astype(np.float32)
        off = np.column_stack([dx, dy, np.sqrt(dx * dx + dy * dy)])
        z = np.maximum(self.Pv[nodes] + c + off @ self.W1o.T, 0.0)
        for i, (Wl, bl) in enumerate(self.rest):
            z = z @ Wl.T + bl
            if i < len(self.rest) - 1:
                z = np.maximum(z, 0.0)
        z = z[:, 0].astype(np.float64)
        return 1.0 + np.logaddexp(0.0, z)


class LearnedHeuristic(Heuristic):
    """h_learn = r_hat * h_adm for one query, with batched inference."""

    def __init__(self, model: HeuristicModel, cm: CostModel, target: int, w, mode: str = "tile"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.name = model.name
        self.model, self.mode, self.target = model, mode, target
        self.w = as_weight_array(w)
        self.h_adm = cm.h_adm(target, self.w)
        self._head = cm.tg.head_list
        self._const = model.query_constant(target, self.w)
        self.n_inferred = 0
        self.n_batches = 0
        if mode == "full":
            self.values = self.node_values(np.arange(len(self.h_adm)))
            self._h = self.values.tolist()
        else:
            self._cache: dict[int, float] = {}

    def node_values(self, nodes: np.ndarray) -> np.ndarray:
        self.n_inferred += len(nodes)
        self.n_batches += 1
        r = self.model.ratio(nodes, self.target, self.w, self._const)
        return r * self.h_adm[nodes]

    def evaluate(self, states):
        head = self._head
        if self.mode == "full":
            h = self._h
            return [h[head[s]] for s in states]
        cache = self._cache
        nodes = [head[s] for s in states]
        missing = {v for v in nodes if v not in cache}
        if missing:
            if self.mode == "tile":
                tiles = {self.model.tile_of[v] for v in missing}
                batch = np.concatenate([self.model.tile_members[t] for t in tiles])
            else:
                batch = np.fromiter(missing, dtype=np.int64)
            for v, hv in zip(batch.tolist(), self.node_values(batch).tolist()):
                cache[v] = hv
        return [cache[v] for v in nodes]
