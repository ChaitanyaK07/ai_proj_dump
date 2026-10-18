"""The preference-conditioned heuristic network and its input features.

    x      = [ node2vec(v), node2vec(t), w (5 sliders), (Δx, Δy, ‖Δ‖) / 1 km ]      Δ = position(t) - position(v)
    r_hat  = 1 + softplus(MLP(x))  >= 1
    h_learn(v, t, w) = r_hat * h_adm(v, t, w)

Because ``r_hat >= 1`` the learned heuristic is never less informed than the classical one. The
regression target is ``log(h* / h_adm)``; the loss is the mean squared error in log space.
Embeddings are standardised per dimension (statistics stored with the model) and kept frozen.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

from wayfinder import WEIGHT_KEYS

OFFSET_SCALE_M = 1000.0
N_WEIGHTS = len(WEIGHT_KEYS)
N_OFFSET = 3


class HeuristicNet(nn.Module):
    def __init__(self, emb_dim: int, hidden: Sequence[int] = (256, 256)):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden = list(hidden)
        dims = [2 * emb_dim + N_WEIGHTS + N_OFFSET, *hidden]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predicted ratio r_hat >= 1, shape (batch,)."""
        return 1.0 + nn.functional.softplus(self.mlp(x).squeeze(-1))

    def log_ratio(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(self.forward(x))


class FeatureBuilder:
    """Builds network inputs for (node, target, weight) index triples, entirely on tensors."""

    def __init__(self, emb: np.ndarray, x: np.ndarray, y: np.ndarray, emb_mean: np.ndarray | None = None,
                 emb_std: np.ndarray | None = None):
        emb = np.asarray(emb, dtype=np.float32)
        self.emb_mean = emb.mean(0) if emb_mean is None else np.asarray(emb_mean, dtype=np.float32)
        self.emb_std = (emb.std(0) + 1e-8) if emb_std is None else np.asarray(emb_std, dtype=np.float32)
        self.E = torch.from_numpy((emb - self.emb_mean) / self.emb_std)
        self.X = torch.from_numpy(np.asarray(x, dtype=np.float64))
        self.Y = torch.from_numpy(np.asarray(y, dtype=np.float64))

    @property
    def emb_dim(self) -> int:
        return int(self.E.shape[1])

    def build(self, v: torch.Tensor, t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        dx = ((self.X[t] - self.X[v]) / OFFSET_SCALE_M).float()
        dy = ((self.Y[t] - self.Y[v]) / OFFSET_SCALE_M).float()
        dist = torch.sqrt(dx * dx + dy * dy)
        return torch.cat([self.E[v], self.E[t], w.float(), torch.stack([dx, dy, dist], dim=1)], dim=1)


def save_checkpoint(path, net: HeuristicNet, fb: FeatureBuilder, meta: dict) -> None:
    torch.save({"state_dict": net.state_dict(), "emb_dim": net.emb_dim, "hidden": net.hidden,
                "emb_mean": fb.emb_mean, "emb_std": fb.emb_std, "meta": meta}, path)


def load_checkpoint(path) -> tuple[HeuristicNet, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = HeuristicNet(ck["emb_dim"], ck["hidden"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck
