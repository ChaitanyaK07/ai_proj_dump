"""node2vec (Grover & Leskovec, 2016), implemented from scratch.

1. Second-order biased random walks on the undirected street graph. Having moved ``t -> v``, the next
   node ``x`` in N(v) is drawn with unnormalised probability ``1/p`` if ``x == t`` (return), ``1`` if
   ``x`` is also a neighbour of ``t`` (stay close) and ``1/q`` otherwise (move outward).
   All walks advance in lock-step, vectorised over a padded neighbour matrix.
2. Skip-gram with negative sampling over (centre, context) pairs drawn within a window, negatives from
   the unigram distribution raised to 3/4, trained with Adam in PyTorch.
"""
from __future__ import annotations

import time

import numpy as np
import torch


def undirected_neighbours(tail: np.ndarray, head: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Padded neighbour matrix (n, max_degree) with -1 padding, and the degree of every node."""
    mask = tail != head
    a = np.concatenate([tail[mask], head[mask]])
    b = np.concatenate([head[mask], tail[mask]])
    pairs = np.unique(np.stack([a, b], axis=1), axis=0)
    deg = np.bincount(pairs[:, 0], minlength=n)
    if np.any(deg == 0):
        raise ValueError("isolated node: node2vec needs every node to have a neighbour")
    NB = np.full((n, int(deg.max())), -1, dtype=np.int64)
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(deg, out=ptr[1:])
    col = np.arange(len(pairs)) - ptr[pairs[:, 0]]
    NB[pairs[:, 0], col] = pairs[:, 1]
    return NB, deg


def random_walks(NB: np.ndarray, deg: np.ndarray, walk_length: int, walks_per_node: int,
                 p: float, q: float, rng: np.random.Generator) -> np.ndarray:
    n = len(deg)
    starts = np.tile(np.arange(n), walks_per_node)
    rng.shuffle(starts)
    walks = np.empty((len(starts), walk_length), dtype=np.int64)
    walks[:, 0] = starts
    if walk_length == 1:
        return walks
    # first step: uniform over neighbours
    pick = (rng.uniform(size=len(starts)) * deg[starts]).astype(np.int64)
    walks[:, 1] = NB[starts, pick]
    rows = np.arange(len(starts))
    for step in range(2, walk_length):
        prev, cur = walks[:, step - 2], walks[:, step - 1]
        cand = NB[cur]                                           # (W, D)
        valid = cand >= 0
        prev_nb = NB[prev]                                       # (W, D)
        is_return = cand == prev[:, None]
        is_close = (cand[:, :, None] == prev_nb[:, None, :]).any(axis=2) & valid
        bias = np.where(is_return, 1.0 / p, np.where(is_close, 1.0, 1.0 / q))
        bias = np.where(valid, bias, 0.0)
        cum = np.cumsum(bias, axis=1)
        r = rng.uniform(size=len(cur)) * cum[:, -1]
        choice = (cum < r[:, None]).sum(axis=1)
        choice = np.minimum(choice, deg[cur] - 1)
        walks[:, step] = cand[rows, choice]
    return walks


class SkipGram(torch.nn.Module):
    def __init__(self, n: int, dim: int):
        super().__init__()
        self.inp = torch.nn.Embedding(n, dim)
        self.out = torch.nn.Embedding(n, dim)
        torch.nn.init.uniform_(self.inp.weight, -0.5 / dim, 0.5 / dim)
        torch.nn.init.zeros_(self.out.weight)

    def loss(self, centre, context, negatives):
        u = self.inp(centre)                                     # (B, d)
        v = self.out(context)                                    # (B, d)
        vn = self.out(negatives)                                 # (B, k, d)
        pos = torch.nn.functional.logsigmoid((u * v).sum(-1))
        neg = torch.nn.functional.logsigmoid(-(vn @ u.unsqueeze(-1)).squeeze(-1)).sum(-1)
        return -(pos + neg).mean()


def train_node2vec(tail: np.ndarray, head: np.ndarray, n: int, dim: int = 32, walk_length: int = 40,
                   walks_per_node: int = 10, window: int = 5, p: float = 1.0, q: float = 0.5,
                   negatives: int = 5, epochs: int = 2, batch_size: int = 8192, lr: float = 0.01,
                   seed: int = 0, verbose: bool = True) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    NB, deg = undirected_neighbours(tail, head, n)
    walks = random_walks(NB, deg, walk_length, walks_per_node, p, q, rng)
    t_walk = time.perf_counter() - t0
    if verbose:
        print(f"  {len(walks)} walks of length {walk_length} in {t_walk:.1f} s", flush=True)

    counts = np.bincount(walks.ravel(), minlength=n).astype(np.float64)
    noise = counts ** 0.75
    noise /= noise.sum()
    noise_t = torch.tensor(noise, dtype=torch.float)

    model = SkipGram(n, dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_walks, L = walks.shape
    walks_t = torch.from_numpy(walks)
    pairs_per_epoch = walks.size * window
    steps = max(1, pairs_per_epoch // batch_size)
    gen = torch.Generator().manual_seed(seed)
    history = []
    for ep in range(epochs):
        total = 0.0
        for _ in range(steps):
            wi = torch.randint(0, n_walks, (batch_size,), generator=gen)
            pos = torch.randint(0, L, (batch_size,), generator=gen)
            off = torch.randint(1, window + 1, (batch_size,), generator=gen)
            sign = torch.randint(0, 2, (batch_size,), generator=gen) * 2 - 1
            ctx = pos + sign * off
            ctx = torch.where((ctx < 0) | (ctx >= L), pos - sign * off, ctx)   # reflect at walk ends
            ctx = ctx.clamp(0, L - 1)
            centre = walks_t[wi, pos]
            context = walks_t[wi, ctx]
            neg = torch.multinomial(noise_t, batch_size * negatives, replacement=True,
                                    generator=gen).view(batch_size, negatives)
            loss = model.loss(centre, context, neg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        history.append(total / steps)
        if verbose:
            print(f"  skip-gram epoch {ep + 1}/{epochs}: loss {history[-1]:.4f}", flush=True)

    emb = model.inp.weight.detach().numpy().astype(np.float32)
    info = {"dim": dim, "walk_length": walk_length, "walks_per_node": walks_per_node, "window": window,
            "p": p, "q": q, "negatives": negatives, "epochs": epochs, "seed": seed, "loss": history,
            "walk_seconds": t_walk, "total_seconds": time.perf_counter() - t0}
    return emb, info
