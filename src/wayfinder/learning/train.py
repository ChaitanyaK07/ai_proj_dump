"""Training loop, hyperparameter pass and checkpointing for the learned heuristic.

Data: the training-weight labels, split by target node (train targets for gradient steps, val targets
for early stopping and model selection). Test targets and the held-out weight vectors are never seen
here; they are scored by ``evaluate_label_sets`` after training.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wayfinder.learning.labels import split_mask
from wayfinder.learning.model import FeatureBuilder, HeuristicNet, save_checkpoint


class LabelTensors:
    """Label samples (v, t, weight row, log r*) as tensors, with v == t and degenerate rows removed."""

    def __init__(self, labels: dict, mask: np.ndarray | None = None, max_samples: int | None = None,
                 seed: int = 0):
        keep = labels["node"] != labels["target"]
        keep &= labels["h_adm"] > 0
        if mask is not None:
            keep &= mask
        idx = np.nonzero(keep)[0]
        if max_samples is not None and len(idx) > max_samples:
            idx = np.sort(np.random.default_rng(seed).choice(idx, size=max_samples, replace=False))
        ratio = labels["h_star"][idx] / labels["h_adm"][idx]
        self.v = torch.from_numpy(labels["node"][idx].astype(np.int64))
        self.t = torch.from_numpy(labels["target"][idx].astype(np.int64))
        self.w = torch.from_numpy(labels["weights"][labels["widx"][idx].astype(np.int64)]).float()
        self.log_r = torch.from_numpy(np.log(np.maximum(ratio, 1.0))).float()
        self.ratio = torch.from_numpy(ratio).float()

    def __len__(self):
        return len(self.v)


@torch.no_grad()
def predict_log_ratio(net: HeuristicNet, fb: FeatureBuilder, data: LabelTensors, batch: int = 65536) -> torch.Tensor:
    net.eval()
    out = []
    for a in range(0, len(data), batch):
        sl = slice(a, a + batch)
        out.append(net.log_ratio(fb.build(data.v[sl], data.t[sl], data.w[sl])))
    return torch.cat(out) if out else torch.zeros(0)


def label_metrics(net: HeuristicNet, fb: FeatureBuilder, data: LabelTensors) -> dict:
    """Regression and heuristic-quality metrics on a label set."""
    pred = predict_log_ratio(net, fb, data)
    err = pred - data.log_r
    r_hat = torch.exp(pred)
    return {
        "samples": len(data),
        "mse_log": float((err ** 2).mean()),
        "mae_log": float(err.abs().mean()),
        "overestimate_rate": float((r_hat > data.ratio * (1 + 1e-6)).float().mean()),
        "mean_overestimate_pct": float(((r_hat / data.ratio - 1).clamp(min=0) * 100).mean()),
        "informedness_learned": float((r_hat / data.ratio).clamp(max=10).mean()),   # h_learn / h*
        "informedness_adm": float((1.0 / data.ratio).mean()),                       # h_adm / h*
    }


def train_one(fb: FeatureBuilder, train: LabelTensors, val: LabelTensors, hidden, lr: float, epochs: int,
              batch_size: int, seed: int, weight_decay: float = 0.0, patience: int = 3,
              checkpoint: Path | None = None, meta: dict | None = None, verbose: bool = True):
    torch.manual_seed(seed)
    net = HeuristicNet(fb.emb_dim, hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad = float("inf"), None, 0
    log = []
    for ep in range(1, epochs + 1):
        net.train()
        t0 = time.perf_counter()
        perm = torch.randperm(len(train), generator=gen)
        total, nb = 0.0, 0
        for a in range(0, len(train), batch_size):
            b = perm[a:a + batch_size]
            x = fb.build(train.v[b], train.t[b], train.w[b])
            loss = ((net.log_ratio(x) - train.log_r[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        vm = label_metrics(net, fb, val)
        sched.step(vm["mse_log"])
        row = {"epoch": ep, "train_mse_log": total / max(nb, 1), "val_mse_log": vm["mse_log"],
               "val_mae_log": vm["mae_log"], "val_overestimate_rate": vm["overestimate_rate"],
               "lr": opt.param_groups[0]["lr"], "seconds": time.perf_counter() - t0}
        log.append(row)
        if verbose:
            print(f"    epoch {ep}: train {row['train_mse_log']:.5f}  val {row['val_mse_log']:.5f}  "
                  f"({row['seconds']:.0f} s)", flush=True)
        if vm["mse_log"] < best - 1e-7:
            best, bad = vm["mse_log"], 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            if checkpoint is not None:
                save_checkpoint(checkpoint, net, fb, {**(meta or {}), "epoch": ep, "val_mse_log": best,
                                                      "hidden": list(hidden), "lr": lr, "seed": seed})
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net, pd.DataFrame(log), best


def hyperparameter_pass(fb: FeatureBuilder, labels: dict, targets: dict, cfg: dict, out_csv: Path,
                        verbose: bool = True) -> dict:
    tc = cfg["training"]
    train = LabelTensors(labels, split_mask(labels, targets["train"]), tc["hparam_max_train_samples"], seed=0)
    val = LabelTensors(labels, split_mask(labels, targets["val"]))
    rows = []
    for hidden, lr in itertools.product(tc["hparam_grid"]["hidden"], tc["hparam_grid"]["lr"]):
        if verbose:
            print(f"  hparam: hidden={hidden} lr={lr}", flush=True)
        _, _, best = train_one(fb, train, val, hidden, lr, tc["hparam_epochs"], tc["batch_size"], seed=0,
                               weight_decay=tc["weight_decay"], patience=tc["hparam_epochs"], verbose=verbose)
        rows.append({"hidden": json.dumps(list(hidden)), "lr": lr, "val_mse_log": best})
    df = pd.DataFrame(rows).sort_values("val_mse_log")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    top = df.iloc[0]
    return {"hidden": json.loads(top["hidden"]), "lr": float(top["lr"])}
