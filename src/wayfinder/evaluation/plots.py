"""All result figures (static PNGs for the report).

Colour follows the entity, never its rank: every search family has one fixed categorical slot
(validated reference palette, light mode) and keeps it in every figure. Lines are 2 px with 8 px
markers, grids and axes are recessive, text stays in ink colours, and every multi-series figure has a
legend.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

FAMILY_COLOR = {           # fixed slot per entity
    "astar_hadm": "#2a78d6",      # 1 blue
    "focal_learned": "#eb6834",   # 2 orange
    "focal_tuned": "#1baf7a",     # 3 aqua
    "wastar_hadm": "#eda100",     # 4 yellow
    "astar_learned": "#e87ba4",   # 5 magenta
    "dijkstra": "#008300",        # 6 green
    "focal_hstar": "#6250d6",     # 7 violet
    "astar_hstar": "#e34948",     # 8 red
    "h_adm": "#2a78d6", "h_tuned": "#1baf7a", "h_learn": "#eb6834",
}
LABEL = {"dijkstra": "Dijkstra", "astar_hadm": "A* (h_adm)", "astar_hstar": "A* (h*, oracle)",
         "wastar_hadm": "Weighted A* (h_adm)", "focal_learned": "Focal (learned)",
         "focal_tuned": "Focal (tuned classical)", "focal_hstar": "Focal (h* guide, oracle)",
         "astar_learned": "A* (learned, unbounded)", "h_adm": "h_adm", "h_tuned": "h_tuned",
         "h_learn": "h_learn"}


def _style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2)


def _fig(w=6.4, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE)
    return fig, ax


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def _line(ax, x, y, fam, label=None, **kw):
    ax.plot(x, y, "-o", color=FAMILY_COLOR[fam], linewidth=2, markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=label or LABEL[fam], **kw)


def plot_problem_curve(curve: pd.DataFrame, path: Path):
    fig, ax = _fig()
    for fam in ("dijkstra", "astar_hadm", "focal_learned", "astar_hstar"):
        cols = [c for c in curve.columns if c.startswith(fam) and c.endswith("_mean")]
        if cols:
            _line(ax, curve["lambda"], curve[cols[0]], fam,
                  label=LABEL[fam] + (f" W={cols[0].split('_W')[1].split('_')[0]}" if "_W" in cols[0] else ""))
    _style(ax, "Expansions climb as preferences move away from distance",
           "weight on non-distance sliders (λ)", "mean states expanded per query")
    top = max(float(curve[c].max()) for c in curve.columns if c.endswith("_mean"))
    ax.set_ylim(0, top * 1.35)                       # headroom so the legend never covers a line
    ax.legend(frameon=False, labelcolor=INK_2, loc="upper left", ncol=2, fontsize=8)
    _save(fig, path)


def plot_bound(bound: pd.DataFrame, path: Path):
    fig, ax = _fig()
    Ws = sorted(bound["W"].dropna().unique())
    ax.plot([min(Ws), max(Ws)], [min(Ws), max(Ws)], "--", color=INK_2, linewidth=1.5, label="guaranteed bound W")
    for fam in ("focal_learned", "focal_tuned", "wastar_hadm"):
        d = bound[bound["family"] == fam]
        if not len(d):
            continue
        g = d.groupby("W").agg(mx=("achieved_max", "max"), mean=("achieved_mean", "median")).reset_index()
        _line(ax, g["W"], g["mx"], fam, label=f"{LABEL[fam]}: worst query")
        ax.plot(g["W"], g["mean"], ":", color=FAMILY_COLOR[fam], linewidth=2, label=f"{LABEL[fam]}: mean")
    _style(ax, "Achieved suboptimality stays under the guarantee",
           "guaranteed factor W", "achieved cost / optimal cost")
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=8)
    _save(fig, path)


def plot_expansions_vs_W(bound: pd.DataFrame, summary: pd.DataFrame, path: Path):
    fig, ax = _fig()
    for fam in ("focal_learned", "focal_tuned", "wastar_hadm", "focal_hstar"):
        d = bound[bound["family"] == fam]
        if len(d):
            g = d.groupby("W")["expanded_median"].median().reset_index()
            _line(ax, g["W"], g["expanded_median"], fam)
    fl = bound[bound["family"] == "focal_learned"]
    if len(fl) and fl["focal_floor_median"].notna().any():
        g = fl.groupby("W")["focal_floor_median"].median().reset_index()
        ax.plot(g["W"], g["focal_floor_median"], ":", color=INK_2, linewidth=2,
                label="Focal floor (needed by any guide)")
    for fam in ("astar_hadm", "dijkstra"):
        hit = summary[summary["group"] == fam]
        if len(hit):
            ax.axhline(float(hit["expanded_median"].iloc[0]), color=FAMILY_COLOR[fam], linewidth=2,
                       linestyle="--", label=LABEL[fam])
    _style(ax, "Search effort against the suboptimality bound", "bound W", "median states expanded")
    ax.set_yscale("log")
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=8)
    _save(fig, path)


def plot_informedness(samples: pd.DataFrame, path: Path):
    fig, ax = _fig()
    bins = np.linspace(0, 2.0, 61)
    for name in ("h_adm", "h_tuned", "h_learn_s0"):
        d = samples[samples["heuristic"] == name]["ratio"]
        if len(d):
            fam = "h_learn" if name.startswith("h_learn") else name
            ax.hist(np.clip(d, 0, 2.0), bins=bins, histtype="step", linewidth=2, color=FAMILY_COLOR[fam],
                    label=LABEL[fam] + (" (seed 0)" if fam == "h_learn" else ""))
    ax.axvline(1.0, color=INK_2, linewidth=1.5, linestyle="--", label="h = h* (perfect)")
    _style(ax, "Informedness h / h* (right of 1 = overestimate)", "h / h* (clipped at 2)", "sampled nodes")
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=8)
    _save(fig, path)


def plot_generalisation(gen: pd.DataFrame, groups: list[str], path: Path):
    d = gen[gen["group"].isin(groups)]
    if not len(d):
        return
    fig, ax = _fig(7.0, 4.2)
    x = np.arange(len(groups))
    width = 0.38
    for i, (ws, color) in enumerate((("train", "#2a78d6"), ("heldout", "#eb6834"))):
        vals = [float(d[(d["group"] == g) & (d["weight_set"] == ws)]["expanded_median"].iloc[0])
                if len(d[(d["group"] == g) & (d["weight_set"] == ws)]) else np.nan for g in groups]
        ax.bar(x + (i - 0.5) * (width + 0.02), vals, width, color=color,
               label="training weight vectors" if ws == "train" else "held-out weight vectors")
    ax.set_xticks(x, groups, rotation=20, ha="right", fontsize=8)
    _style(ax, "Seen vs held-out preferences", "", "median states expanded")
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=8)
    _save(fig, path)


def plot_training_curves(logs: dict[int, pd.DataFrame], path: Path):
    fig, ax = _fig()
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    for (seed, log), c in zip(sorted(logs.items()), colors):
        ax.plot(log["epoch"], log["train_mse_log"], "-", color=c, linewidth=2, label=f"seed {seed} train")
        ax.plot(log["epoch"], log["val_mse_log"], ":", color=c, linewidth=2, label=f"seed {seed} val")
    _style(ax, "Training curves (MSE of log ratio)", "epoch", "MSE of log(h*/h_adm)")
    ax.set_yscale("log")
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=8, ncol=2)
    _save(fig, path)


def plot_latency(raw: pd.DataFrame, groups: list[str], path: Path):
    data = [raw[raw["group"] == g]["latency_ms"].values for g in groups if (raw["group"] == g).any()]
    names = [g for g in groups if (raw["group"] == g).any()]
    if not data:
        return
    fig, ax = _fig(7.0, 4.2)
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.5)
    for patch, g in zip(bp["boxes"], names):
        fam = raw[raw["group"] == g]["family"].iloc[0]
        patch.set_facecolor(FAMILY_COLOR.get(fam, "#2a78d6"))
        patch.set_edgecolor(SURFACE)
    for k in ("medians",):
        for line in bp[k]:
            line.set_color(INK)
    ax.set_xticks(range(1, len(names) + 1), names, rotation=20, ha="right", fontsize=8)
    _style(ax, "Query latency (single core, Python)", "", "latency per query (ms)")
    _save(fig, path)
