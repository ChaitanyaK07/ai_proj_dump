"""The pipeline steps behind ``python -m wayfinder <step>``. Each step reads and writes files only."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from wayfinder import PER_METRE_KEYS, WEIGHT_KEYS
from wayfinder.config import Config
from wayfinder.network import build_road_graph, load_network, processed_graph_path


def _banner(name: str):
    print(f"\n=== {name} ===", flush=True)


# ---------------------------------------------------------------------------------------------- graph
def step_preprocess(cfg: Config) -> dict:
    _banner("preprocess: graphml -> arrays, features, turn graph")
    from wayfinder.cost import CostModel
    from wayfinder.graph.features import FEATURE_STATUS
    from wayfinder.graph.turn_graph import TurnGraph

    t0 = time.perf_counter()
    rg = build_road_graph(cfg)
    p = processed_graph_path(cfg)
    rg.save(p)
    tg = TurnGraph(rg)
    cm = CostModel(rg, tg)
    feat = []
    for i, k in enumerate(PER_METRE_KEYS):
        f = cm.F[:, i]
        feat.append({"feature": k, "source": FEATURE_STATUS[k][0], "status": FEATURE_STATUS[k][1],
                     "mean": f.mean(), "length_weighted_mean": float((f * rg.length).sum() / rg.length.sum()),
                     "p10": np.percentile(f, 10), "p90": np.percentile(f, 90), "share_nonzero": float((f > 0).mean())})
    pen = tg.t_pen
    feat.append({"feature": "junction", "source": FEATURE_STATUS["junction"][0],
                 "status": FEATURE_STATUS["junction"][1], "mean": pen.mean(), "length_weighted_mean": None,
                 "p10": np.percentile(pen, 10), "p90": np.percentile(pen, 90),
                 "share_nonzero": float((pen > 0).mean())})
    out = cfg.path("results_dir")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(feat).to_csv(out / "feature_summary.csv", index=False)
    summary = {"nodes": rg.n, "directed_edges": rg.m, "turn_states": tg.m, "transitions": tg.n_transitions,
               "scc_share": rg.meta["scc_share"], "network_km_both_directions": float(rg.length.sum() / 1000),
               "haversine_scale_s": cm.hav_scale, "stairs_edges": int(rg.steps.sum()),
               "elevation_relief_m": float(np.nanmax(rg.elevation) - np.nanmin(rg.elevation)),
               "bounds": rg.bounds_latlon(), "seconds": time.perf_counter() - t0}
    (cfg.path("processed_dir") / "graph_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def step_splits(cfg: Config, force: bool = False) -> dict:
    _banner("splits: weight vectors and label targets")
    from wayfinder.learning.splits import make_splits
    from wayfinder.graph.road_graph import RoadGraph

    rg = RoadGraph.load(processed_graph_path(cfg))
    s = make_splits(cfg, rg.n, rg.osmid, force=force)
    print(f"  weights: {len(s['weights']['train'])} train, {len(s['weights']['heldout'])} held-out; "
          f"targets: " + ", ".join(f"{k} {len(v)}" for k, v in s["targets"].items()))
    return s


def step_benchmark(cfg: Config, force: bool = False) -> pd.DataFrame:
    _banner("benchmark: freeze 200 queries")
    from wayfinder.evaluation.benchmark import make_benchmark
    from wayfinder.graph.road_graph import RoadGraph
    from wayfinder.learning.splits import check_splits_match_graph, load_splits

    rg = RoadGraph.load(processed_graph_path(cfg))
    splits = load_splits(cfg)
    check_splits_match_graph(splits, rg.osmid)
    df = make_benchmark(cfg, rg, splits, force=force)
    print(f"  {len(df)} queries ({(df.weight_set == 'heldout').sum()} with held-out weights), "
          f"median straight-line distance {df.euclid_m.median():.0f} m")
    return df


# ---------------------------------------------------------------------------------------------- learning
def step_labels(cfg: Config) -> dict:
    _banner("labels: backward Dijkstra sweep")
    from wayfinder.learning.labels import (DATASHEET, HELDOUT_FILE, TRAIN_FILE, generate_labels, save_labels)
    from wayfinder.learning.splits import check_splits_match_graph, load_splits

    net = load_network(cfg)
    splits = load_splits(cfg)
    check_splits_match_graph(splits, net.rg.osmid)
    lc = cfg["labels"]
    out = cfg.path("labels_dir")
    all_targets = np.concatenate([splits["targets"][k] for k in ("train", "val", "test")])
    eval_targets = np.concatenate([splits["targets"]["val"], splits["targets"]["test"]])
    meta = {"graph": net.rg.meta["source"], "weight_keys": list(WEIGHT_KEYS), "nodes_per_run": lc["nodes_per_run"]}

    print(f"  training weights: {len(splits['weights']['train'])} x {len(all_targets)} targets")
    tr = generate_labels(net, splits["weights"]["train"], all_targets, lc["nodes_per_run"], lc["seed"])
    save_labels(tr, out / TRAIN_FILE, {**meta, "weight_set": "train"})
    print(f"  held-out weights: {len(splits['weights']['heldout'])} x {len(eval_targets)} targets")
    ho = generate_labels(net, splits["weights"]["heldout"], eval_targets, lc["nodes_per_run"], lc["seed"] + 1)
    save_labels(ho, out / HELDOUT_FILE, {**meta, "weight_set": "heldout"})

    n_tr_runs = len(splits["weights"]["train"]) * len(all_targets)
    n_ho_runs = len(splits["weights"]["heldout"]) * len(eval_targets)
    (out / "DATASHEET.md").write_text(DATASHEET.format(
        n=net.rg.n, m=net.rg.m, keys=", ".join(WEIGHT_KEYS), train_file=TRAIN_FILE, heldout_file=HELDOUT_FILE,
        train_runs=n_tr_runs, train_samples=len(tr["node"]), heldout_runs=n_ho_runs,
        heldout_samples=len(ho["node"])), encoding="utf-8")
    summary = {"train_runs": n_tr_runs, "train_samples": int(len(tr["node"])), "train_seconds": float(tr["seconds"]),
               "heldout_runs": n_ho_runs, "heldout_samples": int(len(ho["node"])),
               "heldout_seconds": float(ho["seconds"]),
               "admissibility_violations": int(tr["admissibility_violations"] + ho["admissibility_violations"])}
    (out / "labels_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["admissibility_violations"]:
        raise RuntimeError("h_adm exceeded the exact cost-to-go: the admissible heuristic is broken")
    return summary


def step_node2vec(cfg: Config) -> dict:
    _banner("node2vec embeddings")
    from wayfinder.graph.road_graph import RoadGraph
    from wayfinder.learning.artifacts import EMBEDDINGS_META, embeddings_path
    from wayfinder.learning.node2vec import train_node2vec

    rg = RoadGraph.load(processed_graph_path(cfg))
    nc = cfg["node2vec"]
    emb, info = train_node2vec(rg.tail, rg.head, rg.n, dim=nc["dim"], walk_length=nc["walk_length"],
                               walks_per_node=nc["walks_per_node"], window=nc["window"], p=nc["p"], q=nc["q"],
                               negatives=nc["negatives"], epochs=nc["epochs"], batch_size=nc["batch_size"],
                               lr=nc["lr"], seed=nc["seed"])
    np.save(embeddings_path(cfg), emb)
    (cfg.path("processed_dir") / EMBEDDINGS_META).write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def step_train(cfg: Config, hparam: bool = False, seeds=None) -> pd.DataFrame:
    _banner("train the learned heuristic")
    import torch

    from wayfinder.graph.road_graph import RoadGraph
    from wayfinder.learning.artifacts import HPARAMS_FILE, TUNED_FILE, checkpoint_path, load_embeddings
    from wayfinder.learning.labels import HELDOUT_FILE, TRAIN_FILE, load_labels, split_mask
    from wayfinder.learning.model import FeatureBuilder
    from wayfinder.learning.splits import load_splits
    from wayfinder.learning.train import LabelTensors, hyperparameter_pass, label_metrics, train_one
    from wayfinder.search.heuristics import TunedClassicalHeuristic

    torch.set_num_threads(max(1, torch.get_num_threads()))
    rg = RoadGraph.load(processed_graph_path(cfg))
    splits = load_splits(cfg)
    targets = splits["targets"]
    labels = load_labels(cfg.path("labels_dir") / TRAIN_FILE)
    heldout = load_labels(cfg.path("labels_dir") / HELDOUT_FILE)
    fb = FeatureBuilder(load_embeddings(cfg, rg.n), rg.x, rg.y)
    models_dir = cfg.path("models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)
    tdir = cfg.path("results_dir") / "training"
    tdir.mkdir(parents=True, exist_ok=True)
    tc = cfg["training"]

    hp_path = models_dir / HPARAMS_FILE
    if hparam:
        best = hyperparameter_pass(fb, labels, targets, cfg, tdir / "hparam_search.csv")
        hp_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
        print(f"  chosen: {best}")
    if hp_path.exists():
        best = json.loads(hp_path.read_text(encoding="utf-8"))
    else:
        best = {"hidden": cfg["model"]["hidden"], "lr": tc["lr"]}

    # tuned classical fallback: log kappa(w) fitted on the training targets only
    m = split_mask(labels, targets["train"]) & (labels["node"] != labels["target"]) & (labels["h_adm"] > 0)
    tuned = TunedClassicalHeuristic.fit(labels["weights"][labels["widx"][m].astype(np.int64)],
                                        np.log(labels["h_star"][m] / labels["h_adm"][m]))
    (models_dir / TUNED_FILE).write_text(json.dumps(tuned.to_dict(), indent=2), encoding="utf-8")

    train = LabelTensors(labels, split_mask(labels, targets["train"]), tc["max_train_samples"], seed=0)
    val = LabelTensors(labels, split_mask(labels, targets["val"]))
    eval_sets = {
        "test targets, training weights": LabelTensors(labels, split_mask(labels, targets["test"])),
        "test targets, held-out weights": LabelTensors(heldout, split_mask(heldout, targets["test"])),
        "val targets, held-out weights": LabelTensors(heldout, split_mask(heldout, targets["val"])),
    }
    print(f"  samples: train {len(train)}, val {len(val)}, " +
          ", ".join(f"{k} {len(v)}" for k, v in eval_sets.items()))
    rows = []
    for seed in (seeds if seeds is not None else tc["seeds"]):
        print(f"  seed {seed}: hidden={best['hidden']} lr={best['lr']}", flush=True)
        net, log, _ = train_one(fb, train, val, best["hidden"], best["lr"], tc["epochs"], tc["batch_size"], seed,
                                tc["weight_decay"], tc["patience"], checkpoint_path(cfg, seed),
                                meta={"train_samples": len(train)})
        log.to_csv(tdir / f"train_log_seed{seed}.csv", index=False)
        for name, data in eval_sets.items():
            rows.append({"seed": seed, "set": name, **label_metrics(net, fb, data)})
    df = pd.DataFrame(rows)
    df.to_csv(tdir / "label_metrics.csv", index=False)
    print(df.groupby("set")[["mae_log", "overestimate_rate", "informedness_learned", "informedness_adm"]]
          .median().to_string())
    return df


# ---------------------------------------------------------------------------------------------- evaluation
def step_evaluate(cfg: Config, limit: int | None = None, skip_curve: bool = False) -> dict:
    _banner("evaluate")
    from wayfinder.evaluation.benchmark import load_benchmark
    from wayfinder.evaluation.experiments import problem_curve, summarise_problem_curve
    from wayfinder.evaluation.harness import build_conditions, run_benchmark
    from wayfinder.evaluation.quality import quality_analysis
    from wayfinder.learning.artifacts import load_models, load_tuned

    ev = cfg["evaluation"]
    net = load_network(cfg)
    queries = load_benchmark(cfg, net.rg)
    if limit:
        queries = queries.head(limit)
    models = load_models(cfg, net.rg)
    tuned = load_tuned(cfg)
    if not models:
        print("  warning: no trained models found; learned conditions are skipped")
    res = cfg.path("results_dir")
    res.mkdir(parents=True, exist_ok=True)

    print("  [1/4] benchmark conditions", flush=True)
    raw = run_benchmark(net, queries, build_conditions(net, cfg, models, tuned))
    raw.to_csv(res / "benchmark_raw.csv", index=False)

    print("  [2/4] heuristic quality", flush=True)
    per_q, samples = quality_analysis(net, queries, models, tuned, ev["quality_nodes_per_query"])
    per_q.to_csv(res / "heuristic_quality_per_query.csv", index=False)
    samples.to_csv(res / "heuristic_quality_samples.csv", index=False)

    if not skip_curve:
        print("  [3/4] problem-statement curve", flush=True)
        names = {"dijkstra", "astar_hadm", "astar_hstar", f"focal_learned_W{ev['headline_W']:g}_s{min(models)}"
                 if models else ""}
        curve_conds = [c for c in build_conditions(net, cfg, {min(models): models[min(models)]} if models else {},
                                                   None) if c.name in names]
        curve_raw = problem_curve(net, queries, ev["lambdas"], curve_conds)
        curve_raw.to_csv(res / "problem_curve_raw.csv", index=False)
        summarise_problem_curve(curve_raw).to_csv(res / "problem_curve.csv", index=False)

    print("  [4/4] derived tables, plots and report", flush=True)
    return step_report(cfg)


def step_report(cfg: Config) -> dict:
    """Rebuild every derived table, figure and results/summary.md from the raw result files."""
    from wayfinder.evaluation import plots
    from wayfinder.evaluation.experiments import bound_sweep, generalisation_search, go_no_go, write_json
    from wayfinder.evaluation.harness import summarise
    from wayfinder.evaluation.quality import summarise_quality
    from wayfinder.evaluation.report import write_summary

    ev = cfg["evaluation"]
    res = cfg.path("results_dir")
    pl = res / "plots"
    raw = pd.read_csv(res / "benchmark_raw.csv")
    summary = summarise(raw)
    summary.to_csv(res / "benchmark_summary.csv", index=False)
    bound = bound_sweep(raw)
    bound.to_csv(res / "bound_sweep.csv", index=False)
    gen = generalisation_search(raw)
    gen.to_csv(res / "generalisation.csv", index=False)
    verdict = go_no_go(summary, ev["headline_W"])
    write_json(verdict, res / "go_no_go.json")
    print(json.dumps(verdict, indent=2, default=float))

    per_q = pd.read_csv(res / "heuristic_quality_per_query.csv")
    summarise_quality(per_q).to_csv(res / "heuristic_quality.csv", index=False)
    samples_path = res / "heuristic_quality_samples.csv"
    if samples_path.exists():
        plots.plot_informedness(pd.read_csv(samples_path), pl / "informedness_hist.png")

    curve_path = res / "problem_curve.csv"
    if curve_path.exists():
        plots.plot_problem_curve(pd.read_csv(curve_path), pl / "problem_curve.png")
    plots.plot_bound(bound, pl / "bound_guaranteed_vs_achieved.png")
    plots.plot_expansions_vs_W(bound, summary, pl / "expansions_vs_W.png")
    hw = f"{ev['headline_W']:g}"
    plots.plot_generalisation(gen, ["astar_hadm", f"wastar_hadm_W{hw}", f"focal_tuned_W{hw}",
                                    f"focal_learned_W{hw}", f"focal_hstar_W{hw}", "astar_learned"],
                              pl / "generalisation.png")
    plots.plot_latency(raw, ["dijkstra", "astar_hadm", f"wastar_hadm_W{hw}", f"focal_tuned_W{hw}",
                             f"focal_learned_W{hw}", "astar_learned"], pl / "latency.png")
    logs = {}
    for p in sorted((res / "training").glob("train_log_seed*.csv")):
        logs[int(p.stem.replace("train_log_seed", ""))] = pd.read_csv(p)
    if logs:
        plots.plot_training_curves(logs, pl / "training_curves.png")
    out = write_summary(res)
    print(f"  report: {out}")
    return verdict


# ---------------------------------------------------------------------------------------------- study
def step_study_prepare(cfg: Config) -> list:
    _banner("user study: prepare trials")
    from wayfinder.study.user_study import prepare_study

    net = load_network(cfg)
    trials = prepare_study(net, cfg, cfg.path("study_dir"))
    print(f"  {len(trials)} trials written to {cfg.path('study_dir')}")
    return trials


def step_study_analyze(cfg: Config) -> pd.DataFrame:
    _banner("user study: analyse responses")
    from wayfinder.evaluation.report import write_summary
    from wayfinder.study.user_study import analyze_responses

    df = analyze_responses(cfg.path("results_dir") / "user_study")
    print(df.to_string(index=False))
    write_summary(cfg.path("results_dir"))
    return df
