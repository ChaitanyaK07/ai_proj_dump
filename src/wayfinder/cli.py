"""Command line: ``python -m wayfinder <step> [options]``."""
from __future__ import annotations

import argparse

from wayfinder.config import load_config


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="wayfinder", description="Routes That Fit You — pipeline and service")
    ap.add_argument("--config", help="YAML file overriding configs/default.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preprocess", help="graphml -> data/processed (arrays, features, turn graph)")
    p = sub.add_parser("splits", help="freeze weight vectors and label targets")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("benchmark", help="freeze the 200-query benchmark")
    p.add_argument("--force", action="store_true")
    sub.add_parser("labels", help="backward-Dijkstra label sweep")
    sub.add_parser("node2vec", help="train node2vec embeddings")
    p = sub.add_parser("train", help="train the learned heuristic (3 seeds)")
    p.add_argument("--hparam", action="store_true", help="run the hyperparameter pass first")
    p.add_argument("--seeds", type=int, nargs="+")
    p = sub.add_parser("evaluate", help="run every experiment, plots and results/summary.md")
    p.add_argument("--limit", type=int, help="only the first N benchmark queries (smoke test)")
    p.add_argument("--skip-curve", action="store_true")
    sub.add_parser("report", help="rebuild derived tables, figures and summary.md from the raw result files")
    sub.add_parser("study-prepare", help="choose the 12 user-study OD pairs")
    sub.add_parser("study-analyze", help="agreement rate per profile from recorded responses")
    p = sub.add_parser("serve", help="FastAPI service + web UI")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p = sub.add_parser("all", help="preprocess -> splits -> benchmark -> labels -> node2vec -> train -> evaluate")
    p.add_argument("--force", action="store_true", help="regenerate frozen splits and benchmark too")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    from wayfinder import pipeline as P

    if args.cmd == "preprocess":
        P.step_preprocess(cfg)
    elif args.cmd == "splits":
        P.step_splits(cfg, args.force)
    elif args.cmd == "benchmark":
        P.step_benchmark(cfg, args.force)
    elif args.cmd == "labels":
        P.step_labels(cfg)
    elif args.cmd == "node2vec":
        P.step_node2vec(cfg)
    elif args.cmd == "train":
        P.step_train(cfg, args.hparam, args.seeds)
    elif args.cmd == "evaluate":
        P.step_evaluate(cfg, args.limit, args.skip_curve)
    elif args.cmd == "report":
        P.step_report(cfg)
    elif args.cmd == "study-prepare":
        P.step_study_prepare(cfg)
    elif args.cmd == "study-analyze":
        P.step_study_analyze(cfg)
    elif args.cmd == "serve":
        from wayfinder.service.api import serve
        serve(cfg, args.host, args.port)
    elif args.cmd == "all":
        P.step_preprocess(cfg)
        splits_dir, bench = cfg.path("splits_dir"), cfg.path("benchmark_dir") / "queries.csv"
        if args.force or not (splits_dir / "weights.json").exists():
            P.step_splits(cfg, force=args.force)
        if args.force or not bench.exists():
            P.step_benchmark(cfg, force=args.force)
        P.step_labels(cfg)
        P.step_node2vec(cfg)
        P.step_train(cfg, hparam=True)
        P.step_evaluate(cfg)
        P.step_study_prepare(cfg)


if __name__ == "__main__":
    main()
