import json

import numpy as np
import pandas as pd
import pytest

from wayfinder.evaluation.benchmark import load_benchmark, make_benchmark
from wayfinder.evaluation.experiments import bound_sweep, go_no_go
from wayfinder.evaluation.harness import build_conditions, run_benchmark, summarise
from wayfinder.evaluation.quality import quality_analysis, summarise_quality
from wayfinder.graph.road_graph import RoadGraph
from wayfinder.learning.splits import load_splits, make_splits
from wayfinder.service.planner import RoutePlanner
from wayfinder.study.user_study import analyze_responses, prepare_study, record_response


def test_road_graph_save_load_roundtrip(grid, tmp_path):
    p = tmp_path / "g.npz"
    grid.save(p)
    g2 = RoadGraph.load(p)
    assert g2.n == grid.n and g2.m == grid.m
    assert np.array_equal(g2.tail, grid.tail) and np.array_equal(g2.highway, grid.highway)
    assert np.array_equal(g2.out_idx, grid.out_idx)
    assert g2.meta == grid.meta


def test_graphml_loader(tmp_path):
    """A tiny OSMnx-style GraphML: geometry orientation, SCC filtering, list-valued tags."""
    import networkx as nx
    G = nx.MultiDiGraph(crs="EPSG:32644")
    pts = {"1": (225900.0, 1928000.0), "2": (226000.0, 1928000.0), "3": (226000.0, 1928100.0), "9": (0.0, 0.0)}
    for n, (x, y) in pts.items():
        G.add_node(n, x=str(x), y=str(y), elevation="500")
    G.add_edge("1", "2", length="100.5", highway="residential", grade="0.01", canopy="0.2",
               geometry="LINESTRING (226000 1928000, 225950 1928001, 225900 1928000)")   # stored reversed
    G.add_edge("2", "1", length="100.5", highway="residential", grade="-0.01", canopy="0.2")
    G.add_edge("2", "3", length="100.2", highway="['steps', 'footway']", grade="0.2", canopy="0", name="['A', 'B']")
    G.add_edge("3", "2", length="100.2", highway="steps", grade="-0.2", canopy="0")
    G.add_edge("9", "1", length="5", highway="service", grade="0", canopy="0")       # not in the SCC
    p = tmp_path / "t.graphml"
    nx.write_graphml(G, p)
    rg = RoadGraph.from_graphml(p)
    assert rg.n == 3 and rg.m == 4
    e12 = [e for e in range(rg.m) if rg.osmid[rg.tail[e]] == "1" and rg.osmid[rg.head[e]] == "2"][0]
    geom = rg.edge_geometry(e12)
    assert tuple(geom[0]) == pts["1"]                        # oriented tail -> head
    assert rg.b_start[e12] == pytest.approx(90.0, abs=2)
    e23 = [e for e in range(rg.m) if rg.osmid[rg.tail[e]] == "2" and rg.osmid[rg.head[e]] == "3"][0]
    assert rg.steps[e23] and rg.name[e23] == "A" and rg.highway[e23] == "steps|footway"
    assert rg.meta["scc_share"] == pytest.approx(0.75)


def test_splits_and_frozen_benchmark(grid, tmp_cfg):
    cfg = tmp_cfg
    cfg["splits"].update({"n_label_targets": 20, "n_train_weights": 6, "n_heldout_weights": 2})
    cfg["benchmark"].update({"n_queries": 12, "min_euclid_m": 150})
    make_splits(cfg, grid.n, grid.osmid)
    with pytest.raises(FileExistsError):
        make_splits(cfg, grid.n, grid.osmid)
    splits = load_splits(cfg)
    assert sum(len(v) for v in splits["targets"].values()) == 20
    df = make_benchmark(cfg, grid, splits)
    with pytest.raises(FileExistsError):
        make_benchmark(cfg, grid, splits)
    excluded = set(np.concatenate(list(splits["targets"].values())).tolist())
    assert not set(df["target"]) & excluded
    assert (df["weight_set"] == "heldout").sum() == 6
    assert load_benchmark(cfg, grid).equals(df)
    q = cfg.path("benchmark_dir") / "queries.csv"
    q.write_text(q.read_text().replace("train", "trian", 1))
    with pytest.raises(ValueError):
        load_benchmark(cfg)


def test_harness_quality_and_gate(net):
    rng = np.random.default_rng(3)
    rows = []
    for qid in range(6):
        s, t = rng.choice(net.rg.n, size=2, replace=False)
        w = rng.uniform(size=5)
        rows.append({"qid": qid, "source": s, "target": t, "weight_set": "train" if qid % 2 else "heldout",
                     **{f"w_{k}": v for k, v in zip(("time", "junction", "traffic", "canopy", "climb"), w)}})
    queries = pd.DataFrame(rows)
    from wayfinder.config import load_config
    from wayfinder.search import TunedClassicalHeuristic
    cfg = load_config()
    tuned = TunedClassicalHeuristic(np.array([0.5, 0, 0, 0, 0, 0]))
    raw = run_benchmark(net, queries, build_conditions(net, cfg, {}, tuned), verbose=False)
    assert raw["found"].all()
    assert (raw[raw["family"].isin(["dijkstra", "astar_hadm", "astar_hstar"])]["suboptimality"]
            .sub(1).abs() < 1e-9).all()
    assert raw["bound_ok"].dropna().astype(bool).all()
    summary = summarise(raw)
    assert set(summary["group"]) >= {"dijkstra", "astar_hadm", "focal_tuned_W1.5"}
    assert bound_sweep(raw)["bound_violations"].sum() == 0
    verdict = go_no_go(summary, 1.5)
    assert verdict["verdict"] == "NO-GO"                  # no learned model in this run
    per_q, samples = quality_analysis(net, queries, {}, tuned, nodes_per_query=30)
    q = summarise_quality(per_q)
    adm = q[(q["heuristic"] == "h_adm") & (q["weight_set"] == "all")].iloc[0]
    assert adm["admissibility_violation_rate"] == 0 and adm["consistency_violation_rate"] == 0
    assert adm["informedness_mean"] <= 1


def test_planner_routes_and_rejects_far_points(net):
    p = RoutePlanner(net)
    rg = net.rg
    o, d = (rg.lat[0], rg.lon[0]), (rg.lat[rg.n - 1], rg.lon[rg.n - 1])
    for algo in ("dijkstra", "astar", "wastar", "focal"):
        out = p.route(o, d, {"time": 0.3, "canopy": 0.7}, algo, 1.5, compare_fastest=True, verify=True)
        assert out["route"]["found"] and len(out["route"]["latlon"]) >= 2
        assert out["optimal"]["suboptimality"] <= out["search"]["W"] + 1e-9
        json.dumps(out)                                    # serialisable
    with pytest.raises(ValueError):
        p.route((17.0, 78.0), d, {"time": 1})


def test_api(net, tmp_cfg):
    from fastapi.testclient import TestClient
    from wayfinder.service.api import create_app

    planner = RoutePlanner(net)
    client = TestClient(create_app(planner, tmp_cfg))
    assert client.get("/api/health").json()["status"] == "ok"
    meta = client.get("/api/meta").json()
    assert meta["sliders"] == ["time", "junction", "traffic", "canopy", "climb"]
    rg = net.rg
    body = {"origin": {"lat": rg.lat[0], "lon": rg.lon[0]}, "destination": {"lat": rg.lat[40], "lon": rg.lon[40]},
            "weights": {"time": 0.5, "traffic": 1.0}, "algorithm": "focal", "W": 1.25}
    r = client.post("/api/route", json=body)
    assert r.status_code == 200 and r.json()["search"]["W"] == 1.25
    assert client.post("/api/route", json={**body, "algorithm": "bfs"}).status_code == 400
    assert client.post("/api/route", json={**body, "W": 0.5}).status_code == 422
    assert client.post("/api/route", json={**body, "origin": {"lat": 10, "lon": 70}}).status_code == 400
    assert client.get("/api/study/trials").status_code == 404
    assert client.get("/").status_code == 200 and client.get("/static/app.js").status_code == 200


def test_user_study_prepare_record_analyze(net, tmp_cfg):
    cfg = tmp_cfg
    cfg["user_study"].update({"n_pairs": 3, "radius_m": 2000, "min_euclid_m": 200, "max_overlap": 0.95,
                              "max_detour_pct": 80, "min_gain_pct": 0.1})
    trials = prepare_study(net, cfg, cfg.path("study_dir"), max_tries=500)
    assert trials, "fixture should yield at least one usable pair"
    from fastapi.testclient import TestClient
    from wayfinder.service.api import create_app
    client = TestClient(create_app(RoutePlanner(net), cfg))
    served = client.get("/api/study/trials").json()["trials"]
    assert {t["trial_id"] for t in served} == {t["trial_id"] for t in trials}
    assert "metrics" not in served[0]["routes"]["profile"]           # nothing that reveals which is which
    tid = served[0]["trial_id"]
    for pid, choice in (("a", "profile"), ("b", "fastest"), ("a", "profile")):
        r = client.post("/api/study/response", json={"participant_id": pid, "trial_id": tid, "choice": choice,
                                                     "shown_first": "fastest"})
        assert r.status_code == 200
    assert client.post("/api/study/response", json={"participant_id": "a", "trial_id": "nope", "choice": "profile",
                                                    "shown_first": "fastest"}).status_code == 404
    out = analyze_responses(cfg.path("results_dir") / "user_study")
    row = out[out["profile"] == "all"].iloc[0]
    assert row["participants"] == 2 and row["choices"] == 2 and row["agreement_rate"] == 0.5
    record_response(cfg.path("results_dir") / "user_study", {"participant_id": "c", "trial_id": tid,
                                                              "choice": "profile", "profile": "shade"})


def test_default_route_is_the_exact_best_route(net):
    """What the app returns by default must be the optimum for exactly the slider values given."""
    from wayfinder.search import SearchSpace, dijkstra
    p = RoutePlanner(net)
    rg = net.rg
    rng = np.random.default_rng(7)
    for _ in range(25):
        s, t = (int(v) for v in rng.choice(rg.n, size=2, replace=False))
        w = rng.uniform(0, 1, size=5) * (rng.uniform(size=5) > 0.3)          # arbitrary, non-round sliders
        out = p.route((rg.lat[s], rg.lon[s]), (rg.lat[t], rg.lon[t]),
                      dict(zip(("time", "junction", "traffic", "canopy", "climb"), w)), compare_fastest=False)
        exact = dijkstra(SearchSpace.build(net.cm, w), s, t).cost
        assert out["search"]["algorithm"] == "astar" and out["search"]["W"] == 1.0
        assert out["route"]["cost"] == pytest.approx(exact, rel=1e-12)


def test_raising_a_slider_never_increases_what_it_penalises(net):
    """Exact optimiser => raising w_k cannot increase the route's total of f_k (or of junction load)."""
    p = RoutePlanner(net)
    rg, cm, tg = net.rg, net.cm, net.tg
    rng = np.random.default_rng(11)

    def amount(path, i):
        path = np.asarray(path)
        if i == 1:
            return float(sum(tg.t_pen[tg.transition_index(int(a), int(b))] for a, b in zip(path[:-1], path[1:])))
        col = {0: 0, 2: 1, 3: 2, 4: 3}[i]
        return float((rg.length[path] * cm.F[path, col]).sum())

    for _ in range(20):
        s, t = (int(v) for v in rng.choice(rg.n, size=2, replace=False))
        w = rng.uniform(0, 1, size=5)
        for i in range(5):
            lo, hi = w.copy(), w.copy()
            lo[i], hi[i] = 0.0, 1.0
            assert amount(p.search(s, t, hi)[0].path, i) <= amount(p.search(s, t, lo)[0].path, i) * (1 + 1e-9) + 1e-9


def test_planner_is_correct_under_concurrent_requests(net):
    """The web service answers from a thread pool; concurrent routes must all stay exact."""
    from concurrent.futures import ThreadPoolExecutor

    from wayfinder.search import SearchSpace, dijkstra
    p = RoutePlanner(net, cache_size=4)                     # small cache: forces concurrent evictions
    rg = net.rg
    rng = np.random.default_rng(3)
    jobs = []
    for _ in range(60):
        s, t = (int(v) for v in rng.choice(rg.n, size=2, replace=False))
        jobs.append((s, t, np.round(rng.uniform(0, 1, size=5), 2)))

    def run(job):
        s, t, w = job
        return p.search(s, t, w)[0].cost, dijkstra(SearchSpace.build(net.cm, w), s, t).cost

    with ThreadPoolExecutor(max_workers=8) as ex:
        for got, exact in ex.map(run, jobs):
            assert got == pytest.approx(exact, rel=1e-12)
    assert len(p._spaces) <= 4
