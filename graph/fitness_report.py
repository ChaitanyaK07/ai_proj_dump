"""
Dataset fitness report for the preference-aware pedestrian routing project.

    python fitness_report.py data/jubilee_2km.graphml data/miyapur_2km.graphml

Give one or more graphs built by build_dataset.py. For each graph it runs:

  1. Integrity    connectivity, junction mix, bad edges, size of the turn-aware state space
  2. Features     spread of every cost feature, grade noise, turn mix, rho_min risk
  3. Sliders      does each of the 5 sliders actually change the route?  (turn-aware search)
  4. Search gap   Dijkstra vs A*(h_adm) vs A*(h*) node expansions as preference weight grows
                  -> the project's problem statement: is there room for a learned heuristic?
  5. Labels       time for one full backward sweep, extrapolated to the 8,000-run label budget
  6. User study   OD pairs where a profile route clearly differs from the fastest route

Writes to fitness/:
    <name>_verdict.csv  <name>_sliders.csv  <name>_search_curve.csv  <name>_search_curve.png
    <name>_study_candidates.csv   and comparison.csv when several graphs are given.

The cost model below follows the proposal (A.2/A.3) with two fixes: walking time is
normalised by its max (never 0) and a BASE term keeps rho_min > 0. Edit to match interfaces.py.
"""
import ast
import heapq
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import osmnx as ox
import pandas as pd

# ------------------------------------------------------------------ cost model
BASE = 0.1                  # cost per metre floor, keeps rho_min > 0
GRADE_CAP = 0.15            # grades above this are clipped (30 m DEM noise)
TURN_M = 30.0               # a penalty-1.0 turn costs as much as 30 m of walking, at w_turn = 1
SPEED = {"footway": 1.3, "path": 1.3, "pedestrian": 1.3, "cycleway": 1.3, "track": 1.2,
         "living_street": 1.25, "residential": 1.2, "service": 1.2, "unclassified": 1.2,
         "tertiary": 1.15, "secondary": 1.1, "primary": 1.1, "trunk": 1.0, "steps": 0.5}
TRAFFIC = {"trunk": 1.0, "primary": 1.0, "secondary": 0.8, "tertiary": 0.6, "unclassified": 0.4,
           "residential": 0.3, "living_street": 0.2, "service": 0.2, "track": 0.1,
           "footway": 0.0, "path": 0.0, "pedestrian": 0.0, "cycleway": 0.0, "steps": 0.0}
FEATS = ["time", "traffic", "canopy", "climb"]       # per-metre features; 'turn' is the 5th slider

# ------------------------------------------------------------------ test settings
SEED = 0
N_PAIRS = 150               # OD pairs for the slider and user-study tests
N_SEARCH = 40               # OD pairs for the search-gap test
MIN_EUCLID = 700            # metres between origin and destination
LAMBDAS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
LABEL_RUNS = 8000           # proposal A.5

T = dict(scc=0.95, overlap=0.80, gain=10.0, detour=20.0, relief=30.0,
         curve_rise=0.15, headroom=0.5, candidates=4, label_hours=6.0)

INF = float("inf")


# ================================================================== loading
def classes(h):
    if isinstance(h, str) and h.startswith("["):
        h = ast.literal_eval(h)
    return [c.replace("_link", "") for c in (h if isinstance(h, list) else [h])]


def bearing(p, q):
    dx, dy = q[0] - p[0], q[1] - p[1]
    return math.degrees(math.atan2(dx, dy)) % 360 if (dx or dy) else None


def turn_penalty(b_in, b_out):
    if b_in is None or b_out is None:
        return 0.0
    d = abs(((b_out - b_in + 180) % 360) - 180)
    return 0.0 if d < 30 else 0.3 if d < 75 else 1.0 if d < 150 else 3.0


class Net:
    """Compact arrays over the largest strongly connected component."""

    def __init__(self, path):
        G = ox.load_graphml(path,
                            node_dtypes={"x": float, "y": float, "elevation": float,
                                         "lat": float, "lon": float},
                            edge_dtypes={"length": float, "canopy": float, "grade": float})
        self.name = Path(path).stem
        self.G_full = G
        import networkx as nx
        scc = max(nx.strongly_connected_components(G), key=len)
        self.scc_share = len(scc) / G.number_of_nodes()
        G = G.subgraph(scc).copy()
        self.G = G

        self.nodes = list(G.nodes)
        idx = {n: i for i, n in enumerate(self.nodes)}
        self.n = len(self.nodes)
        nd = G.nodes
        self.x = np.array([nd[n]["x"] for n in self.nodes])
        self.y = np.array([nd[n]["y"] for n in self.nodes])
        self.lat = [nd[n].get("lat", nd[n]["y"]) for n in self.nodes]
        self.lon = [nd[n].get("lon", nd[n]["x"]) for n in self.nodes]
        self.elev = np.array([nd[n].get("elevation", np.nan) for n in self.nodes], dtype=float)

        tail, head, L, cls, steps, canopy, grade, b0, b1 = [], [], [], [], [], [], [], [], []
        for u, v, d in G.edges(data=True):
            iu, iv = idx[u], idx[v]
            cl = classes(d.get("highway", "residential"))
            g = d.get("grade", 0.0)
            g = 0.0 if g is None or g != g else float(g)
            pu, pv = (nd[u]["x"], nd[u]["y"]), (nd[v]["x"], nd[v]["y"])
            geom = d.get("geometry")
            pts = list(geom.coords) if geom is not None else [pu, pv]
            if math.dist(pts[0], pu) > math.dist(pts[-1], pu):
                pts = pts[::-1]
            bs = next((bearing(pts[i], pts[i + 1]) for i in range(len(pts) - 1)
                       if bearing(pts[i], pts[i + 1]) is not None), None)
            be = next((bearing(pts[i - 1], pts[i]) for i in range(len(pts) - 1, 0, -1)
                       if bearing(pts[i - 1], pts[i]) is not None), None)
            tail.append(iu); head.append(iv); L.append(float(d["length"]))
            cls.append(cl[0]); steps.append("steps" in cl)
            canopy.append(float(d.get("canopy", 0.0) or 0.0)); grade.append(g)
            b0.append(bs); b1.append(be)
        self.tail, self.head = tail, head
        self.L = np.array(L)
        self.cls, self.steps = cls, np.array(steps)
        self.canopy, self.grade = np.array(canopy), np.array(grade)
        self.b_start, self.b_end = b0, b1
        self.m = len(L)

        inv_speed = np.array([1 / min(SPEED.get(c, 1.2) for c in classes(d.get("highway", "residential")))
                              for _, _, d in G.edges(data=True)])
        f_time = inv_speed / inv_speed.max()
        f_traffic = np.array([max(TRAFFIC.get(c, 0.3) for c in classes(d.get("highway", "residential")))
                              for _, _, d in G.edges(data=True)])
        f_canopy = 1.0 - np.clip(self.canopy, 0, 1)
        f_climb = np.where(self.steps, 1.0, np.clip(self.grade, 0, GRADE_CAP) / GRADE_CAP)
        self.F = np.vstack([f_time, f_traffic, f_canopy, f_climb]).T       # m x 4

        self.out = [[] for _ in range(self.n)]
        self.rin = [[] for _ in range(self.n)]
        for e in range(self.m):
            self.out[tail[e]].append(e)
            self.rin[head[e]].append(e)

    # ------------------------------------------------------------- costs
    def rho(self, w):
        """Per-metre cost of every edge. w: dict over FEATS (+ 'turn' ignored here)."""
        wv = np.array([w.get(k, 0.0) for k in FEATS])
        return BASE + self.F @ wv

    def edge_cost(self, w):
        return (self.L * self.rho(w)).tolist()

    # ------------------------------------------------------------- node-based search
    def astar(self, s, t, c, h):
        g = {s: 0.0}
        closed = set()
        heap = [(h[s], -0.0, s)]
        exp = 0
        out, head = self.out, self.head
        while heap:
            _, ng_neg, u = heapq.heappop(heap)
            gs = -ng_neg
            if u in closed:
                continue
            closed.add(u)
            exp += 1
            if u == t:
                return gs, exp
            for e in out[u]:
                v = head[e]
                ng = gs + c[e]
                if ng < g.get(v, INF):
                    g[v] = ng
                    heapq.heappush(heap, (ng + h[v], -ng, v))     # ties: prefer deeper nodes
        return INF, exp

    def cost_to_go(self, t, c):
        dist = [INF] * self.n
        dist[t] = 0.0
        heap = [(0.0, t)]
        rin, tail = self.rin, self.tail
        while heap:
            d, v = heapq.heappop(heap)
            if d > dist[v]:
                continue
            for e in rin[v]:
                u = tail[e]
                nd_ = d + c[e]
                if nd_ < dist[u]:
                    dist[u] = nd_
                    heapq.heappush(heap, (nd_, u))
        return dist

    # ------------------------------------------------------------- turn-aware search
    def turn_search(self, s, t, c, w_turn):
        """State = edge arrived on. Returns (cost, edge list). t=None -> full sweep, returns states settled."""
        dist, prev, heap = {}, {}, []
        for e in self.out[s]:
            dist[e] = c[e]
            prev[e] = None
            heapq.heappush(heap, (c[e], e))
        k = w_turn * TURN_M
        out, head, b_end, b_start = self.out, self.head, self.b_end, self.b_start
        settled = 0
        while heap:
            d, e = heapq.heappop(heap)
            if d > dist[e]:
                continue
            settled += 1
            v = head[e]
            if t is not None and v == t:
                path = []
                while e is not None:
                    path.append(e)
                    e = prev[e]
                return d, path[::-1]
            be = b_end[e]
            for e2 in out[v]:
                nd_ = d + c[e2] + (k * turn_penalty(be, b_start[e2]) if k else 0.0)
                if nd_ < dist.get(e2, INF):
                    dist[e2] = nd_
                    prev[e2] = e
                    heapq.heappush(heap, (nd_, e2))
        return (INF, []) if t is not None else settled

    def route_metrics(self, path):
        L = self.L[path].sum()
        expo = {k: float((self.L[path] * self.F[path, i]).sum() / L) for i, k in enumerate(FEATS)}
        expo["shade"] = float((self.L[path] * self.canopy[path]).sum() / L)
        pen = sum(turn_penalty(self.b_end[a], self.b_start[b]) for a, b in zip(path[:-1], path[1:]))
        expo["turn"] = pen / (L / 1000)                     # turn penalty per km
        return float(L), expo


# ================================================================== tests
def verdict(value, good, weak=None, higher=True):
    ok = value >= good if higher else value <= good
    if ok:
        return "PASS"
    if weak is not None and (value >= weak if higher else value <= weak):
        return "WEAK"
    return "FAIL"


def test_integrity(net, V):
    G = net.G_full
    und_deg = np.array([len(set(G.predecessors(n)) | set(G.successors(n))) for n in net.nodes])
    area_km2 = (net.x.max() - net.x.min()) * (net.y.max() - net.y.min()) / 1e6
    print("\n[1] INTEGRITY")
    print(f"  nodes {G.number_of_nodes()}, directed edges {G.number_of_edges()}, "
          f"area {area_km2:.1f} km2")
    print(f"  largest strongly connected component: {100 * net.scc_share:.1f}% of nodes")
    print(f"  turn-aware states (= directed edges in SCC): {net.m}")
    print(f"  junctions (3+ neighbours): {100 * (und_deg >= 3).mean():.1f}% of nodes, "
          f"{(und_deg >= 3).sum() / area_km2:.0f} per km2; dead ends: {100 * (und_deg == 1).mean():.1f}%")
    print(f"  self-loops: {sum(1 for e in range(net.m) if net.tail[e] == net.head[e])}, "
          f"edges < 1 m: {(net.L < 1).sum()}, median edge length {np.median(net.L):.0f} m")
    V.append(("connectivity (SCC share)", round(net.scc_share, 3), f">= {T['scc']}",
              verdict(net.scc_share, T["scc"], 0.9)))


def test_features(net, V):
    print("\n[2] FEATURES (per edge, after normalisation; climb clipped at "
          f"{int(GRADE_CAP * 100)}%)")
    rows = []
    for i, k in enumerate(FEATS):
        f = net.F[:, i]
        rows.append((k, f.mean(), f.std(), 100 * (f == f.min()).mean(), np.percentile(f, 10),
                     np.percentile(f, 90)))
    print(pd.DataFrame(rows, columns=["feature", "mean", "std", "% at min", "p10", "p90"])
          .round(3).to_string(index=False))

    share = pd.Series(net.L).groupby(pd.Series(net.cls)).sum().sort_values(ascending=False)
    print("  highway class share of length (%): " +
          ", ".join(f"{k} {v:.1f}" for k, v in (100 * share / share.sum()).head(8).items()))

    relief = np.nanmax(net.elev) - np.nanmin(net.elev) if np.isfinite(net.elev).any() else 0.0
    ag = np.abs(net.grade)
    noisy = ag > GRADE_CAP
    print(f"  relief {relief:.0f} m; |grade| p50 {np.percentile(ag, 50):.3f}, p90 {np.percentile(ag, 90):.3f}, "
          f"p99 {np.percentile(ag, 99):.3f}; > cap: {noisy.sum()} edges "
          f"({(noisy & (net.L < 30)).sum()} shorter than 30 m); steps edges: {int(net.steps.sum())}")
    print(f"  canopy: {100 * (net.canopy > 0).mean():.1f}% of edges > 0, "
          f"{100 * (net.canopy >= 0.25).mean():.1f}% >= 0.25")

    # turn mix over all transitions
    pens = [turn_penalty(net.b_end[e], net.b_start[e2]) for e in range(net.m) for e2 in net.out[net.head[e]]]
    pens = np.array(pens)
    print(f"  turn mix over {len(pens)} transitions: straight {100 * (pens == 0).mean():.0f}%, "
          f"slight {100 * (pens == 0.3).mean():.0f}%, turn {100 * (pens == 1).mean():.0f}%, "
          f"sharp/u-turn {100 * (pens == 3).mean():.0f}%")

    # rho_min risk under the proposal's pure min-max normalisation (no BASE)
    F = net.F
    mm = (F - F.min(0)) / np.where(F.max(0) > F.min(0), F.max(0) - F.min(0), 1)
    all_zero = int((mm.max(1) == 0).sum())
    rng = np.random.default_rng(SEED)
    rmins = [(mm @ rng.dirichlet(np.ones(4))).min() for _ in range(50)]
    print(f"  rho_min check, pure min-max (proposal A.2): {all_zero} edges zero on every feature; "
          f"smallest rho_min over 50 Dirichlet weights = {min(rmins):.4f}"
          f"{'  -> h_adm collapses to 0, keep BASE' if min(rmins) < 1e-3 else ''}")
    print(f"  with BASE = {BASE}: rho_min >= {BASE} for every weight vector")

    V.append(("terrain relief (m)", round(float(relief), 1), f">= {T['relief']}",
              verdict(relief, T["relief"], 15)))


def sample_pairs(net, k, seed):
    rng = random.Random(seed)
    pairs = []
    while len(pairs) < k:
        a, b = rng.sample(range(net.n), 2)
        if math.hypot(net.x[a] - net.x[b], net.y[a] - net.y[b]) >= MIN_EUCLID:
            pairs.append((a, b))
    return pairs


def test_sliders(net, V, out_dir):
    print("\n[3] SLIDERS (fastest route vs preference route, turn-aware search)")
    pairs = sample_pairs(net, N_PAIRS, SEED)
    base_c = net.edge_cost({"time": 1.0})
    base = [net.turn_search(s, t, base_c, 0.0)[1] for s, t in pairs]
    base_m = [net.route_metrics(p) for p in base]

    # metric per slider: (exposure key, lower-is-better?)
    sliders = {"traffic": ("traffic", True), "canopy": ("shade", False),
               "climb": ("climb", True), "turn": ("turn", True)}
    rows, cand = [], []
    for name, (key, lower) in sliders.items():
        for strength in (0.5, 0.8):
            w = {"time": 1 - strength}
            w_turn = 0.0
            if name == "turn":
                w_turn = strength
            else:
                w[name] = strength
            c = net.edge_cost(w)
            ov, gains, det, changed = [], [], [], 0
            for (s, t), p0, (L0, x0) in zip(pairs, base, base_m):
                _, p1 = net.turn_search(s, t, c, w_turn)
                L1, x1 = net.route_metrics(p1)
                shared = net.L[list(set(p0) & set(p1))].sum()
                overlap = shared / L0
                a, b = x0[key], x1[key]
                gain = 100 * ((a - b) / a if lower else (b - a) / a) if a > 1e-9 else 0.0
                detour = 100 * (L1 / L0 - 1)
                ov.append(overlap); gains.append(gain); det.append(detour)
                changed += overlap < 0.95
                if strength == 0.8 and overlap <= 0.5 and detour <= 15 and gain >= 20:
                    cand.append((name, net.nodes[s], net.nodes[t], net.lat[s], net.lon[s],
                                 net.lat[t], net.lon[t], round(L0), round(overlap, 2),
                                 round(detour, 1), round(gain, 1)))
            rows.append((name, strength, changed, np.median(ov), np.mean(gains), np.mean(det)))

    df = pd.DataFrame(rows, columns=["slider", "weight", f"routes changed /{N_PAIRS}",
                                     "median overlap", "mean gain %", "mean detour %"])
    print(df.round(2).to_string(index=False))
    print("  gain = drop in traffic / climb / turns per km, or RISE in canopy along the route")
    df.to_csv(out_dir / f"{net.name}_sliders.csv", index=False)

    strong = df[df["weight"] == 0.8].set_index("slider")
    for name in sliders:
        r = strong.loc[name]
        ok_ov = r["median overlap"] <= T["overlap"]
        ok_gain = r["mean gain %"] >= T["gain"]
        ok_det = r["mean detour %"] <= T["detour"]
        v = "PASS" if (ok_ov and ok_gain and ok_det) else ("WEAK" if (ok_gain or ok_ov) and ok_det else "FAIL")
        V.append((f"slider '{name}' (w=0.8): overlap / gain% / detour%",
                  f"{r['median overlap']:.2f} / {r['mean gain %']:.1f} / {r['mean detour %']:.1f}",
                  f"<= {T['overlap']} / >= {T['gain']} / <= {T['detour']}", v))

    cdf = pd.DataFrame(cand, columns=["profile", "origin_osmid", "dest_osmid", "origin_lat", "origin_lon",
                                      "dest_lat", "dest_lon", "fastest_m", "overlap", "detour_%", "gain_%"])
    cdf.to_csv(out_dir / f"{net.name}_study_candidates.csv", index=False)
    print("\n[6] USER STUDY candidates (w=0.8, overlap <= 0.5, detour <= 15%, gain >= 20%):")
    counts = cdf["profile"].value_counts().reindex(list(sliders), fill_value=0)
    print("  " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    usable = int((counts >= T["candidates"]).sum())
    V.append(("user-study profiles with enough candidate pairs", f"{usable}/4",
              f">= 3 profiles with >= {T['candidates']} pairs",
              "PASS" if usable >= 3 else "WEAK" if usable >= 2 else "FAIL"))


def test_search_gap(net, V, out_dir):
    print("\n[4] SEARCH GAP (node expansions; lambda = share of weight on non-time features)")
    pairs = sample_pairs(net, N_SEARCH, SEED + 1)
    rows, mismatches = [], 0
    for lam in LAMBDAS:
        w = {"time": 1 - lam, "traffic": lam / 3, "canopy": lam / 3, "climb": lam / 3}
        c = net.edge_cost(w)
        rho_min = float(net.rho(w).min())
        zero = [0.0] * net.n
        e_d, e_a, e_s, inf_ = [], [], [], []
        for s, t in pairs:
            h_adm = (rho_min * 0.999 * np.hypot(net.x - net.x[t], net.y - net.y[t])).tolist()
            h_star = net.cost_to_go(t, c)
            cd, xd = net.astar(s, t, c, zero)
            ca, xa = net.astar(s, t, c, h_adm)
            cs, xs = net.astar(s, t, c, h_star)
            mismatches += (abs(cd - ca) > 1e-6 * cd) + (abs(cd - cs) > 1e-6 * cd)
            e_d.append(xd); e_a.append(xa); e_s.append(xs)
            inf_.append(h_adm[s] / cd)
        rows.append((lam, rho_min, np.mean(e_d), np.mean(e_a), np.mean(e_s),
                     np.mean(e_a) / np.mean(e_d), np.mean(e_s) / np.mean(e_d), np.mean(inf_)))
    df = pd.DataFrame(rows, columns=["lambda", "rho_min", "dijkstra", "astar_hadm", "astar_hstar",
                                     "hadm/dijkstra", "hstar/dijkstra", "informedness h_adm/h*"])
    print(df.round(3).to_string(index=False))
    print(f"  optimal-cost mismatches between the three searches: {mismatches} (must be 0)")
    df.to_csv(out_dir / f"{net.name}_search_curve.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["lambda"], df["dijkstra"], "o-", label="Dijkstra")
        ax.plot(df["lambda"], df["astar_hadm"], "o-", label="A* (h_adm)")
        ax.plot(df["lambda"], df["astar_hstar"], "o-", label="A* (h*, perfect)")
        ax.set_xlabel("weight on non-time features (lambda)")
        ax.set_ylabel("mean nodes expanded")
        ax.set_title(net.name)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{net.name}_search_curve.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass

    rise = df["hadm/dijkstra"].iloc[-1] - df["hadm/dijkstra"].iloc[0]
    headroom = df["astar_hstar"].iloc[-1] / df["astar_hadm"].iloc[-1]
    V.append(("problem exists: rise in A*(h_adm)/Dijkstra, lambda 0 -> 0.9", round(rise, 3),
              f">= {T['curve_rise']}", verdict(rise, T["curve_rise"], 0.05)))
    V.append(("headroom: A*(h*)/A*(h_adm) expansions at lambda 0.9", round(headroom, 3),
              f"<= {T['headroom']}", verdict(headroom, T["headroom"], 0.75, higher=False)))
    V.append(("search correctness (cost mismatches)", mismatches, "0", "PASS" if mismatches == 0 else "FAIL"))


def test_labels(net, V):
    print("\n[5] LABEL BUDGET (one full turn-aware backward sweep ~ one forward sweep)")
    w = {"time": 0.4, "traffic": 0.2, "canopy": 0.2, "climb": 0.2}
    c = net.edge_cost(w)
    rng = random.Random(SEED + 2)
    times = []
    for _ in range(5):
        s = rng.randrange(net.n)
        t0 = time.perf_counter()
        net.turn_search(s, None, c, 0.2)
        times.append(time.perf_counter() - t0)
    per = float(np.median(times))
    hours = per * LABEL_RUNS / 3600
    print(f"  {per:.3f} s per sweep -> {hours:.1f} h for {LABEL_RUNS} runs on one core "
          f"(divide by cores used in parallel)")
    V.append(("label sweep, single core (h)", round(hours, 2), f"<= {T['label_hours']}",
              verdict(hours, T["label_hours"], 24, higher=False)))


# ================================================================== main
def run(path, out_dir):
    print("=" * 78)
    print(f"DATASET: {path}")
    net = Net(path)
    V = []
    test_integrity(net, V)
    test_features(net, V)
    test_sliders(net, V, out_dir)          # also runs test 6
    test_search_gap(net, V, out_dir)
    test_labels(net, V)

    vdf = pd.DataFrame(V, columns=["criterion", "value", "target", "verdict"])
    print("\nVERDICT")
    print(vdf.to_string(index=False))
    print(f"  PASS {sum(vdf.verdict == 'PASS')}, WEAK {sum(vdf.verdict == 'WEAK')}, "
          f"FAIL {sum(vdf.verdict == 'FAIL')}")
    vdf.to_csv(out_dir / f"{net.name}_verdict.csv", index=False)
    return net.name, vdf


def main():
    paths = sys.argv[1:] or ["data/jubilee_2km.graphml"]
    out_dir = Path("fitness")
    out_dir.mkdir(exist_ok=True)
    results = [run(p, out_dir) for p in paths]

    if len(results) > 1:
        comp = pd.DataFrame({"criterion": results[0][1]["criterion"]})
        for name, vdf in results:
            comp[name] = (vdf["value"].astype(str) + "  " + vdf["verdict"]).values
        print("\n" + "=" * 78 + "\nCOMPARISON")
        print(comp.to_string(index=False))
        comp.to_csv(out_dir / "comparison.csv", index=False)
    print(f"\nFiles written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
