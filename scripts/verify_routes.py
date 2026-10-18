"""Cross-check the app's routing on the real graph.

    python scripts/verify_routes.py        (after `python -m wayfinder preprocess`)

1. The default route (what the web app returns) equals the exact optimum (Dijkstra, and re-priced).
2. Raising one slider never increases the total amount of what that slider penalises along the route.
   For an exact optimiser this is a theorem: if w_k goes up, the optimal route's sum of L*f_k cannot go up.
"""
import numpy as np

from wayfinder import WEIGHT_KEYS
from wayfinder.config import load_config
from wayfinder.search import SearchSpace, dijkstra
from wayfinder.service.planner import RoutePlanner

cfg = load_config()
planner = RoutePlanner.from_config(cfg)
net = planner.net
rg, cm = net.rg, net.cm
rng = np.random.default_rng(42)
PM = {"time": 0, "traffic": 1, "canopy": 2, "climb": 3}


def amount(path, key):
    """Total of what slider `key` penalises along the route."""
    p = np.asarray(path)
    if key == "junction":
        return float(sum(net.tg.t_pen[net.tg.transition_index(int(a), int(b))] for a, b in zip(p[:-1], p[1:])))
    return float((rg.length[p] * cm.F[p, PM[key]]).sum())


opt_fail, mono_fail, changed, n_routes = 0, 0, {k: 0 for k in WEIGHT_KEYS}, 0
worst = 1.0
for q in range(150):
    s, t = (int(v) for v in rng.choice(rg.n, size=2, replace=False))
    if np.hypot(rg.x[s] - rg.x[t], rg.y[s] - rg.y[t]) < 500:
        continue
    w = rng.uniform(0, 1, size=5) * (rng.uniform(size=5) > 0.3)
    o, d = (rg.lat[s], rg.lon[s]), (rg.lat[t], rg.lon[t])
    out = planner.route(o, d, dict(zip(WEIGHT_KEYS, w)), compare_fastest=False)      # app default
    res, _, _ = planner.search(s, t, w)                                               # same default, with path
    exact = dijkstra(SearchSpace.build(cm, w), s, t)
    repriced = cm.path_cost(res.path, w)
    n_routes += 1
    ratio = out["route"]["cost"] / exact.cost
    worst = max(worst, ratio)
    if abs(ratio - 1) > 1e-9 or abs(repriced / exact.cost - 1) > 1e-9:
        opt_fail += 1
    # slider monotonicity: raise one slider to 1.0, keep the others
    for i, k in enumerate(WEIGHT_KEYS):
        lo, hi = w.copy(), w.copy()
        lo[i], hi[i] = 0.0, 1.0
        p_lo = planner.search(s, t, lo)[0].path
        p_hi = planner.search(s, t, hi)[0].path
        a_lo, a_hi = amount(p_lo, k), amount(p_hi, k)
        if a_hi > a_lo * (1 + 1e-9) + 1e-9:
            mono_fail += 1
            print(f"  monotonicity violated: query {q} slider {k}: {a_lo:.3f} -> {a_hi:.3f}")
        if p_lo != p_hi:
            changed[k] += 1

print(f"routes checked: {n_routes}")
print(f"default route == exact optimum: {n_routes - opt_fail}/{n_routes} (worst ratio {worst:.12f})")
print(f"slider monotonicity violations: {mono_fail} / {n_routes * len(WEIGHT_KEYS)}")
print("route changed when the slider went 0 -> 1:", {k: f"{v}/{n_routes}" for k, v in changed.items()})
