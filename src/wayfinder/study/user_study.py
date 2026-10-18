"""Human evaluation: does the cost function reflect what people would choose?

``prepare_study`` picks 12 origin-destination pairs inside Jubilee Hills / Banjara Hills (four per
stated preference profile) where the profile route clearly differs from the fastest route, and stores
both exact-optimal routes. The study page shows the two routes unlabelled, in an order randomised per
participant and trial, with the profile's scenario; the participant picks the one they would take.
``analyze_responses`` reports, per profile, the share of choices for the preference route and the
number of participants. No statistical significance is claimed at this sample size.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from wayfinder import geo
from wayfinder.cost import as_weight_array, weight_dict
from wayfinder.learning.weights import FASTEST, PROFILES
from wayfinder.network import Network
from wayfinder.search import SearchSpace, dijkstra

TRIALS_FILE = "trials.json"
RESPONSES_FILE = "responses.jsonl"

# the route property each profile is meant to improve: (metric key, lower is better)
PROFILE_METRIC = {"shade": ("canopy_share", False), "quiet": ("traffic", True), "easy": ("effort_per_km", True)}


def _metrics(cm, path):
    m = cm.route_metrics(path)
    km = m["length_m"] / 1000.0
    m["effort_per_km"] = (m["climb_m"] + m["stairs_m"] * 0.5 + m["turn_load"]) / km if km > 0 else 0.0
    return m


def _clean(name) -> str:
    return " ".join(str(name).split())       # some OSM names contain line breaks


def _street_near(rg, v: int, max_m: float = 400.0) -> str:
    """Name of a street at the node, else the nearest named street (only ~1 in 5 OSM edges has a name)."""
    names = Counter(_clean(rg.name[e]) for e in list(rg.out_edges(v)) + list(rg.in_edges(v)) if rg.name[e])
    if names:
        return names.most_common(1)[0][0]
    named = np.nonzero(rg.name != "")[0]
    if len(named):
        mx = (rg.x[rg.tail[named]] + rg.x[rg.head[named]]) / 2
        my = (rg.y[rg.tail[named]] + rg.y[rg.head[named]]) / 2
        d = np.hypot(mx - rg.x[v], my - rg.y[v])
        i = int(np.argmin(d))
        if d[i] <= max_m:
            return f"{_clean(rg.name[named[i]])} (about {d[i]:.0f} m away)"
    return "an unnamed street"


def prepare_study(net: Network, cfg: dict, out_dir: Path, max_tries: int = 4000) -> list[dict]:
    rg, cm = net.rg, net.cm
    sc, area = cfg["user_study"], cfg["study_area"]
    rng = np.random.default_rng(sc["seed"])
    cx, cy = geo.latlon_to_utm(area["center_lat"], area["center_lon"])
    inside = np.nonzero(np.hypot(rg.x - cx, rg.y - cy) <= sc["radius_m"])[0]
    fastest_space = SearchSpace.build(cm, FASTEST)
    per_profile = sc["n_pairs"] // len(PROFILES)
    trials = []
    for name, prof in PROFILES.items():
        space = SearchSpace.build(cm, prof["weights"])
        key, lower = PROFILE_METRIC[name]
        found, tries = 0, 0
        while found < per_profile and tries < max_tries:
            tries += 1
            s, t = (int(i) for i in rng.choice(inside, size=2, replace=False))
            if np.hypot(rg.x[s] - rg.x[t], rg.y[s] - rg.y[t]) < sc["min_euclid_m"]:
                continue
            r0 = dijkstra(fastest_space, s, t)
            r1 = dijkstra(space, s, t)
            m0, m1 = _metrics(cm, r0.path), _metrics(cm, r1.path)
            shared = float(rg.length[list(set(r0.path) & set(r1.path))].sum())
            overlap = shared / m0["length_m"]
            detour = 100 * (m1["length_m"] / m0["length_m"] - 1)
            a, b = m0[key], m1[key]
            if a <= 1e-9:
                continue
            gain = 100 * ((a - b) / a if lower else (b - a) / a)
            if overlap > sc["max_overlap"] or detour > sc["max_detour_pct"] or gain < sc["min_gain_pct"]:
                continue
            found += 1
            trials.append({
                "trial_id": f"{name}-{found}", "profile": name, "scenario": prof["scenario"],
                "weights": weight_dict(as_weight_array(prof["weights"])),
                "origin": {"node": s, "osmid": str(rg.osmid[s]), "lat": float(rg.lat[s]), "lon": float(rg.lon[s]),
                           "near": _street_near(rg, s)},
                "destination": {"node": t, "osmid": str(rg.osmid[t]), "lat": float(rg.lat[t]),
                                "lon": float(rg.lon[t]), "near": _street_near(rg, t)},
                "routes": {"fastest": {"latlon": rg.path_latlon(r0.path), "metrics": m0},
                           "profile": {"latlon": rg.path_latlon(r1.path), "metrics": m1}},
                "overlap": overlap, "detour_pct": detour, "gain_pct": gain, "metric": key,
            })
        if found < per_profile:
            print(f"  warning: only {found} usable pairs for profile '{name}' after {tries} tries")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / TRIALS_FILE).write_text(json.dumps({"created": time.strftime("%Y-%m-%d"), "trials": trials}, indent=1),
                                       encoding="utf-8")
    pd.DataFrame([{"trial_id": t["trial_id"], "profile": t["profile"], "origin_near": t["origin"]["near"],
                   "destination_near": t["destination"]["near"], "overlap": round(t["overlap"], 3),
                   "detour_pct": round(t["detour_pct"], 1), "gain_pct": round(t["gain_pct"], 1),
                   "metric": t["metric"]} for t in trials]).to_csv(out_dir / "trials.csv", index=False)
    return trials


def load_trials(study_dir: Path) -> list[dict]:
    p = study_dir / TRIALS_FILE
    return json.loads(p.read_text(encoding="utf-8"))["trials"] if p.exists() else []


def record_response(results_dir: Path, response: dict) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / RESPONSES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({**response, "received": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")


def analyze_responses(results_dir: Path) -> pd.DataFrame:
    p = results_dir / RESPONSES_FILE
    if not p.exists():
        raise FileNotFoundError(f"no responses yet at {p}")
    df = pd.read_json(p, lines=True)
    # one answer per participant and trial: keep the latest
    df = df.sort_values("received").drop_duplicates(["participant_id", "trial_id"], keep="last")
    rows = []
    for prof, d in df.groupby("profile"):
        rows.append({"profile": prof, "participants": int(d["participant_id"].nunique()), "choices": len(d),
                     "chose_preference_route": int((d["choice"] == "profile").sum()),
                     "agreement_rate": float((d["choice"] == "profile").mean())})
    rows.append({"profile": "all", "participants": int(df["participant_id"].nunique()), "choices": len(df),
                 "chose_preference_route": int((df["choice"] == "profile").sum()),
                 "agreement_rate": float((df["choice"] == "profile").mean())})
    out = pd.DataFrame(rows)
    out.to_csv(results_dir / "summary.csv", index=False)
    return out
