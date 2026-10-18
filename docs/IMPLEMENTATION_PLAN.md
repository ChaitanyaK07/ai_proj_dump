# Implementation Plan — Routes That Fit You

This document maps every element of the CS F407 proposal to the code that implements it,
fixes the interfaces the four pairs work against, and lists the order in which the pipeline runs.

## 1. Requirements traced from the proposal

| Proposal item | Where it lives |
|---|---|
| OSM extraction, WorldCover canopy, DEM grade, coverage audit | `scripts/acquisition/build_dataset.py`, `scripts/acquisition/fitness_report.py`, `scripts/acquisition/check_elevation.py` (need `osmnx`; outputs already in `data/` and `results/audit/`) |
| UTM projection (EPSG:32644) | `src/wayfinder/geo.py` (forward/inverse UTM, haversine, bearings) |
| Graph loading, largest SCC, edge geometry | `src/wayfinder/graph/road_graph.py` |
| Five feature extractors (time, junction load, traffic, canopy, climb) | `src/wayfinder/graph/features.py` |
| Turn-aware states `(node, incoming edge)` | `src/wayfinder/graph/turn_graph.py` |
| Cost model `rho`, `c(e_in, e_out, w)`, `rho_min`, `h_adm` | `src/wayfinder/cost.py` |
| Dijkstra, A*, weighted A*, node accounting | `src/wayfinder/search/astar.py` |
| Focal Search (OPEN by `g + h_adm`, FOCAL chosen by the learned heuristic) | `src/wayfinder/search/focal.py` |
| Backward Dijkstra (exact cost-to-go, reference + fast scipy version) | `src/wayfinder/search/backward.py` |
| Heuristics: zero, admissible, perfect `h*`, tuned classical, learned | `src/wayfinder/search/heuristics.py`, `src/wayfinder/learning/inference.py` |
| Weight-vector sampling (40 train + 10 held-out), λ family, study profiles | `src/wayfinder/learning/weights.py` |
| Label pipeline (40 × 200 backward runs) and dataset publication | `src/wayfinder/learning/labels.py`, `data/labels/DATASHEET.md` |
| node2vec embeddings (biased walks + skip-gram, implemented from scratch) | `src/wayfinder/learning/node2vec.py` |
| Network, ratio parameterisation `r_hat = 1 + softplus(MLP(x))` | `src/wayfinder/learning/model.py` |
| Training loop, hyperparameter pass, checkpointing, 3 seeds | `src/wayfinder/learning/train.py` |
| Batched inference with cached embeddings | `src/wayfinder/learning/inference.py` |
| Frozen 200-query benchmark | `src/wayfinder/evaluation/benchmark.py` → `data/benchmark/queries.csv` |
| Problem-statement curve (expansions vs non-distance weight) | `src/wayfinder/evaluation/experiments.py::problem_curve` |
| Evaluation harness: expansions, generated, latency, suboptimality | `src/wayfinder/evaluation/harness.py` |
| Heuristic quality: admissibility / consistency violations, informedness | `src/wayfinder/evaluation/quality.py` |
| W sweep 1.0 → 2.0, guaranteed vs achieved suboptimality | `src/wayfinder/evaluation/experiments.py::bound_sweep` |
| Generalisation to held-out weight vectors | `src/wayfinder/evaluation/experiments.py::generalisation` |
| Week 4 go / no-go verdict and fallback | `src/wayfinder/evaluation/report.py` |
| All plots | `src/wayfinder/evaluation/plots.py` |
| Routing service (FastAPI) | `src/wayfinder/service/api.py`, `src/wayfinder/service/planner.py` |
| Web interface with five sliders on a map | `web/index.html`, `web/app.js`, `web/styles.css` |
| User study (12 OD pairs × 3 profiles, unlabelled random order, agreement rate) | `src/wayfinder/study/user_study.py`, `web/study.html`, `web/study.js` |
| Regression tests against exact optimal routes | `tests/` |
| AI-usage log, weekly progress log | `docs/ai_usage_log.md`, `docs/progress_log.md` |

## 2. Frozen interfaces (Week 1)

* **Weight vector** — five sliders, each in `[0, 1]`, always in this order:
  `w = (time, junction, traffic, canopy, climb)`. `junction` is `w_turn` in the formulas.
* **Per-metre features** `f_k(e) ∈ [0, 1]` for `k ∈ (time, traffic, canopy, climb)`:
  * `time` = `(1 / speed(class)) / (1 / 0.5)`: 1.3 m/s footway → 0.385, steps 0.5 m/s → 1.0
  * `traffic` = ordinal rank of the OSM highway class (trunk 1.0 … footway 0.0)
  * `canopy` = `1 − tree-cover share` in the 15 m buffer (lack of canopy is the cost)
  * `climb` = `clip(grade, 0, 0.15) / 0.15`; `highway=steps` gets the fixed penalty 1.0
* **Cost** —
  `rho(e, w) = 0.1 + Σ_k w_k f_k(e)`,
  `c(e_in, e_out, w) = length(e_out) · rho(e_out, w) + w_junction · 30 m · turn_penalty(e_in, e_out)`
* **Junction load** `turn_penalty(e_in, e_out)` = angle term from bearings
  (`<30°: 0`, `<75°: 0.3`, `<150°: 1.0`, else `3.0` for a U-turn) plus `0.2` when the node has
  three or more neighbours (node degree). No OSM tag is used.
* **Search state** = directed edge just traversed; the origin is a virtual state with no incoming edge.
  A query is solved when a state whose head is the destination is *expanded*.
* **`h_adm(v, t, w) = rho_min(w) · s · haversine(v, t)`**, `rho_min(w) = min_e rho(e, w)`, and `s ≤ 1`
  is the smallest `length(e) / haversine(u, v)` over the graph (×(1 − 1e-9)), which makes `h_adm`
  consistent *by construction* on this graph (proof in `cost.py`).
* **Learned heuristic** `h_learn = r_hat · h_adm`, `r_hat = 1 + softplus(MLP(x)) ≥ 1`,
  `x = [node2vec(v), node2vec(t), w, (Δx, Δy, ‖Δ‖) / 1 km]`.
* **Label** = exact optimal cost-to-go from node `v` (as an origin) to `t` under `w`, from one turn-aware
  backward Dijkstra run per `(w, t)`.

## 3. Pipeline (run order)

```
python -m wayfinder preprocess      # graphml → data/processed/ (arrays, features, turn graph)
python -m wayfinder splits          # 40 train + 10 held-out weights, 200 label targets (frozen)
python -m wayfinder benchmark       # 200 frozen queries (refuses to overwrite)
python -m wayfinder labels          # 8,000 backward runs (+ held-out-weight test runs)
python -m wayfinder node2vec        # node embeddings
python -m wayfinder train --hparam  # hyperparameter pass, then 3 seeds
python -m wayfinder evaluate        # every experiment, plots, results/summary.md
python -m wayfinder study-prepare   # 12 OD pairs × 3 profiles for the user study
python -m wayfinder serve           # FastAPI + web UI on http://127.0.0.1:8000
```

`python -m wayfinder all` runs the first seven steps in order.

## 4. Experiments and the result file behind each claim

| Claim | Result file |
|---|---|
| Expansions of A*(h_adm) climb toward Dijkstra as non-distance weight grows | `results/problem_curve.csv`, `results/plots/problem_curve.png` |
| Learned guidance expands fewer nodes than the classical baseline (go/no-go) | `results/benchmark_summary.csv`, `results/go_no_go.json` |
| Focal Search bound holds for every W in 1.0 … 2.0 | `results/bound_sweep.csv`, `results/plots/bound_guaranteed_vs_achieved.png` |
| Heuristic quality (violations, informedness) | `results/heuristic_quality.csv`, `results/plots/informedness_hist.png` |
| Focal Search cannot beat its floor (states with `g* + h_adm < C*/W`), whatever the guide | `results/benchmark_raw.csv` (`focal_floor`), `results/bound_sweep.csv` |
| Generalisation to held-out weights | `results/generalisation.csv` |
| Latency is interactive | `results/benchmark_summary.csv` (`latency_ms`) |
| Training curves and chosen hyperparameters | `results/training/*.csv` |
| User-study agreement per profile | `results/user_study/summary.csv` |

## 5. Week-4 fallback

`results/go_no_go.json` compares the median expansions of Focal Search guided by the learned heuristic
against A*(h_adm) **and** against Focal Search guided by the *tuned classical* heuristic
(`h_tuned = κ(w) · h_adm`, κ fitted on training labels) at the same W. GO needs both wins. Otherwise the
report switches the headline to the tuned classical guide and keeps the failure analysis: the quality
table, Focal Search with the perfect guide h* (the ceiling for any guide) and the per-query `focal_floor`.

## 6. Out of scope (as stated in the proposal)

Live traffic, public transport, turn-by-turn navigation, full-city routing, night-time lighting,
ALT landmark baselines, frontend polish beyond a functional map with sliders.
