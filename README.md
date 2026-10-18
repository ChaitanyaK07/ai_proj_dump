# Routes That Fit You

**A Personalised Multi-Objective Walking Route Planner with Learned Search Heuristics**

CS F407 Artificial Intelligence · Team **Wayfinder** · Track: Application (Search & Problem Solving + Applications of Deep Learning)

A walking route planner for Jubilee Hills / Banjara Hills, Hyderabad, in which **the user decides what matters**.
Five sliders set weights over **walking time, junction load, traffic exposure, canopy and climb**, and the planner
returns the best route under those weights. A small neural network, conditioned on the slider weights, learns to
guide the search, and **Focal Search** keeps every route provably within a stated factor of optimal. Everything is
evaluated on 200 frozen benchmark queries against exact ground truth, plus a small user study.

---

## Contents

1. [Quick start](#quick-start)
2. [Using the app](#using-the-app)
3. [Running the pipeline](#running-the-pipeline)
4. [How it works](#how-it-works)
5. [Results](#results)
6. [Repository layout](#repository-layout)
7. [Data](#data)
8. [Testing](#testing)
9. [Configuration](#configuration)
10. [Team and work division](#team-and-work-division)
11. [Scope, licences and references](#scope-licences-and-references)

---

## Quick start

Requires **Python 3.10+** (tested on 3.11.9). Everything runs on a laptop CPU.

```bash
git clone https://github.com/ChaitanyaK07/WayFinder.git
cd WayFinder
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
```

The repository already contains the graph, the frozen splits and benchmark, the trained models and all results, so
the app works straight after installing:

```bash
python -m wayfinder preprocess        # ~15 s: builds data/processed/graph.npz from the GraphML
python -m wayfinder serve             # open http://127.0.0.1:8000
```

Check everything with the test suite (~1 minute):

```bash
python -m pytest
```

> **PyTorch:** the CPU build is enough. If pip downloads a large CUDA build, install the CPU one first:
> `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`

---

## Using the app

Start the service with `python -m wayfinder serve` and open **http://127.0.0.1:8000**.

1. **Click the map** to set a start, then click again to set a destination (drag either marker to move it).
2. **Move the five sliders** (each 0 to 1). The route is recomputed as you drag:

   | Slider | What a higher value does |
   |---|---|
   | Walking time | prefers shorter, faster routes |
   | Junction load | avoids sharp turns, U-turns and busy junctions |
   | Traffic exposure | stays off major roads (proxy from road class) |
   | Canopy | walks near tree cover (proximity, not measured shade) |
   | Climb | avoids uphill grades and stairs |

3. Or pick a **preset**: Fastest, Shade seeker, Quiet streets, Easy on the knees.
4. The panel compares **your route with the fastest route** (length, walking time, canopy share, traffic exposure,
   climb, stairs, junction load) and shows the search effort and latency.

**The default search is exact:** the route shown is the best possible route for the slider values you chose
(A* with an admissible heuristic). Under *Search* you can switch to Focal Search with the learned guide, or
weighted A*, which trade a guaranteed **within W × the best route** bound for speed. Tick *Verify against the exact
optimum* to see how close a bounded route actually is.

The **user-study page** is at **http://127.0.0.1:8000/study** (see [Results](#results)).

API, if you want to call it directly:

```bash
curl -X POST http://127.0.0.1:8000/api/route -H "Content-Type: application/json" -d "{\"origin\": {\"lat\": 17.4325, \"lon\": 78.4070}, \"destination\": {\"lat\": 17.4140, \"lon\": 78.4380}, \"weights\": {\"time\": 0.2, \"canopy\": 1.0}}"
```

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | service status, graph size, which guide is loaded |
| `GET /api/meta` | study area bounds, slider names, presets, algorithms |
| `POST /api/route` | route for `origin`, `destination`, `weights`; optional `algorithm` (`astar`, `focal`, `wastar`, `dijkstra`), `W`, `compare_fastest`, `verify` |
| `GET /api/study/trials`, `POST /api/study/response` | user study |

---

## Running the pipeline

Every step reads and writes files only, so steps can be re-run independently. In order:

| # | Command | What it does | Time* | Writes |
|---|---|---|---|---|
| 1 | `python -m wayfinder preprocess` | GraphML → arrays, features, turn-aware graph | 15 s | `data/processed/`, `results/feature_summary.csv` |
| 2 | `python -m wayfinder splits` | freezes 40 + 10 weight vectors and 200 label targets | 1 s | `data/splits/` |
| 3 | `python -m wayfinder benchmark` | freezes the 200 benchmark queries (refuses to overwrite) | 1 s | `data/benchmark/` |
| 4 | `python -m wayfinder labels` | 8,000 + 500 backward-Dijkstra runs | 2–3 min | `data/labels/` |
| 5 | `python -m wayfinder node2vec` | node embeddings | 1 min | `data/processed/node2vec.npy` |
| 6 | `python -m wayfinder train --hparam` | hyperparameter pass, then 3 training seeds | 25–40 min | `models/`, `results/training/` |
| 7 | `python -m wayfinder evaluate` | every experiment, plots, report | 40–50 min | `results/` |
| 8 | `python -m wayfinder report` | rebuilds tables, plots and `summary.md` from raw results | 10 s | `results/` |
| 9 | `python -m wayfinder study-prepare` | picks 12 OD pairs × 3 preference profiles | 1 min | `data/user_study/` |
| 10 | `python -m wayfinder study-analyze` | agreement rate per profile from collected answers | 1 s | `results/user_study/` |

\* single laptop CPU.

`python -m wayfinder all` runs steps 1–7 and 9. Steps 2 and 3 are **frozen**: they are committed and are not
regenerated unless you pass `--force`. Use `python -m wayfinder evaluate --limit 5` for a quick smoke test.
On Unix, `make install`, `make test`, `make pipeline` and `make serve` wrap the same commands.

To re-download the raw data from OpenStreetMap, ESA WorldCover and the Copernicus DEM (not needed, the graph is
committed): `python -m pip install -r requirements-acquisition.txt`, then see `scripts/acquisition/build_dataset.py`.

---

## How it works

### Cost model

Every edge `e` has four per-metre feature scores `f_k(e) ∈ [0, 1]`, and every turn between two edges has a junction-load penalty:

```
rho(e, w)          = 0.1 + Σ_k w_k · f_k(e)                     k = time, traffic, canopy, climb
c(e_in, e_out, w)  = length(e_out) · rho(e_out, w) + w_junction · 30 m · turn_penalty(e_in, e_out)
h_adm(v, t, w)     = rho_min(w) · haversine(v, t)               rho_min(w) = min_e rho(e, w)
h_learn(v, t, w)   = r_hat · h_adm(v, t, w),   r_hat = 1 + softplus(MLP(x)) ≥ 1
```

| Feature | Source | Status | Definition |
|---|---|---|---|
| Walking time | OSM highway class, length | measured | `0.5 / speed`; 1.3 m/s on footways down to 0.5 m/s on steps |
| Junction load | node degree, way geometry | derived, exact | turn angle from bearings (`<30°` 0, `<75°` 0.3, `<150°` 1, U-turn 3), +0.2 at nodes with ≥ 3 neighbours |
| Traffic exposure | OSM highway class | proxy | ordinal rank (trunk/primary 1.0 … footway 0.0); OSM has no traffic-volume tag |
| Canopy proximity | ESA WorldCover 2021 (10 m) | proxy | `1 − tree-cover share` in a 15 m buffer; correlates with shade but does not measure it |
| Climb | Copernicus GLO-30 DEM | derived, reliable | uphill grade clipped at 15 %, scaled to [0, 1]; stairs get a fixed 1 |

Search states are `(node, incoming edge)` pairs, so turns can be priced.

### Search

* **Dijkstra, A\*, weighted A\*** over the turn-aware state graph, with counts of expanded and generated states.
* **`h_adm` is admissible and consistent by construction.** It uses haversine on the same sphere OSMnx measured the
  edge lengths on, scaled by the graph's smallest `length / haversine` ratio. The proof is in `src/wayfinder/cost.py`,
  and the tests check it on every edge.
* **Focal Search.** OPEN is ordered by the safe `f = g + h_adm`. FOCAL holds every OPEN state with `f ≤ W · f_min`.
  The learned heuristic, which is free to overestimate, picks which FOCAL state to expand. The route cost is then
  at most `W ×` optimal.
* **Backward Dijkstra** gives the exact cost-to-go from every node to a target in one run. It provides the training
  labels, the perfect heuristic `h*`, and the ground truth for evaluation.

### Learned heuristic

* **Input:** node2vec embeddings of the node and the target (implemented from scratch, 32-d), the 5 slider weights,
  and the straight-line offset to the target. **Output:** the ratio `r_hat ≥ 1`, so `h_learn` is never less
  informed than `h_adm`.
* **Labels:** 40 training weight vectors × 200 target nodes = 8,000 backward runs (~2 min with SciPy). 10 further
  weight vectors are held out, and the train/val/test split (150/25/25) is by target node.
* **Training:** an MLP with two hidden layers of 256 units, Adam, MSE on `log(h*/h_adm)`, early stopping and
  checkpointing, a hyperparameter pass, and 3 seeds.
* **Batched inference:** the first layer is split so that node and target projections are cached. On a cache miss,
  the whole 250 m spatial tile is inferred in one numpy batch.

---

## Results

Full tables, with the file behind every number: **[`results/summary.md`](results/summary.md)**. On the 200 frozen queries:

* **Correctness.** Dijkstra, A\*(h_adm) and A\*(h\*) agree on every optimum. `h_adm` has zero admissibility and
  consistency violations. Across the whole W sweep (1.0–2.0, three seeds), Focal Search has **0 bound violations**.
* **The problem exists.** A\*(h_adm) expansions rise from 35 % to 81 % of Dijkstra's as weight moves off walking
  time (`results/problem_curve.csv`).
* **The learned heuristic is accurate and generalises.** Its informedness `h/h*` is ≈ 1.02, against 0.17 for
  `h_adm`, and its accuracy on held-out weight vectors matches that on training weights. Used directly in A\* (no
  bound) it expands ~11× fewer states than A\*(h_adm), at 4 % mean and 51 % worst-case suboptimality.
* **Week-4 gate: NO-GO for the headline.** Inside Focal Search, the learned guide cuts expansions by 15 % against
  A\*(h_adm) at W = 1.5, but does not beat the tuned classical guide or weighted A\* at the same W. Even a
  *perfect* guide (h\*) only reaches 18 %. FOCAL is gated by the weak admissible OPEN heuristic, so guide quality
  barely matters; the per-query `focal_floor` column measures this. Following the proposal's fallback, the headline
  becomes Focal Search with the tuned classical guide.
* **App check.** Over 147 random trips with random slider settings, the default route equals the exact optimum
  147/147 times. Raising any slider never increased what it penalises (0 violations in 735 checks). Reproduce with
  `python scripts/verify_routes.py`.
* **User study.** 12 trials (4 per profile) are prepared in `data/user_study/`. Collect answers at `/study`, then run
  `python -m wayfinder study-analyze`. No statistical significance is claimed at this sample size.

Figures are in `results/plots/`: `problem_curve.png`, `expansions_vs_W.png`, `bound_guaranteed_vs_achieved.png`,
`informedness_hist.png`, `generalisation.png`, `latency.png`, `training_curves.png`.

---

## Repository layout

```
WayFinder/
├── README.md                      this file
├── pyproject.toml                 package metadata; `pip install -e .` gives the `wayfinder` command
├── requirements.txt               runtime dependencies
├── requirements-dev.txt           + test dependencies
├── requirements-acquisition.txt   + raw-data download (osmnx, rasterio, ...)
├── Makefile                       Unix shortcuts for the commands above
├── configs/default.yaml           every pipeline setting (seeds, sizes, hyperparameters, paths)
├── data/                          graph, frozen splits and benchmark, labels, embeddings, study trials (data/README.md)
├── docs/
│   ├── IMPLEMENTATION_PLAN.md     every proposal item mapped to its code and result file
│   ├── ai_usage_log.md            AI-usage disclosure
│   └── progress_log.md            weekly gates
├── models/                        3 trained heuristic checkpoints, tuned classical fallback, chosen hyperparameters
├── results/                       all result tables and figures, summary.md; results/audit/ = pre-project data audit
├── scripts/
│   ├── acquisition/               OSM + WorldCover + DEM extraction, coverage audit, fitness report
│   └── verify_routes.py           cross-check: app routes are optimal and sliders behave
├── src/wayfinder/
│   ├── geo.py                     UTM <-> WGS84, haversine, bearings, WKT
│   ├── cost.py                    cost model and h_adm (with the consistency proof)
│   ├── network.py                 loads graph + turn graph + cost model
│   ├── graph/                     road_graph.py (arrays, SCC, geometry), features.py (5 extractors), turn_graph.py
│   ├── search/                    astar.py, focal.py, backward.py, heuristics.py, space.py
│   ├── learning/                  weights, splits, labels, node2vec, model, train, inference, artifacts
│   ├── evaluation/                benchmark, harness, quality, experiments, plots, report
│   ├── study/                     user-study trials, responses, agreement analysis
│   ├── service/                   planner.py (routing facade), api.py (FastAPI)
│   ├── pipeline.py                the pipeline steps
│   └── cli.py                     `python -m wayfinder <step>`
├── tests/                         pytest suite (83 tests)
└── web/                           index.html + app.js (map and sliders), study.html + study.js, styles.css
```

---

## Data

| Path | Contents | In git |
|---|---|---|
| `data/jubilee_3500.graphml` | 7 × 7 km walking graph centred at 17.4220 N, 78.4200 E (UTM 44N): 13,246 nodes, 34,708 directed edges, fully strongly connected; `canopy` and `grade` on every edge | yes |
| `data/jubilee_3500_coverage.csv` | Week-1 OSM tag coverage table | yes |
| `data/splits/`, `data/benchmark/` | frozen weight vectors, label targets, 200 queries (+ SHA-256, checked on load) | yes |
| `data/processed/node2vec.*`, `graph_summary.json` | embeddings the trained models depend on; graph statistics | yes |
| `data/processed/graph.npz` | graph as arrays, regenerated by `preprocess` | no |
| `data/labels/*.npz` | ~150 MB of exact labels, regenerated by `labels` (`DATASHEET.md` describes them) | no |
| `data/user_study/` | 12 study trials with both routes | yes |

Details: [`data/README.md`](data/README.md).

---

## Testing

```bash
python -m pytest                  # all 83 tests, ~1 min
python -m pytest -m "not slow"    # skip the checks that need the real graph
python scripts/verify_routes.py   # 147 random trips on the real graph
```

The suite runs on a synthetic fixture graph plus the real graph. It checks:

* Dijkstra against a networkx reference;
* A\*, weighted A\* and backward Dijkstra (reference vs SciPy) against Dijkstra;
* `h_adm` admissibility and consistency;
* the Focal Search bound under a random, wildly inadmissible guide;
* the focal floor;
* node2vec walks;
* numpy inference against PyTorch;
* the frozen-benchmark hash;
* the API and the user-study flow;
* that the app's default route is exactly optimal and every slider behaves.

---

## Configuration

All settings live in [`configs/default.yaml`](configs/default.yaml). To change them without editing that file,
pass an override file: `python -m wayfinder --config my.yaml evaluate`. Cost-model constants (`BASE = 0.1`,
`TURN_M = 30`, speeds, traffic ranks, the 15 % grade cap) are frozen interfaces in `src/wayfinder/cost.py` and
`src/wayfinder/graph/features.py`.

---

## Team and work division

| Pair | Build phase (Weeks 1–4) | Second phase (Weeks 5–7) | Members |
|---|---|---|---|
| Graph | OSM extraction, coverage audit, UTM projection, turn-aware graph, five feature extractors, cost model | routing service (FastAPI), web interface with sliders | Anisha Potdar, Ananya Avunooru |
| Search | Dijkstra, A\*, weighted A\*, admissible heuristic, turn-aware expansion, node accounting, batched inference | Focal Search, W sweep, empirical bound verification | Srujana N S, Anushka Sahoo |
| Data | backward Dijkstra at scale, weight-vector sampling, label pipeline, dataset publication, node2vec | evaluation harness, heuristic-quality analysis, plots | Soujanya Lakshmi Tayyala, Chandana N |
| Model | network architecture, ratio parameterisation, training loop, hyperparameter pass, checkpointing | generalisation on held-out weights, user study | Vinny Polamarasetty, Chaitanya Srikar Kastury |

The weekly gates are tracked in [`docs/progress_log.md`](docs/progress_log.md). AI assistance is disclosed in
[`docs/ai_usage_log.md`](docs/ai_usage_log.md). The project makes no novelty claim: its contribution is the
ground-truth-verified evaluation of preference-conditioned learned heuristics and Focal Search.

---

## Scope, licences and references

**In scope:** personalised walking routes inside the Jubilee Hills / Banjara Hills study area, from static OSM, ESA
WorldCover and Copernicus DEM data. **Out of scope:** live traffic, public transport, turn-by-turn navigation,
full-city routing, night-time lighting, ALT landmark baselines, and frontend polish beyond a functional map with sliders.

**Data licences:** OpenStreetMap (ODbL), ESA WorldCover 2021 (CC BY 4.0), Copernicus GLO-30 DEM. No purchased
data, API keys or scraping.

1. Hart, Nilsson, Raphael (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths. *IEEE Trans. SSC* 4(2).
2. Pearl, Kim (1982). Studies in Semi-Admissible Heuristics. *IEEE TPAMI* 4(4). (Focal Search)
3. Goldberg, Harrelson (2005). Computing the Shortest Path: A\* Search Meets Graph Theory. *SODA*.
4. Boeing (2017). OSMnx: New Methods for Acquiring, Constructing, Analyzing, and Visualizing Complex Street Networks. *CEUS* 65.
5. Grover, Leskovec (2016). node2vec: Scalable Feature Learning for Networks. *KDD*.
6. Russell, Norvig (2020). *Artificial Intelligence: A Modern Approach*, 4th ed., Ch. 3–4 and 22.
