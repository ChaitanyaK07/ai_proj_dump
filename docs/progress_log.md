# Weekly progress log

One entry per week per pair: what landed, which gate was checked, what is blocked. Link result files.

## Week 1 — interfaces frozen, fixture graph, OSM extraction, Dijkstra and A*

* Gate: *a route computes on real data; coverage table exists* —
  `data/jubilee_3500_coverage.csv`, `results/audit/*` (pre-project audit), `tests/test_search.py`.

## Week 2 — projection, geometry, weighted A*

* Gate: *A* matches Dijkstra's path with fewer expansions* — `tests/test_search.py::test_astar_optimal_with_fewer_expansions`.

## Week 3 — turn-aware states, five features, backward Dijkstra, frozen benchmark

* Gate: *moving a slider visibly changes the route* — web UI; `tests/test_search.py::test_turn_penalties_change_routes`.
* Benchmark frozen: `data/benchmark/queries.csv` (+ `queries.sha256`).

## Week 4 — label sweep, embeddings, first trained model, batched inference

* Gate: **go / no-go** — `results/go_no_go.json`.

## Week 5 — Focal Search and the W sweep, routing service, heuristic quality, generalisation

* Gate: *bound holds empirically across the sweep* — `results/bound_sweep.csv`.

## Week 6 — web interface, latency profiling, user study, results locked

* Gate: *every claim traces to a result file* — `results/summary.md`.

## Week 7 — buffer, clean-clone reproducibility test, report, demo video, mock viva

* Clean clone: `pip install -r requirements.txt && pip install -e . --no-deps && python -m wayfinder all && python -m pytest`.
