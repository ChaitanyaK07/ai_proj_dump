# Label dataset — Routes That Fit You

Exact optimal cost-to-go labels for the Jubilee Hills / Banjara Hills walking graph
(`data/jubilee_3500.graphml`, 13246 nodes, 34708 directed edges, largest strongly connected component).

* **Generator:** `python -m wayfinder labels` (turn-aware backward Dijkstra, `wayfinder.search.backward`).
* **Cost model:** `wayfinder.cost` — `rho(e,w) = 0.1 + Σ_k w_k f_k(e)`, turn cost `w_junction · 30 m · penalty`.
* **Weights:** `data/splits/weights.json` (40 train, 10 held-out); keys `time, junction, traffic, canopy, climb`.
* **Targets:** `data/splits/targets.json` (200 nodes; train / val / test split by target).

| file | weight vectors | targets | runs | samples |
|---|---|---|---|---|
| `labels_train_weights.npz` | 40 train | all 200 | 8000 | 8000000 |
| `labels_heldout_weights.npz` | 10 held-out | val + test | 500 | 500000 |

Columns (one row per sample): `node`, `target` (node indices into `data/processed/graph.npz`), `widx`
(row of `weights`), `h_star` (exact optimal cost from `node` to `target`), `h_adm` (the admissible
heuristic). `h_star / h_adm >= 1` always holds; the number of violations found while generating is
stored as `admissibility_violations` (must be 0).

Licences: derived from OpenStreetMap (ODbL), ESA WorldCover 2021 (CC BY 4.0), Copernicus GLO-30 DEM.
