# Label dataset — Routes That Fit You

Exact optimal cost-to-go labels for the Jubilee Hills / Banjara Hills walking graph
(`data/jubilee_3500.graphml`, 13246 nodes, 34708 directed edges, largest strongly connected component).

* **Generator:** `python -m wayfinder labels` (turn-aware backward Dijkstra, `wayfinder.search.backward`).
* **Cost model:** `wayfinder.cost` — `rho(e,w) = 0.1 + Σ_k w_k f_k(e)`, turn cost `w_junction · 30 m · penalty`.
* **Weights:** `data/splits/weights.json` (40 train, 10 held-out); keys `time, junction, traffic, canopy, climb`.
* **Targets:** `data/splits/targets.json` (200 nodes; train / val / test split by target).

| folder | weight vectors | targets | runs | samples | shards |
|---|---|---|---|---|---|
| `train_weights/` | 40 train | all 200 | 8000 | 8000000 | `w00.npz` … `w39.npz` |
| `heldout_weights/` | 10 held-out | val + test | 500 | 500000 | `w00.npz` … `w09.npz` |

Each shard holds the samples of one weight vector (`wNN` = row NN of that folder's `weights`), with
columns `node`, `target` (node indices into `data/processed/graph.npz`), `widx`, `h_star` (exact optimal
cost from `node` to `target`) and `h_adm` (the admissible heuristic). `index.json` stores the weight
vectors, a SHA-256 per shard (checked on load), and `admissibility_violations` — the number of samples
with `h_adm > h_star` found while generating, which must be 0.

Load in Python:

```python
from wayfinder.learning.labels import load_labels
labels = load_labels("data/labels/train_weights")   # dict of numpy arrays, 8,000,000 rows
```

Licences: derived from OpenStreetMap (ODbL), ESA WorldCover 2021 (CC BY 4.0), Copernicus GLO-30 DEM.
