# Data

| Path | What | Tracked | Produced by |
|---|---|---|---|
| `jubilee_3500.graphml` | Walking graph, 7 × 7 km centred at 17.4220 N, 78.4200 E, UTM 44N, with `canopy` (ESA WorldCover) and `grade` (Copernicus GLO-30) on every edge | yes | `scripts/acquisition/build_dataset.py` |
| `jubilee_3500_coverage.csv` | Week-1 OSM tag coverage table | yes | same |
| `processed/graph.npz` | Graph as arrays (largest SCC, oriented geometry, bearings); byte-identical on every rebuild | yes | `python -m wayfinder preprocess` |
| `processed/graph_summary.json` | Size, transitions, relief, haversine scale | yes | same |
| `processed/node2vec.npy`, `node2vec.json` | 32-d node2vec embeddings the trained models depend on | yes | `python -m wayfinder node2vec` |
| `splits/weights.json` | 40 training + 10 held-out weight vectors (frozen) | yes | `python -m wayfinder splits` |
| `splits/targets.json` | 200 label targets split 150 / 25 / 25 by target node (frozen) | yes | same |
| `benchmark/queries.csv`, `queries.sha256` | The 200 frozen benchmark queries and their hash | yes | `python -m wayfinder benchmark` (refuses to overwrite) |
| `labels/train_weights/` | Exact cost-to-go labels: 40 training weight vectors × 200 targets = 8,000 backward runs, 8,000,000 samples, one shard per weight vector (`w00.npz` … `w39.npz`, ~3.4 MB each) + `index.json` with SHA-256 checksums | yes | `python -m wayfinder labels` |
| `labels/heldout_weights/` | 10 held-out weight vectors × 50 val/test targets = 500 runs, 500,000 samples (`w00.npz` … `w09.npz`) | yes | same |
| `labels/DATASHEET.md`, `labels_summary.json` | Dataset description and generation summary | yes | same |
| `user_study/trials.json`, `trials.csv` | 12 OD pairs × 3 profiles with both routes | yes | `python -m wayfinder study-prepare` |

Node indices in every file refer to `processed/graph.npz`; the splits and benchmark also store OSM ids and are
checked against the loaded graph.

Licences: OpenStreetMap (ODbL), ESA WorldCover 2021 (CC BY 4.0), Copernicus GLO-30 DEM.
