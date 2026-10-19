import numpy as np
import pytest
import torch

from wayfinder.learning.inference import HeuristicModel, LearnedHeuristic
from wayfinder.learning.labels import generate_labels, split_mask
from wayfinder.learning.model import FeatureBuilder, HeuristicNet, load_checkpoint, save_checkpoint
from wayfinder.learning.node2vec import random_walks, train_node2vec, undirected_neighbours
from wayfinder.learning.train import LabelTensors, label_metrics, train_one
from wayfinder.learning.weights import lambda_weights, make_weight_sets
from wayfinder.search import SearchSpace, TunedClassicalHeuristic, admissible_heuristic, dijkstra, focal_search


def test_weight_sets_frozen_and_distinct():
    a = make_weight_sets(40, 10, seed=7)
    b = make_weight_sets(40, 10, seed=7)
    assert np.array_equal(a["train"], b["train"]) and np.array_equal(a["heldout"], b["heldout"])
    rows = {tuple(r) for r in np.vstack([a["train"], a["heldout"]])}
    assert len(rows) == 50
    assert a["train"].min() >= 0 and a["train"].max() <= 1


def test_lambda_family():
    assert lambda_weights(0).tolist() == [1, 0, 0, 0, 0]
    assert lambda_weights(0.8).sum() == pytest.approx(1)


def test_random_walks_follow_edges(grid):
    NB, deg = undirected_neighbours(grid.tail, grid.head, grid.n)
    walks = random_walks(NB, deg, 12, 2, p=1.0, q=0.5, rng=np.random.default_rng(0))
    assert walks.shape == (2 * grid.n, 12)
    pairs = [(int(a), int(b)) for a, b in zip(grid.tail, grid.head)]
    adj = set(pairs) | {(b, a) for a, b in pairs}
    for w in walks[:50]:
        for a, b in zip(w[:-1], w[1:]):
            assert (int(a), int(b)) in adj


def test_node2vec_embeds_neighbours_closer(grid):
    emb, info = train_node2vec(grid.tail, grid.head, grid.n, dim=8, walk_length=10, walks_per_node=5,
                               window=3, epochs=2, batch_size=512, seed=0, verbose=False)
    assert emb.shape == (grid.n, 8)
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    nb = (e[grid.tail] * e[grid.head]).sum(1).mean()
    rng = np.random.default_rng(0)
    a, b = rng.integers(grid.n, size=500), rng.integers(grid.n, size=500)
    assert nb > (e[a] * e[b]).sum(1).mean()


def test_ratio_never_below_one():
    net = HeuristicNet(4, [16])
    x = torch.randn(1000, 2 * 4 + 5 + 3) * 50
    assert torch.all(net(x) >= 1.0)


@pytest.fixture(scope="module")
def trained(net, tmp_path_factory):
    W = make_weight_sets(4, 2, seed=1)
    targets = np.arange(0, net.rg.n, 4)
    labels = generate_labels(net, W["train"], targets, nodes_per_run=40, seed=0, log_every=0)
    assert int(labels["admissibility_violations"]) == 0
    assert np.all(labels["h_star"] >= labels["h_adm"] - 1e-9)
    emb = np.random.default_rng(0).normal(size=(net.rg.n, 6)).astype(np.float32)
    fb = FeatureBuilder(emb, net.rg.x, net.rg.y)
    tr = LabelTensors(labels, split_mask(labels, targets[::2]))
    va = LabelTensors(labels, split_mask(labels, targets[1::2]))
    path = tmp_path_factory.mktemp("models") / "h.pt"
    model, log, best = train_one(fb, tr, va, [32, 32], 3e-3, epochs=4, batch_size=256, seed=0,
                                 checkpoint=path, verbose=False)
    return {"labels": labels, "emb": emb, "fb": fb, "model": model, "path": path, "val": va, "log": log, "W": W}


def test_training_reduces_validation_error(trained):
    log = trained["log"]
    assert log["val_mse_log"].iloc[-1] < log["val_mse_log"].iloc[0] or log["val_mse_log"].min() < 0.5
    m = label_metrics(trained["model"], trained["fb"], trained["val"])
    assert 0 <= m["overestimate_rate"] <= 1 and m["informedness_adm"] <= 1 + 1e-9


def test_checkpoint_roundtrip(trained, tmp_path):
    p = tmp_path / "c.pt"
    save_checkpoint(p, trained["model"], trained["fb"], {"x": 1})
    net2, ck = load_checkpoint(p)
    x = torch.randn(10, trained["model"].mlp[0].in_features)
    assert torch.allclose(net2(x), trained["model"](x))
    assert ck["meta"]["x"] == 1


def test_numpy_inference_matches_torch(net, trained):
    hm = HeuristicModel.from_checkpoint(trained["path"], trained["emb"], net.rg.x, net.rg.y)
    net_t, _ = load_checkpoint(trained["path"])
    ck = torch.load(trained["path"], weights_only=False)
    fb = FeatureBuilder(trained["emb"], net.rg.x, net.rg.y, ck["emb_mean"], ck["emb_std"])
    w = np.array([0.3, 0.2, 0.5, 0.1, 0.9])
    nodes = np.arange(net.rg.n)
    t = 5
    r_np = hm.ratio(nodes, t, w)
    with torch.no_grad():
        x = fb.build(torch.from_numpy(nodes), torch.full((len(nodes),), t), torch.tensor(w).repeat(len(nodes), 1))
        r_t = net_t(x)
    assert np.allclose(r_np, r_t.numpy(), rtol=1e-4)
    assert np.all(r_np >= 1.0)


def test_lazy_and_full_learned_heuristic_agree_and_focal_bound_holds(net, trained):
    hm = HeuristicModel.from_checkpoint(trained["path"], trained["emb"], net.rg.x, net.rg.y)
    w = np.array([0.1, 0.6, 0.4, 0.8, 0.3])
    t = 11
    full = LearnedHeuristic(hm, net.cm, t, w, "full")
    lazy = LearnedHeuristic(hm, net.cm, t, w, "lazy")
    tile = LearnedHeuristic(hm, net.cm, t, w, "tile")
    states = list(range(0, net.tg.m, 3))
    assert np.allclose(full.evaluate(states), lazy.evaluate(states), rtol=1e-5)     # float32 batching
    assert np.allclose(full.evaluate(states), tile.evaluate(states), rtol=1e-5)
    assert sum(len(m) for m in hm.tile_members) == net.rg.n
    assert full.values[t] == 0.0
    assert np.all(full.values >= net.cm.h_adm(t, w) - 1e-12)          # never less informed than h_adm
    sp = SearchSpace.build(net.cm, w)
    for s in range(0, net.rg.n, 9):
        opt = dijkstra(sp, s, t).cost
        for W in (1.0, 1.5, 2.0):
            r = focal_search(sp, s, t, admissible_heuristic(net.cm, t, w), LearnedHeuristic(hm, net.cm, t, w), W)
            assert r.cost <= W * opt * (1 + 1e-12)


def test_tuned_classical_fit_recovers_coefficients():
    rng = np.random.default_rng(0)
    W = rng.uniform(size=(500, 5))
    coef = np.array([0.5, 0.1, -0.2, 0.3, 0.0, 0.4])
    y = coef[0] + W @ coef[1:]
    tuned = TunedClassicalHeuristic.fit(W, y)
    assert np.allclose(tuned.coef, coef, atol=1e-9)
    assert tuned.kappa(np.zeros(5)) == pytest.approx(np.exp(0.5))
    assert TunedClassicalHeuristic(np.array([-5, 0, 0, 0, 0, 0])).kappa(np.zeros(5)) == 1.0     # never below 1


def test_label_shards_roundtrip_and_checksum(trained, tmp_path):
    from wayfinder.learning.labels import COLUMNS, load_labels, save_labels
    labels = trained["labels"]
    out = tmp_path / "train_weights"
    save_labels(labels, out, {"weight_set": "train"})
    assert len(list(out.glob("w*.npz"))) == len(labels["weights"])
    back = load_labels(out)
    for c in COLUMNS:
        assert np.array_equal(back[c], labels[c]) and back[c].dtype == labels[c].dtype
    assert np.array_equal(back["weights"], labels["weights"])
    assert int(back["admissibility_violations"]) == 0 and back["meta"]["weight_set"] == "train"
    shard = out / "w00.npz"
    shard.write_bytes(shard.read_bytes()[:-1] + b"x")          # tamper with one byte
    with pytest.raises(ValueError):
        load_labels(out)
