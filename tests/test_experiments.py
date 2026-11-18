import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pytest, datetime as dt
from tests.fixtures import synthetic_catalogue, make_omm, EPOCH
from experiments.common import chronological_split
from experiments.features import feature_frame, training_table, DEFAULT_FEATURES, history_targets
from experiments.regression import run_regression
from experiments.classification import run_classification
from experiments.feature_selection import run_feature_selection
from experiments.dimred import run_dimred
from experiments.clustering import run_kmeans, run_hierarchical
from experiments.mlp import MLP, gradient_check
from experiments import nn_fundamentals as nf
from orbital_engine.catalog import Catalogue, classify
from orbital_engine.backtest import backtest

CAT = synthetic_catalogue(n_payload=300, n_debris=250)


def test_chronological_split_no_leakage():
    df = pd.DataFrame({"t": np.random.default_rng(0).permutation(100), "v": range(100)})
    tr, va, te = chronological_split(df, "t")
    assert tr.t.max() < va.t.min() <= va.t.max() < te.t.min() and len(tr) + len(va) + len(te) == 100


def test_classify_rules():
    assert classify("X", "cosmos-2251-debris") == ("DEBRIS", "celestrak_group")
    assert classify("FOO R/B", "active") == ("ROCKET_BODY", "name_heuristic")
    assert classify("STARLINK-1", "active")[0] == "PAYLOAD"


def test_training_table_falls_back_honestly_without_history():
    df, mode, targets, label, note = training_table(CAT, None)
    assert mode == "cross_sectional" and "NOT future forecasting" in note


def test_regression_runs_all_models_and_reports_baseline():
    df, mode, targets, *_ = training_table(CAT)
    r = run_regression(df, DEFAULT_FEATURES, targets[0], models=["multiple_linear", "ridge", "random_forest", "knn"])
    assert {m["model"] for m in r["models"]} == {"multiple_linear", "ridge", "random_forest", "knn"}
    assert r["train_epoch_max"] <= r["test_epoch_min"] and "baseline_mean" in r
    assert all("rmse" in m["test"] for m in r["models"])


def test_regression_requires_enough_rows():
    df, _, targets, *_ = training_table(CAT)
    assert "error" in run_regression(df.head(20), DEFAULT_FEATURES, targets[0])


def test_classification_and_missing_class_error():
    f = feature_frame(CAT)
    r = run_classification(f, DEFAULT_FEATURES); assert r["models"][0]["roc_auc"] > 0.9
    assert "error" in run_classification(f[f.obj_class == "PAYLOAD"], DEFAULT_FEATURES)


def test_feature_selection_and_dimred_and_clustering():
    df, _, targets, *_ = training_table(CAT)
    fs = run_feature_selection(df, DEFAULT_FEATURES, targets[0]); assert len(fs["table"]) == len(DEFAULT_FEATURES)
    f = feature_frame(CAT)
    d = run_dimred(f, DEFAULT_FEATURES); assert set(d["methods"]) >= {"PCA", "ICA", "LDA"}
    k = run_kmeans(f); assert 2 <= k["optimal_k_silhouette"] <= 10 and len(k["sweep"]) == 9
    h = run_hierarchical(f); assert set(h["linkages"]) == {"ward", "complete", "average"} and h["dendrogram"]["icoord"]


def test_neuron_and_activations():
    r = nf.neuron([1, 2], [0.5, -1], 0.5, "sigmoid")
    assert r["net_input"] == pytest.approx(-1.0) and r["output"] == pytest.approx(1 / (1 + np.e))
    assert nf.relu([-1, 2]).tolist() == [0, 2] and nf.binary_step([-1, 0]).tolist() == [0, 1]
    assert nf.softmax([1, 1, 1]).sum() == pytest.approx(1) and nf.bipolar_sigmoid(0) == 0
    assert nf.leaky_relu(-2, 0.1) == pytest.approx(-0.2)
    with pytest.raises(ValueError): nf.neuron([1], [1, 2], 0, "relu")


@pytest.mark.parametrize("hidden,task,out", [("tanh", "regression", 1), ("sigmoid", "classification", 1),
                                              ("relu", "classification", 3)])
def test_mlp_gradient_check(hidden, task, out):
    rng = np.random.default_rng(0); X = rng.normal(size=(20, 4))
    y = rng.normal(size=20) if task == "regression" else rng.integers(0, max(2, out), 20)
    net = MLP([4, 5, out], hidden, task, seed=1)
    assert gradient_check(net, X, y) < 1e-4


def test_mlp_learns_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float); y = np.array([0, 1, 1, 0])
    net = MLP([2, 8, 1], "tanh", "classification", lr=0.5, seed=0)
    net.fit(X, y, epochs=2000, batch_size=4)
    assert (net.predict(X) == y).all()


def test_history_targets_and_backtest():
    rows = []
    for k in range(4):
        e = EPOCH + dt.timedelta(days=k)
        o = make_omm(1, 600 - 2 * k, epoch=e, ma=(k * 5.0) % 360)
        rows.append({"norad_id": 1, "epoch_utc": e.replace(tzinfo=None), "raw_omm": o, "mean_motion": o["MEAN_MOTION"],
                     "eccentricity": o["ECCENTRICITY"], "inclination": 53.0, "raan": 0.0, "arg_perigee": 0.0,
                     "mean_anomaly": o["MEAN_ANOMALY"], "bstar": 1e-4, "mean_motion_dot": 1e-5})
    h = history_targets(pd.DataFrame(rows))
    assert len(h) == 3 and (h["d_altitude_km_per_day"] < 0).all()      # altitude lowered 2 km/day by construction
    bt = backtest(rows); assert bt["n_pairs"] == 3 and all(p["position_error_km"] > 0 for p in bt["pairs"])
