import numpy as np, pytest
try:
    import torch
except Exception as exc:          # missing OR broken (e.g. CUDA libs absent) torch install
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)
from experiments.deep_learning import (windows_in_range, forecast_experiment, build, ARCHS, autoencoder_anomaly)


def sine(T=900):
    t = np.arange(T); return np.stack([np.sin(2 * np.pi * t / 60) * 100 + 500, np.cos(2 * np.pi * t / 60)], 1)


def test_windows_never_leak_targets_across_split():
    s = np.arange(200, dtype=float)[:, None]
    X, Y, T0 = windows_in_range(s, 10, 5, 100, 150)
    assert T0.min() >= 100 and (T0 + 5).max() <= 150
    assert Y.min() >= 100 and Y.max() < 150           # targets strictly in region


@pytest.mark.parametrize("name", ARCHS)
def test_all_architectures_forward_shapes(name):
    m = build(name, 2, 48, 12); out = m(torch.randn(4, 48, 2)); assert out.shape == (4, 12)


def test_forecast_beats_persistence_on_periodic_signal_and_reproducible():
    a = forecast_experiment(sine(), models=["gru", "mlp"], L=48, H=12, epochs=15, seed=3)
    b = forecast_experiment(sine(), models=["gru"], L=48, H=12, epochs=15, seed=3)
    assert a["models"]["gru"]["mae"] < a["models"]["persistence"]["mae"]
    assert a["models"]["gru"]["mae"] == pytest.approx(b["models"]["gru"]["mae"], rel=1e-6)   # fixed seed
    assert a["split_index"]["train_end"] < a["split_index"]["val_end"] < a["split_index"]["n"]


def test_autoencoder_flags_shifted_distribution():
    rng = np.random.default_rng(0)
    Xn = rng.normal(0, 1, (400, 6)); Xt = np.vstack([rng.normal(0, 1, (100, 6)), rng.normal(4, 1, (50, 6))])
    y = np.r_[np.zeros(100), np.ones(50)]
    r = autoencoder_anomaly(Xn, Xt, y, epochs=120)
    assert r["roc_auc"] > 0.9
