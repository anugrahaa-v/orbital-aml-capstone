"""Deep-learning models (PyTorch): MLP, 1D-CNN, LSTM, GRU, CNN-LSTM, Transformer, Autoencoder.
Pure torch+numpy so it is testable in isolation. Preprocessing (scaling) is fit on TRAIN only."""
from __future__ import annotations
import math, time
import numpy as np
import torch
import torch.nn as nn

ARCHS = ["mlp", "cnn1d", "lstm", "gru", "cnn_lstm", "transformer"]


def set_seed(seed: int):
    np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class MLPNet(nn.Module):
    def __init__(s, c, L, H, w=96):
        super().__init__(); s.f = nn.Sequential(nn.Flatten(), nn.Linear(c * L, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU(), nn.Linear(w, H))
    def forward(s, x): return s.f(x)


class CNN1D(nn.Module):
    def __init__(s, c, L, H, w=32):
        super().__init__()
        s.f = nn.Sequential(nn.Conv1d(c, w, 5, padding=2), nn.ReLU(), nn.Conv1d(w, w, 5, padding=2), nn.ReLU(),
                            nn.AdaptiveAvgPool1d(4), nn.Flatten(), nn.Linear(w * 4, H))
    def forward(s, x): return s.f(x.transpose(1, 2))


class RNNNet(nn.Module):
    def __init__(s, c, L, H, kind="lstm", w=48):
        super().__init__(); s.r = (nn.LSTM if kind == "lstm" else nn.GRU)(c, w, batch_first=True, num_layers=1); s.o = nn.Linear(w, H)
    def forward(s, x): return s.o(s.r(x)[0][:, -1])


class CNNLSTM(nn.Module):
    def __init__(s, c, L, H, w=32):
        super().__init__(); s.c = nn.Sequential(nn.Conv1d(c, w, 3, padding=1), nn.ReLU()); s.r = nn.LSTM(w, 48, batch_first=True); s.o = nn.Linear(48, H)
    def forward(s, x): return s.o(s.r(s.c(x.transpose(1, 2)).transpose(1, 2))[0][:, -1])


class TransformerNet(nn.Module):
    def __init__(s, c, L, H, d=32):
        super().__init__(); s.i = nn.Linear(c, d)
        pe = torch.zeros(L, d); pos = torch.arange(L).unsqueeze(1); div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div); s.register_buffer("pe", pe)
        s.e = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 64, 0.0, batch_first=True), 2); s.o = nn.Linear(d, H)
    def forward(s, x): return s.o(s.e(s.i(x) + s.pe).mean(1))


def build(name, c, L, H):
    return {"mlp": lambda: MLPNet(c, L, H), "cnn1d": lambda: CNN1D(c, L, H), "lstm": lambda: RNNNet(c, L, H, "lstm"),
            "gru": lambda: RNNNet(c, L, H, "gru"), "cnn_lstm": lambda: CNNLSTM(c, L, H),
            "transformer": lambda: TransformerNet(c, L, H)}[name]()


def windows_in_range(series, L, H, lo, hi, stride=1, target_col=0):
    """Targets lie fully inside [lo, hi); inputs may reach before lo (past data = known at forecast time)."""
    X, Y, T0 = [], [], []
    for s in range(max(lo, L), hi - H + 1, stride):
        X.append(series[s - L:s]); Y.append(series[s:s + H, target_col]); T0.append(s)
    return (np.array(X, np.float32), np.array(Y, np.float32), np.array(T0, int))


def fit_predict(name, Xtr, Ytr, Xva, Yva, Xte, epochs=25, lr=2e-3, batch=128, seed=0, patience=6):
    set_seed(seed)
    c, L, H = Xtr.shape[2], Xtr.shape[1], Ytr.shape[1]
    m = build(name, c, L, H); opt = torch.optim.Adam(m.parameters(), lr=lr); lossf = nn.MSELoss()
    tr = torch.utils.data.TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr))
    dl = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True, generator=torch.Generator().manual_seed(seed))
    hist, best, best_state, bad = {"train_loss": [], "val_loss": []}, float("inf"), None, 0
    xv, yv = torch.tensor(Xva), torch.tensor(Yva)
    t0 = time.time()
    for ep in range(epochs):
        m.train(); tot = 0.0
        for xb, yb in dl:
            opt.zero_grad(); l = lossf(m(xb), yb); l.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); tot += l.item() * len(xb)
        m.eval()
        with torch.no_grad(): vl = lossf(m(xv), yv).item()
        hist["train_loss"].append(tot / len(tr)); hist["val_loss"].append(vl)
        if vl < best - 1e-9: best, bad, best_state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break          # early stopping on validation (chronologically later than train)
    if best_state: m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad(): pred = m(torch.tensor(Xte)).numpy()
    return pred, hist, {"params": sum(p.numel() for p in m.parameters()), "epochs_run": len(hist["train_loss"]),
                        "seconds": round(time.time() - t0, 2)}


def forecast_experiment(series: np.ndarray, models=None, L=48, H=12, stride=2, epochs=25, seed=0, fracs=(0.7, 0.1, 0.2)) -> dict:
    """series (T,C), column 0 = target. Chronological split by TARGET position; scaler fit on train region only."""
    T = len(series); i1, i2 = int(T * fracs[0]), int(T * (fracs[0] + fracs[1]))
    mu, sd = series[:i1].mean(0), series[:i1].std(0) + 1e-9
    z = ((series - mu) / sd).astype(np.float32)
    Xtr, Ytr, _ = windows_in_range(z, L, H, 0, i1, stride)
    Xva, Yva, _ = windows_in_range(z, L, H, i1, i2, stride)
    Xte, Yte, T0 = windows_in_range(z, L, H, i2, T, 1)
    if min(len(Xtr), len(Xva), len(Xte)) < 5:
        return {"error": "series too short for the requested window/horizon"}
    unz = lambda a: a * sd[0] + mu[0]
    yte = unz(Yte)
    def metr(p):
        e = unz(p) - yte; return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                                  "mae_by_horizon": np.abs(e).mean(0).tolist()}
    res = {"split_index": {"train_end": i1, "val_end": i2, "n": T}, "L": L, "H": H, "seed": seed, "models": {},
           "n_windows": [len(Xtr), len(Xva), len(Xte)], "origins": T0.tolist()}
    res["models"]["persistence"] = {**metr(np.repeat(Xte[:, -1:, 0], H, 1)), "kind": "baseline"}
    from sklearn.linear_model import Ridge
    ar = Ridge(alpha=1.0).fit(Xtr[:, :, 0], Ytr)
    res["models"]["linear_ar"] = {**metr(ar.predict(Xte[:, :, 0])), "kind": "baseline"}
    show = np.linspace(0, len(Xte) - 1, min(6, len(Xte))).astype(int)
    for name in (models or ARCHS):
        p, h, info = fit_predict(name, Xtr, Ytr, Xva, Yva, Xte, epochs=epochs, seed=seed)
        res["models"][name] = {**metr(p), **info, "history": h, "kind": "deep",
                               "examples": {"idx": show.tolist(), "pred": unz(p[show]).tolist()}}
    res["examples_truth"] = yte[show].tolist(); res["examples_origin_index"] = T0[show].tolist()
    res["series_test_target"] = unz(z[i2:, 0]).tolist()
    return res


class AutoEncoder(nn.Module):
    def __init__(s, d, latent=3, w=16):
        super().__init__(); s.e = nn.Sequential(nn.Linear(d, w), nn.ReLU(), nn.Linear(w, latent))
        s.d = nn.Sequential(nn.Linear(latent, w), nn.ReLU(), nn.Linear(w, d))
    def forward(s, x): return s.d(s.e(x))


def autoencoder_anomaly(Xnormal_train, Xtest, ytest_anom, epochs=80, seed=0, latent=3, pct=95) -> dict:
    """Train on NORMAL objects only (older epochs); score = reconstruction MSE; anomalies = label 1."""
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
    set_seed(seed)
    mu, sd = Xnormal_train.mean(0), Xnormal_train.std(0) + 1e-9
    xtr = torch.tensor(((Xnormal_train - mu) / sd).astype(np.float32)); xte = torch.tensor(((Xtest - mu) / sd).astype(np.float32))
    m = AutoEncoder(xtr.shape[1], latent); opt = torch.optim.Adam(m.parameters(), 3e-3); hist = []
    for _ in range(epochs):
        opt.zero_grad(); l = ((m(xtr) - xtr) ** 2).mean(); l.backward(); opt.step(); hist.append(l.item())
    m.eval()
    with torch.no_grad():
        etr = ((m(xtr) - xtr) ** 2).mean(1).numpy(); ete = ((m(xte) - xte) ** 2).mean(1).numpy()
    thr = float(np.percentile(etr, pct)); pred = (ete > thr).astype(int); y = np.asarray(ytest_anom).astype(int)
    out = {"threshold": thr, "train_loss": hist, "threshold_rule": f"{pct}th percentile of training reconstruction error",
           "score_normal_mean": float(ete[y == 0].mean()) if (y == 0).any() else None,
           "score_anomaly_mean": float(ete[y == 1].mean()) if (y == 1).any() else None}
    if len(set(y)) == 2:
        out.update(roc_auc=float(roc_auc_score(y, ete)), precision=float(precision_score(y, pred, zero_division=0)),
                   recall=float(recall_score(y, pred, zero_division=0)), f1=float(f1_score(y, pred, zero_division=0)),
                   confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist())
    out["scores_sample"] = {"score": ete[:600].tolist(), "label": y[:600].tolist()}
    return out
