"""ORBITALX AI API. Every payload carries data-lineage labels (observed / derived / simulated / ml)."""
from __future__ import annotations
import base64, datetime as dt, json, os
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sgp4.propagation import gstime

from backend.jobs import Jobs
from collision_engine.screening import (closest_approach, distance_series, screen_primary, ScreeningConfig,
                                        pc_or_unavailable)
from database import store
from experiments import common
from experiments.features import feature_frame, training_table, DEFAULT_FEATURES
from orbital_engine.catalog import Catalogue
from orbital_engine.elements import derive
from orbital_engine.propagate import to_jd
from orbital_engine.frames import teme_to_ecef_many, ecef_to_geodetic_many

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
LABELS = {"observed": "OBSERVED: published GP element sets (CelesTrak)", "derived": "DERIVED from observed elements",
          "simulated": "SIMULATED: SGP4 propagation / simulator output", "ml": "ML PREDICTION"}


class State:
    def __init__(self):
        self.cat: Catalogue | None = None; self.eng = None; self.jobs = Jobs(); self.residual = None
        self.data_dir = DATA_DIR
    def catalogue(self) -> Catalogue:
        if self.cat is None or len(self.cat) == 0:      # re-check the cache so data ingested after start-up is picked up
            self.cat = Catalogue.from_cache(self.data_dir)
        if len(self.cat) == 0:
            raise HTTPException(409, "No catalogue ingested yet. Run: python -m backend.ingest_cli --groups active "
                                     "cosmos-2251-debris   (needs internet), then reload.")
        return self.cat
    def engine(self):
        if self.eng is None: self.eng = store.get_engine(None, self.data_dir)
        return self.eng


class SafeJSONResponse(JSONResponse):
    """Starlette's default encoder rejects NaN/inf (HTTP 500). SGP4 failures and degenerate metrics can produce them,
    so every response is sanitised: NaN/inf -> null."""
    def render(self, content) -> bytes:
        return json.dumps(store._clean(content), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


S = State()
app = FastAPI(title="ORBITALX AI", version="0.1", default_response_class=SafeJSONResponse)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _now(): return dt.datetime.now(dt.timezone.utc)


def _t(s: str | None) -> dt.datetime:
    if not s: return _now()
    d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def _obj(nid: int):
    cat = S.catalogue()
    if not cat.has(nid): raise HTTPException(404, f"NORAD {nid} not in catalogue")
    return cat.get(nid)


# ------------------------------------------------------------------ status / data
@app.get("/api/status")
def status():
    try:
        cat = S.catalogue(); n = len(cat)
    except HTTPException:
        cat, n = None, 0
    try: tor = __import__("torch").__version__
    except Exception: tor = None
    out = {"catalogue_loaded": n > 0, "n_objects": n, "torch": tor, "labels": LABELS, "data_dir": str(S.data_dir)}
    if cat:
        from collections import Counter
        out.update(classes=dict(Counter(o.obj_class for o in cat.objects)), groups=dict(Counter(o.group for o in cat.objects)),
                   provenance=cat.provenance[-8:], rejected_records=cat.rejected,
                   epoch_range=[min(o.epoch_utc for o in cat.objects).isoformat(), max(o.epoch_utc for o in cat.objects).isoformat()])
        try: out["history"] = store.history_snapshot_counts(S.engine())
        except Exception: out["history"] = None
    return out


class IngestReq(BaseModel):
    groups: list[str] = ["active", "cosmos-2251-debris", "iridium-33-debris", "fengyun-1c-debris", "cosmos-1408-debris"]


@app.post("/api/ingest")
def ingest(req: IngestReq):
    from backend.ingestion.pipeline import ingest as run
    def job():
        out = run(req.groups, S.data_dir, None, True); S.cat = None; return out
    return {"job": S.jobs.submit("ingest", job)}


@app.get("/api/jobs/{jid}")
def job(jid: str):
    j = S.jobs.get(jid)
    if not j: raise HTTPException(404, "unknown job")
    return {k: v for k, v in j.items() if k != "trace" or j["status"] == "error"}


@app.get("/api/catalogue")
def catalogue(q: str | None = None, cls: str | None = None, alt_min: float | None = None, alt_max: float | None = None,
              group: str | None = None, limit: int = Query(100, le=2000)):
    objs = S.catalogue().filter(q, cls, alt_min, alt_max, group, limit)
    return {"label": LABELS["observed"], "count": len(objs), "objects": [o.summary() for o in objs]}


@app.get("/api/object/{nid}")
def object_detail(nid: int):
    o = _obj(nid); d = derive(float(o.fields["MEAN_MOTION"]), float(o.fields["ECCENTRICITY"]))
    snaps = 0
    try: snaps = len(store.get_history(S.engine(), nid))
    except Exception: pass
    return {**o.summary(), "observed_elements": o.fields, "observed_label": LABELS["observed"],
            "derived": {"semi_major_axis_km": d.semi_major_axis_km, "period_min": d.period_min,
                        "perigee_alt_km": d.perigee_alt_km, "apogee_alt_km": d.apogee_alt_km, "note": d.note},
            "derived_label": LABELS["derived"], "snapshots_stored": snaps,
            "epoch_age_days": (_now() - o.epoch_utc).total_seconds() / 86400}


@app.get("/api/orbit/{nid}/path")
def orbit_path(nid: int, start: str | None = None, points: int = 240):
    o = _obj(nid); t0 = _t(start); period = o.period_min * 60
    times = [t0 + dt.timedelta(seconds=period * i / points) for i in range(points + 1)]
    err, r, v = S.catalogue().positions([o], times)
    ok = err[0] == 0
    jd, fr = zip(*(to_jd(t) for t in times))
    lat, lon, alt = ecef_to_geodetic_many(teme_to_ecef_many(r[0], np.array(jd) + np.array(fr)))
    return {"label": LABELS["simulated"], "frame": "TEME", "units": "km", "times": [t.isoformat() for t in times],
            "r_teme": np.where(ok[:, None], r[0], np.nan).tolist(), "lat": lat.tolist(), "lon": lon.tolist(),
            "alt_km": alt.tolist(), "valid": bool(ok.all())}


@app.get("/api/positions")
def positions(start: str | None = None, duration_min: float = 90, step_s: float = 60, cls: str | None = None,
              group: str | None = None, alt_min: float | None = None, alt_max: float | None = None,
              limit: int = Query(3000, le=8000), q: str | None = None):
    cat = S.catalogue(); objs = cat.filter(q, cls, alt_min, alt_max, group)
    if len(objs) > limit:
        objs = [objs[i] for i in np.linspace(0, len(objs) - 1, limit).astype(int)]
    t0 = _t(start); n = int(duration_min * 60 / step_s) + 1
    times = [t0 + dt.timedelta(seconds=i * step_s) for i in range(n)]
    err, r, _ = cat.positions(objs, times)
    good = (err == 0).all(axis=1); objs = [o for o, g in zip(objs, good) if g]; r = r[good]
    data = np.ascontiguousarray(np.transpose(r, (1, 0, 2)).astype(np.float32))     # (T,N,3)
    jds = [to_jd(t) for t in times]
    return {"label": LABELS["simulated"], "frame": "TEME", "units": "km", "start": t0.isoformat(), "step_s": step_s,
            "n_frames": n, "n_objects": len(objs), "ids": [o.norad_id for o in objs], "names": [o.name for o in objs],
            "classes": [o.obj_class for o in objs], "gmst_rad": [float(gstime(a + b)) for a, b in jds],
            "data_b64": base64.b64encode(data.tobytes()).decode()}


@app.get("/api/ground-track/{nid}")
def ground_track(nid: int, start: str | None = None, hours: float = 3, step_s: float = 60):
    o = _obj(nid); t0 = _t(start); n = int(hours * 3600 / step_s) + 1
    times = [t0 + dt.timedelta(seconds=i * step_s) for i in range(n)]
    err, r, _ = S.catalogue().positions([o], times)
    jd = np.array([sum(to_jd(t)) for t in times])
    lat, lon, alt = ecef_to_geodetic_many(teme_to_ecef_many(r[0], jd))
    return {"label": LABELS["simulated"], "lat": lat.tolist(), "lon": lon.tolist(), "alt_km": alt.tolist(),
            "times": [t.isoformat() for t in times]}


# ------------------------------------------------------------------ collision
@app.get("/api/collision/closest")
def collision_closest(a: int, b: int, start: str | None = None, hours: float = 24, window_min: float = 10):
    A, B = _obj(a), _obj(b); t0 = _t(start)
    c = closest_approach(A, B, t0, hours * 3600, 10.0)
    if c is None: raise HTTPException(422, "SGP4 failed for this pair in the window")
    offs, ra, rb, dist, rs, ok = distance_series(A, B, c.tca_utc, window_min * 60, 5.0)
    return {"conjunction": c.to_dict(), "series": {"offset_s": offs.tolist(), "distance_km": np.nan_to_num(dist, nan=-1).tolist(),
            "rel_speed_kms": np.nan_to_num(rs, nan=-1).tolist(), "r_a_teme": ra.tolist(), "r_b_teme": rb.tolist()},
            "primary": A.summary(), "secondary": B.summary(), "label": LABELS["simulated"],
            "warning": "Screening result from public mean elements; not an operational conjunction assessment."}


class ScreenReq(BaseModel):
    primary: int; start: str | None = None; threshold_km: float = Field(10.0, gt=0, le=200)
    horizon_h: float = Field(24.0, gt=0, le=72); coarse_step_s: float = Field(30.0, ge=5, le=120)
    max_results: int = 100


@app.post("/api/collision/screen")
def collision_screen(req: ScreenReq):
    cat, p = S.catalogue(), _obj(req.primary)
    cfg = ScreeningConfig(req.threshold_km, req.horizon_h, req.coarse_step_s, max_results=req.max_results)
    t0 = _t(req.start); res = screen_primary(cat, p, t0, cfg)
    try:
        store.save_conjunctions(S.engine(), _now(), [{"primary_id": c["primary_id"], "secondary_id": c["secondary_id"],
            "tca_utc": dt.datetime.fromisoformat(c["tca_utc"]), "miss_distance_km": c["miss_distance_km"],
            "relative_speed_kms": c["relative_speed_kms"], "threshold_km": cfg.threshold_km, "category": c["category"],
            "pc_available": False, "pc": None, "config": res["config"]} for c in res["conjunctions"]])
    except Exception: pass
    names = {c["secondary_id"]: cat.get(c["secondary_id"]) for c in res["conjunctions"]}
    for c in res["conjunctions"]:
        o = names[c["secondary_id"]]; c["secondary_name"] = o.name; c["secondary_class"] = o.obj_class
    return {**res, "primary": p.summary(), "label": LABELS["simulated"]}


class PcReq(BaseModel):
    a: int; b: int; start: str | None = None; hours: float = 24
    cov_a_km2: list[list[float]] | None = None; cov_b_km2: list[list[float]] | None = None; hbr_km: float | None = None


@app.post("/api/collision/pc")
def collision_pc(req: PcReq):
    A, B = _obj(req.a), _obj(req.b)
    c = closest_approach(A, B, _t(req.start), req.hours * 3600, 10.0)
    if c is None: raise HTTPException(422, "SGP4 failed")
    return {"conjunction": c.to_dict(), "pc": pc_or_unavailable(c, req.cov_a_km2, req.cov_b_km2, req.hbr_km)}


# ------------------------------------------------------------------ environment analytics (catalogue counts, not flux)
@app.get("/api/environment/stats")
def env_stats(bin_km: float = 50.0):
    cat = S.catalogue()
    mean_alt = np.array([0.5 * (o.perigee_km + o.apogee_km) for o in cat.objects]); cls = np.array([o.obj_class for o in cat.objects])
    inc = np.array([o.inc_deg for o in cat.objects]); ecc = np.array([float(o.fields["ECCENTRICITY"]) for o in cat.objects])
    edges = np.arange(0, 2100 + bin_km, bin_km)
    out = {"label": LABELS["observed"] + " (catalogue counts, NOT debris flux)", "bin_km": bin_km, "classes": {}}
    shell_vol = 4 / 3 * np.pi * ((6378.137 + edges[1:]) ** 3 - (6378.137 + edges[:-1]) ** 3)
    for c in sorted(set(cls)):
        m = cls == c; h, _ = np.histogram(mean_alt[m], edges)
        out["classes"][c] = {"count": int(m.sum()), "alt_hist": h.tolist(), "density_per_1e9_km3": (h / shell_vol * 1e9).tolist(),
                             "inc_hist": np.histogram(inc[m], np.arange(0, 181, 5))[0].tolist(), "ecc_median": float(np.median(ecc[m]))}
    out["alt_edges_km"] = edges[:-1].tolist(); out["inc_edges_deg"] = np.arange(0, 180, 5).tolist()
    regions = {"LEO (<2000 km)": mean_alt < 2000, "MEO": (mean_alt >= 2000) & (mean_alt < 35000),
               "GEO belt (35000-36500)": (mean_alt >= 35000) & (mean_alt <= 36500), "HEO/other": mean_alt > 36500}
    out["regions"] = {k: {c: int(((cls == c) & m).sum()) for c in sorted(set(cls))} for k, m in regions.items()}
    return out


# ------------------------------------------------------------------ history / backtest / forecasting
@app.get("/api/history/{nid}")
def history(nid: int):
    from orbital_engine.backtest import backtest
    rows = store.get_history(S.engine(), nid)
    bt = backtest(rows) if len(rows) >= 2 else None
    return {"label": LABELS["observed"], "n_snapshots": len(rows), "epochs": [r["epoch_utc"].isoformat() for r in rows],
            "mean_motion": [r["mean_motion"] for r in rows], "backtest": bt,
            "note": None if len(rows) >= 2 else "Only one snapshot stored. Re-run ingestion on later days to accumulate real history."}


def _need_torch():
    try:
        import torch  # noqa: F401
    except Exception:
        raise HTTPException(503, "PyTorch is not installed (or failed to import). Install it with `pip install torch` and "
                                 "restart the API. Classical ML, screening, RL and the 3-D views work without it.")


class ForecastReq(BaseModel):
    norad_id: int; days: float = Field(6.0, ge=1, le=20); step_s: float = Field(120.0, ge=30, le=600)
    models: list[str] = ["gru", "lstm", "mlp"]; L: int = 48; H: int = 12; epochs: int = Field(20, ge=1, le=100); seed: int = 42
    start: str | None = None


@app.post("/api/forecast/run")
def forecast_run(req: ForecastReq):
    _need_torch()
    o = _obj(req.norad_id)
    def job():
        from orbital_engine.forecast_data import altitude_series
        from experiments.deep_learning import forecast_experiment
        ser = altitude_series(o, o.epoch_utc if not req.start else _t(req.start), req.days * 24, req.step_s)
        ok = ~np.isnan(ser["altitude_km"])
        if ok.mean() < 0.99: raise RuntimeError("SGP4 failed over part of the window (decayed/invalid element set)")
        X = np.stack([ser["altitude_km"], ser["speed_kms"]], 1)
        res = forecast_experiment(X, req.models, req.L, req.H, 2, req.epochs, req.seed)
        res.update(label=ser["label"], norad_id=req.norad_id, step_s=req.step_s, epoch_start=ser["times"][0].isoformat(),
                   note="SGP4 is the generator of this series, so it is the reference (exact by construction), not a competitor. "
                        "Models are learned surrogates; real SGP4-vs-truth comparison needs accumulated history (see backtest).")
        for name, m in res.get("models", {}).items():
            common.log_run(S.engine(), "deep_learning", name, "altitude_km", req.seed, {"L": req.L, "H": req.H, "epochs": req.epochs},
                           {"mae": m["mae"], "rmse": m["rmse"]}, {"norad_id": req.norad_id}, "simulated")
        return res
    return {"job": S.jobs.submit("forecast", job)}


@app.post("/api/dl/anomaly")
def dl_anomaly(seed: int = 42, epochs: int = 120):
    _need_torch()
    cat = S.catalogue()
    def job():
        from experiments.deep_learning import autoencoder_anomaly
        f = feature_frame(cat).dropna(subset=DEFAULT_FEATURES).sort_values("epoch")
        f = f[f.obj_class != "ROCKET_BODY"]
        cut = int(len(f) * 0.7); tr, te = f.iloc[:cut], f.iloc[cut:]
        ntr = tr[tr.obj_class == "PAYLOAD"][DEFAULT_FEATURES].to_numpy()
        r = autoencoder_anomaly(ntr, te[DEFAULT_FEATURES].to_numpy(), (te.obj_class == "DEBRIS").astype(int).to_numpy(), epochs, seed)
        r["note"] = "Autoencoder trained on older PAYLOAD objects only; 'anomaly' = DEBRIS-group label (group-derived, not ground truth)."
        r["label"] = LABELS["ml"]
        return r
    return {"job": S.jobs.submit("anomaly", job)}


# ------------------------------------------------------------------ classical ML
class MLReq(BaseModel):
    task: str = "regression"; target: str | None = None; features: list[str] = DEFAULT_FEATURES
    models: list[str] | None = None; seed: int = 42; k_max: int = 10; n_clusters: int | None = None


def _table():
    cat = S.catalogue()
    hist = None
    try: hist = store.history_frame(S.engine())
    except Exception: pass
    return cat, training_table(cat, hist)


@app.post("/api/ml/run")
def ml_run(req: MLReq):
    cat = S.catalogue()
    def job():
        from experiments import regression, classification, feature_selection, dimred, clustering
        _, (df, mode, targets, dlabel, note) = _table()
        tgt = req.target or targets[0]; prov = common.provenance(cat); ff = feature_frame(cat)
        if req.task == "regression":
            r = regression.run_regression(df, req.features, tgt, "epoch", req.seed, req.models)
            for m in r.get("models", []):
                if "test" in m: common.log_run(S.engine(), "regression", m["model"], tgt, req.seed, m["best_params"], m["test"], prov, dlabel)
        elif req.task == "classification":
            r = classification.run_classification(ff, req.features, req.seed)
            for m in r.get("models", []):
                common.log_run(S.engine(), "classification", m["model"], "DEBRIS_vs_PAYLOAD", req.seed, {}, {k: m[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}, prov, "derived labels")
        elif req.task == "feature_selection": r = feature_selection.run_feature_selection(df, req.features, tgt, "epoch", req.seed)
        elif req.task == "dimred": r = dimred.run_dimred(ff, req.features, seed=req.seed)
        elif req.task == "kmeans": r = clustering.run_kmeans(ff, None, 2, req.k_max, req.seed)
        elif req.task == "hierarchical": r = clustering.run_hierarchical(ff, None, req.n_clusters, req.seed)
        else: raise ValueError(f"unknown task {req.task}")
        r.update(mode=mode, targets_available=targets, data_label=dlabel, data_note=note, provenance=prov)
        return r
    return {"job": S.jobs.submit(req.task, job)}


# ------------------------------------------------------------------ neural-network fundamentals
class NeuronReq(BaseModel):
    inputs: list[float]; weights: list[float]; bias: float = 0.0; activation: str = "sigmoid"; alpha: float = 0.01


@app.post("/api/nn/neuron")
def nn_neuron(req: NeuronReq):
    from experiments import nn_fundamentals as nf
    try: return nf.neuron(req.inputs, req.weights, req.bias, req.activation, req.alpha)
    except ValueError as e: raise HTTPException(422, str(e))


@app.get("/api/nn/curve/{act}")
def nn_curve(act: str):
    from experiments import nn_fundamentals as nf
    if act not in nf.ACTIVATIONS: raise HTTPException(404, "unknown activation")
    return nf.curve(act)


class TrainReq(BaseModel):
    dataset: str = "xor"; hidden: list[int] = [4]; activation: str = "tanh"; lr: float = 0.3; epochs: int = Field(500, le=5000); seed: int = 0


@app.post("/api/nn/train")
def nn_train(req: TrainReq):
    from experiments.mlp import MLP
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    ys = {"xor": [0, 1, 1, 0], "and": [0, 0, 0, 1], "or": [0, 1, 1, 1]}
    if req.dataset not in ys: raise HTTPException(422, "dataset must be xor|and|or")
    y = np.array(ys[req.dataset]); net = MLP([2, *req.hidden, 1], req.activation, "classification", lr=req.lr, seed=req.seed)
    snaps = []
    hist = []
    for e in range(req.epochs):
        net.step(X, y); hist.append(net.loss(X, y))
        if e in (0, req.epochs // 2, req.epochs - 1): snaps.append({"epoch": e, "W": [w.tolist() for w in net.W], "b": [b.tolist() for b in net.b]})
    A, _ = net.forward(X)
    return {"loss": hist, "predictions": A[-1][:, 0].tolist(), "labels": y.tolist(), "snapshots": snaps,
            "accuracy": float((net.predict(X) == y).mean()), "label": "pure-NumPy MLP (educational, tiny dataset)"}


# ------------------------------------------------------------------ RL
class RLReq(BaseModel):
    policy: str = "mpc_explicit"; seed: int = 1; n_scenarios: int = Field(8, le=40); train_model: bool = True
    budget_ms: float = 2.0; dv_step_ms: float = 0.25


def _res(cfg, train: bool):
    from reinforcement_learning.model import ResidualModel
    if train and (S.residual is None or S.residual.cfg != cfg):
        m = ResidualModel(cfg); m.fit(n_episodes=90, epochs=600); S.residual = m
    return S.residual


@app.post("/api/rl/episode")
def rl_episode(req: RLReq):
    from reinforcement_learning.env import RLConfig, ConjunctionEnv
    from reinforcement_learning.run import run_episode, build_policies
    cfg = RLConfig(budget_ms=req.budget_ms, dv_step_ms=req.dv_step_ms)
    pols = build_policies(cfg, _res(cfg, req.train_model and req.policy.endswith("learned")), req.seed)
    if req.policy not in pols: raise HTTPException(422, f"policy must be one of {list(pols)}")
    ep = run_episode(ConjunctionEnv(cfg), pols[req.policy], req.seed)
    try: store.save_rl(S.engine(), common.uuid.uuid4().hex[:16], req.policy, req.seed, ep["final_separation_km"], ep["fuel_used_ms"], ep["total_reward"], cfg.__dict__, ep["trajectory"])
    except Exception: pass
    return {**ep, "config": cfg.__dict__, "label": LABELS["simulated"] + " (simulated encounter, not an operational recommendation)",
            "model_report": S.residual.report if S.residual else None}


@app.post("/api/rl/compare")
def rl_compare(req: RLReq):
    from reinforcement_learning.env import RLConfig
    from reinforcement_learning.run import compare_policies
    cfg = RLConfig(budget_ms=req.budget_ms, dv_step_ms=req.dv_step_ms)
    def job():
        r = compare_policies(cfg, req.n_scenarios, req.seed, _res(cfg, req.train_model))
        r["model_report"] = S.residual.report if S.residual else None
        for p in r["policies"]:
            common.log_run(S.engine(), "reinforcement_learning", p["policy"], "final_separation_km", req.seed, {"n": req.n_scenarios},
                           {k: p[k] for k in ("mean_final_separation_km", "success_rate", "mean_fuel_ms", "mean_reward")}, {"simulated": True}, "simulated")
        return r
    return {"job": S.jobs.submit("rl_compare", job)}


# ------------------------------------------------------------------ analytics
@app.get("/api/experiments")
def experiments(kind: str | None = None, limit: int = 300):
    rows = store.list_experiments(S.engine(), kind, limit)
    for r in rows: r["created_utc"] = r["created_utc"].isoformat()
    return {"experiments": rows}


@app.get("/api/analytics/summary")
def analytics():
    rows = store.list_experiments(S.engine(), None, 2000)
    by: dict = {}
    for r in rows:
        by.setdefault(r["kind"], []).append({"model": r["model_name"], "target": r["target"], "metrics": r["metrics"], "data_label": r["data_label"], "run_id": r["run_id"]})
    return {"n_runs": len(rows), "by_kind": by, "history": store.history_snapshot_counts(S.engine())}


# ------------------------------------------------------------------ static frontend
_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")
    app.mount("/textures", StaticFiles(directory=_DIST / "textures"), name="textures")
    @app.get("/{path:path}")
    def spa(path: str):
        f = _DIST / path
        return FileResponse(f if f.is_file() else _DIST / "index.html")
