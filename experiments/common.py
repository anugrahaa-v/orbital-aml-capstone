"""Shared experiment utilities: provenance, chronological splits, tracking."""
from __future__ import annotations
import datetime as dt, json, uuid
from pathlib import Path
import numpy as np, pandas as pd

RUNS_DIR = Path("experiments/runs")


def provenance(cat) -> dict:
    ep = [o.epoch_utc for o in cat.objects]
    return {"source": "CelesTrak GP (OMM JSON)", "n_objects": len(cat),
            "epoch_min": min(ep).isoformat() if ep else None, "epoch_max": max(ep).isoformat() if ep else None,
            "dataset_versions": sorted({p.get("dataset_version", "")[:16] for p in cat.provenance}),
            "retrieved_utc": sorted({p.get("retrieved_utc", "") for p in cat.provenance}),
            "class_labels": "derived from CelesTrak group membership / name heuristic (not ground truth)"}


def chronological_split(df: pd.DataFrame, time_col: str, frac=(0.7, 0.1, 0.2)):
    """Sort by time and cut: train | validation | test. Guarantees max(train) <= min(val) <= min(test)."""
    assert abs(sum(frac) - 1) < 1e-9
    d = df.sort_values(time_col, kind="stable").reset_index(drop=True)
    n = len(d)
    i1, i2 = int(n * frac[0]), int(n * (frac[0] + frac[1]))
    tr, va, te = d.iloc[:i1], d.iloc[i1:i2], d.iloc[i2:]
    if len(tr) and len(te):
        assert tr[time_col].max() <= te[time_col].min()
    return tr, va, te


def regression_metrics(y, p) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mse = float(mean_squared_error(y, p))
    return {"mae": float(mean_absolute_error(y, p)), "mse": mse, "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y, p))}


def log_run(eng, kind: str, model_name: str | None, target: str | None, seed: int | None, params: dict,
            metrics: dict, prov: dict, data_label: str, extra_artifact: dict | None = None) -> str:
    from database import store
    run_id = uuid.uuid4().hex[:16]
    if eng is not None:
        store.save_experiment(eng, kind, model_name, target, seed, params, metrics, prov, data_label, run_id)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"run_id": run_id, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "kind": kind,
           "model": model_name, "target": target, "seed": seed, "params": params, "metrics": metrics,
           "provenance": prov, "data_label": data_label}
    (RUNS_DIR / f"{run_id}.json").write_text(json.dumps(store._clean(rec), indent=1))
    _mlflow(kind, model_name, params, metrics, run_id)
    return run_id


def _mlflow(kind, model_name, params, metrics, run_id):
    try:
        import mlflow, os
        if not os.getenv("MLFLOW_TRACKING_URI"):
            return
        mlflow.set_experiment(f"orbitalx/{kind}")
        with mlflow.start_run(run_name=f"{model_name or kind}-{run_id}"):
            mlflow.log_params({k: str(v)[:250] for k, v in (params or {}).items()})
            mlflow.log_metrics({k: float(v) for k, v in (metrics or {}).items()
                                if isinstance(v, (int, float)) and np.isfinite(v)})
    except Exception:
        pass   # MLflow is optional
