"""Persistence layer. DATABASE_URL=postgresql+psycopg://user:pw@host/db for PostgreSQL;
defaults to SQLite at data/orbitalx.db so the app runs with zero setup."""
from __future__ import annotations
import datetime as dt, os, uuid
from pathlib import Path
from sqlalchemy import create_engine, select, insert, func
from sqlalchemy.engine import Engine

from .models import (metadata, satellite_objects, orbital_element_history, model_experiments,
                     rl_experiments, conjunction_screening, predictions, propagated_positions)


def get_engine(url: str | None = None, data_dir: Path = Path("data")) -> Engine:
    url = url or os.getenv("DATABASE_URL") or f"sqlite:///{(data_dir / 'orbitalx.db').as_posix()}"
    if url.startswith("sqlite"):
        data_dir.mkdir(parents=True, exist_ok=True)
    eng = create_engine(url, future=True)
    metadata.create_all(eng)
    return eng


def _naive(t: dt.datetime) -> dt.datetime:
    return t.astimezone(dt.timezone.utc).replace(tzinfo=None) if t.tzinfo else t


def _insert_ignore(eng: Engine, table, rows: list[dict], conflict_cols: list[str]):
    if not rows:
        return 0
    with eng.begin() as c:
        if eng.dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            c.execute(pg_insert(table).on_conflict_do_nothing(index_elements=conflict_cols), rows)
        elif eng.dialect.name == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as sl_insert
            c.execute(sl_insert(table).on_conflict_do_nothing(index_elements=conflict_cols), rows)
        else:
            c.execute(insert(table), rows)
    return len(rows)


def upsert_objects(eng: Engine, cat, now: dt.datetime | None = None):
    now = _naive(now or dt.datetime.now(dt.timezone.utc))
    with eng.begin() as c:
        existing = {r[0] for r in c.execute(select(satellite_objects.c.norad_id))}
        new = [dict(norad_id=o.norad_id, name=o.name, object_id=o.object_id, source_group=o.group,
                    obj_class=o.obj_class, class_source=o.class_source, first_seen_utc=now, last_seen_utc=now)
               for o in cat.objects if o.norad_id not in existing]
        for i in range(0, len(new), 5000):
            c.execute(insert(satellite_objects), new[i:i + 5000])
        if existing:
            c.execute(satellite_objects.update().where(satellite_objects.c.norad_id.in_(
                [o.norad_id for o in cat.objects if o.norad_id in existing])).values(last_seen_utc=now))
    return len(new)


def append_history(eng: Engine, cat, retrieved: dt.datetime, dataset_version: str) -> int:
    rows = []
    for o in cat.objects:
        f = o.fields
        rows.append(dict(norad_id=o.norad_id, epoch_utc=_naive(o.epoch_utc), retrieved_utc=_naive(retrieved),
                         dataset_version=dataset_version, mean_motion=float(f["MEAN_MOTION"]),
                         eccentricity=float(f["ECCENTRICITY"]), inclination=float(f["INCLINATION"]),
                         raan=float(f["RA_OF_ASC_NODE"]), arg_perigee=float(f["ARG_OF_PERICENTER"]),
                         mean_anomaly=float(f["MEAN_ANOMALY"]), bstar=float(f["BSTAR"]),
                         mean_motion_dot=_f(f.get("MEAN_MOTION_DOT")), mean_motion_ddot=_f(f.get("MEAN_MOTION_DDOT")),
                         raw_omm=f))
    with eng.begin() as c:
        before = c.execute(select(func.count()).select_from(orbital_element_history)).scalar_one()
    for i in range(0, len(rows), 2000):
        _insert_ignore(eng, orbital_element_history, rows[i:i + 2000], ["norad_id", "epoch_utc"])
    with eng.begin() as c:
        after = c.execute(select(func.count()).select_from(orbital_element_history)).scalar_one()
    return after - before


def _f(x):
    try:
        return None if x in (None, "") else float(x)
    except (TypeError, ValueError):
        return None


def history_snapshot_counts(eng: Engine) -> dict:
    with eng.begin() as c:
        n_rows = c.execute(select(func.count()).select_from(orbital_element_history)).scalar_one()
        n_epochs = c.execute(select(func.count(func.distinct(orbital_element_history.c.epoch_utc)))).scalar_one()
        n_obj = c.execute(select(func.count(func.distinct(orbital_element_history.c.norad_id)))).scalar_one()
        sub = (select(func.count().label("n")).select_from(orbital_element_history)
               .group_by(orbital_element_history.c.norad_id).subquery())
        per = c.execute(select(func.max(sub.c.n))).scalar() if n_rows else 0
    return {"rows": n_rows, "distinct_epochs": n_epochs, "objects": n_obj, "max_snapshots_per_object": per or 0}


def get_history(eng: Engine, norad_id: int) -> list[dict]:
    t = orbital_element_history
    with eng.begin() as c:
        rows = c.execute(select(t).where(t.c.norad_id == norad_id).order_by(t.c.epoch_utc)).mappings().all()
    return [dict(r) for r in rows]


def history_frame(eng: Engine):
    import pandas as pd
    t = orbital_element_history
    with eng.begin() as c:
        return pd.read_sql(select(t.c.norad_id, t.c.epoch_utc, t.c.mean_motion, t.c.eccentricity, t.c.inclination,
                                  t.c.raan, t.c.arg_perigee, t.c.mean_anomaly, t.c.bstar, t.c.mean_motion_dot)
                           .order_by(t.c.norad_id, t.c.epoch_utc), c)


def save_experiment(eng: Engine, kind: str, model_name: str | None, target: str | None, seed: int | None,
                    params: dict, metrics: dict, provenance: dict, data_label: str, run_id: str | None = None) -> str:
    run_id = run_id or uuid.uuid4().hex[:16]
    with eng.begin() as c:
        c.execute(insert(model_experiments).values(
            run_id=run_id, created_utc=_naive(dt.datetime.now(dt.timezone.utc)), kind=kind, model_name=model_name,
            target=target, seed=seed, params=_clean(params), metrics=_clean(metrics), provenance=_clean(provenance),
            data_label=data_label))
    return run_id


def list_experiments(eng: Engine, kind: str | None = None, limit: int = 500) -> list[dict]:
    t = model_experiments
    q = select(t).order_by(t.c.created_utc.desc()).limit(limit)
    if kind:
        q = q.where(t.c.kind == kind)
    with eng.begin() as c:
        return [dict(r) for r in c.execute(q).mappings().all()]


def save_rl(eng: Engine, run_id: str, policy: str, seed: int, final_sep: float, fuel: float, reward: float,
            config: dict, trajectory: dict | None = None):
    with eng.begin() as c:
        c.execute(insert(rl_experiments).values(
            run_id=run_id, created_utc=_naive(dt.datetime.now(dt.timezone.utc)), policy=policy, seed=seed,
            final_separation_km=final_sep, fuel_used_ms=fuel, total_reward=reward, config=_clean(config),
            trajectory=_clean(trajectory) if trajectory else None))


def save_conjunctions(eng: Engine, run_utc: dt.datetime, rows: list[dict]):
    if rows:
        with eng.begin() as c:
            c.execute(insert(conjunction_screening), [{**r, "run_utc": _naive(run_utc), "tca_utc": _naive(r["tca_utc"])} for r in rows])


def _clean(o):
    """Make arbitrary nested dicts JSON-safe (numpy scalars, NaN)."""
    import numpy as np, math
    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, np.ndarray): return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(float(o)) or math.isinf(float(o))) else float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (dt.datetime, dt.date)): return o.isoformat()
    return o
