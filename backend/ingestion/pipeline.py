"""End-to-end ingestion: CelesTrak GP -> validate -> catalogue -> DB history snapshot."""
from __future__ import annotations
import datetime as dt
from pathlib import Path

from backend.ingestion.celestrak import fetch_group
from orbital_engine.catalog import Catalogue
from database import store


def ingest(groups: list[str], data_dir: Path = Path("data"), db_url: str | None = None,
           use_db: bool = True, session=None) -> dict:
    by_group, infos = {}, []
    for g in groups:
        recs, info = fetch_group(g, data_dir, session=session)
        by_group[g] = recs
        infos.append(info)
    cat = Catalogue.from_records(by_group, [i.__dict__ for i in infos])
    out = {"groups": groups, "objects": len(cat), "rejected": cat.rejected,
           "sources": [i.__dict__ for i in infos]}
    if use_db:
        eng = store.get_engine(db_url, data_dir)
        version = ",".join(i.dataset_version[:12] for i in infos)
        retrieved = dt.datetime.now(dt.timezone.utc)
        out["new_objects"] = store.upsert_objects(eng, cat, retrieved)
        out["new_history_rows"] = store.append_history(eng, cat, retrieved, version.ljust(64, "0")[:64])
        out["history"] = store.history_snapshot_counts(eng)
    return out
