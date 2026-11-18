"""Download (cache-first) and validate CelesTrak GP groups, then print a summary.

    python -m backend.ingest_cli --groups active cosmos-2251-debris
"""
from __future__ import annotations
import argparse, collections
from pathlib import Path

from backend.ingestion.celestrak import fetch_group, CelesTrakError
from backend.ingestion.omm import parse_omm_records
from orbital_engine.elements import derive


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["stations"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--no-db", action="store_true", help="skip storing a history snapshot in the database")
    a = ap.parse_args(argv)
    for g in a.groups:
        try:
            records, info = fetch_group(g, Path(a.data_dir))
        except CelesTrakError as exc:
            print(f"[{g}] FAILED: {exc}\nNo data was fabricated. Check network access / fair-use window.")
            return 1
        objs, rejected = parse_omm_records(records)
        why = collections.Counter(r.reason.split(":")[0] for r in rejected)
        print(f"[{g}] raw={info.n_records_raw} valid={len(objs)} rejected={len(rejected)} "
              f"from_cache={info.from_cache} sha256={info.dataset_version[:12]}")
        for k, n in why.most_common(5):
            print(f"    rejected: {k} x{n}")
        if objs:
            alts = sorted(derive(float(o.fields['MEAN_MOTION']), float(o.fields['ECCENTRICITY'])).perigee_alt_km
                          for o in objs)
            print(f"    perigee altitude km (derived): min={alts[0]:.0f} median={alts[len(alts)//2]:.0f} max={alts[-1]:.0f}")
    if not a.no_db:
        from backend.ingestion.pipeline import ingest
        try:
            print(ingest(a.groups, Path(a.data_dir)))
        except CelesTrakError as exc:
            print(f'DB snapshot skipped: {exc}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
