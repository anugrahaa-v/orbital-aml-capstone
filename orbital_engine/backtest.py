"""REAL SGP4 back-test (needs >=2 accumulated snapshots): propagate element set k to the epoch of set k+1
and compare with k+1's own state. Error vs lead time = realised TLE-to-TLE consistency (observed vs predicted)."""
from __future__ import annotations
import numpy as np
from sgp4.api import Satrec
from sgp4 import omm as sgp4_omm
from .propagate import to_jd


def _sat(raw: dict):
    s = Satrec(); sgp4_omm.initialize(s, {"OBJECT_ID": "", **raw}); return s


def backtest(history_rows: list[dict]) -> dict:
    out = []
    for a, b in zip(history_rows[:-1], history_rows[1:]):
        sa, sb = _sat(a["raw_omm"]), _sat(b["raw_omm"])
        tb = b["epoch_utc"]
        import datetime as dt
        t = tb.replace(tzinfo=dt.timezone.utc) if tb.tzinfo is None else tb
        jd, fr = to_jd(t)
        ea, ra, va = sa.sgp4(jd, fr); eb, rb, vb = sb.sgp4(jd, fr)
        if ea or eb:
            continue
        lead_h = (b["epoch_utc"] - a["epoch_utc"]).total_seconds() / 3600
        rel = np.array(rb) - np.array(ra)
        # decompose into radial / along-track / cross using set b's state
        R = np.array(rb) / np.linalg.norm(rb); N = np.cross(rb, vb); N /= np.linalg.norm(N); T = np.cross(N, R)
        out.append({"lead_hours": lead_h, "position_error_km": float(np.linalg.norm(rel)),
                    "radial_km": float(rel @ R), "along_track_km": float(rel @ T), "cross_track_km": float(rel @ N)})
    return {"pairs": out, "label": "OBSERVED vs SGP4-predicted (real backtest)",
            "n_pairs": len(out)}
