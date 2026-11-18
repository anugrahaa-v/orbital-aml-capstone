"""Time series from SGP4 propagation of REAL element sets. LABEL: physics simulation (SGP4-generated)."""
from __future__ import annotations
import datetime as dt
import numpy as np
from .propagate import to_jd
from .frames import teme_to_ecef_many, ecef_to_geodetic_many


def altitude_series(obj, start: dt.datetime, duration_h: float, step_s: float):
    """-> dict(times, altitude_km, speed_kms, lat, lon). SGP4 errors => NaN (dropped by caller)."""
    n = int(duration_h * 3600 / step_s) + 1
    times = [start + dt.timedelta(seconds=i * step_s) for i in range(n)]
    jd0, fr0 = to_jd(start)
    frs = fr0 + np.arange(n) * step_s / 86400.0
    e, r, v = obj.satrec.sgp4_array(np.full(n, jd0), frs)
    ok = e == 0
    ecef = teme_to_ecef_many(r, jd0 + frs)
    lat, lon, alt = ecef_to_geodetic_many(ecef)
    speed = np.linalg.norm(v, axis=1)
    alt = np.where(ok, alt, np.nan)
    return {"times": times, "altitude_km": alt, "speed_kms": np.where(ok, speed, np.nan),
            "lat": np.where(ok, lat, np.nan), "lon": np.where(ok, lon, np.nan),
            "label": "SIMULATED: SGP4-propagated from observed mean elements"}
