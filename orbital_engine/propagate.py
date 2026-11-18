"""SGP4 propagation. Output frame: TEME, km and km/s, UTC timestamps."""
from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
import numpy as np
from sgp4.api import Satrec, jday, SatrecArray

from .frames import teme_to_ecef, ecef_to_geodetic

SGP4_ERRORS = {1: "mean eccentricity out of range", 2: "mean motion <= 0",
               3: "perturbed eccentricity out of range", 4: "semi-latus rectum < 0",
               5: "epoch elements are sub-orbital", 6: "satellite has decayed"}


@dataclass(frozen=True)
class StateTEME:
    time_utc: dt.datetime
    r_km: np.ndarray      # TEME position
    v_kms: np.ndarray     # TEME velocity
    frame: str = "TEME"


def to_jd(t: dt.datetime) -> tuple[float, float]:
    t = t.astimezone(dt.timezone.utc)
    return jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)


def propagate(sat: Satrec, t: dt.datetime) -> StateTEME:
    jd, fr = to_jd(t)
    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        raise ValueError(f"SGP4 error {e}: {SGP4_ERRORS.get(e, 'unknown')}")
    return StateTEME(t, np.array(r), np.array(v))


def propagate_many(sats: list[Satrec], times: list[dt.datetime]):
    """Vectorised: returns (err[N,T], r[N,T,3], v[N,T,3]) in TEME. Non-zero err => invalid."""
    arr = SatrecArray(sats)
    jds, frs = zip(*(to_jd(t) for t in times))
    return arr.sgp4(np.array(jds), np.array(frs))


def ground_point(state: StateTEME):
    """Geodetic lat/lon/alt of a TEME state (UT1~=UTC assumption)."""
    jd, fr = to_jd(state.time_utc)
    r_ecef, _ = teme_to_ecef(state.r_km, state.v_kms, jd + fr)
    return ecef_to_geodetic(r_ecef)
