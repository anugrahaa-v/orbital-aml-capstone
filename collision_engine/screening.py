"""Geometric conjunction SCREENING (not collision probability).

Frames: all states TEME (km, km/s), UTC times. RTN is the primary's radial/transverse/normal frame.
Categories are screening bins on predicted miss distance only, NOT operational warnings.
Pc is computed ONLY when covariance and hard-body radius are supplied (see `encounter_plane_pc`).
"""
from __future__ import annotations
import datetime as dt
from dataclasses import dataclass, field, asdict
import numpy as np
from scipy.optimize import minimize_scalar
from sgp4.api import SatrecArray

from orbital_engine.propagate import to_jd

V_REL_MAX_KMS = 16.0   # upper bound on LEO relative speed (retrograde head-on ~15.5 km/s)


@dataclass
class ScreeningConfig:
    threshold_km: float = 10.0          # report approaches closer than this
    horizon_h: float = 24.0
    coarse_step_s: float = 30.0
    altitude_margin_km: float = 30.0    # perigee/apogee prefilter slack (mean-element + drift allowance)
    category_edges_km: tuple = (1.0, 5.0)
    max_results: int = 200


def category(miss_km: float, cfg: ScreeningConfig) -> str:
    if miss_km < cfg.category_edges_km[0]: return "CLOSE_APPROACH"
    if miss_km < cfg.category_edges_km[1]: return "ELEVATED"
    return "WATCH"


def _t(base: dt.datetime, s: float) -> dt.datetime:
    return base + dt.timedelta(seconds=float(s))


def _propagate_grid(sats, base: dt.datetime, offsets_s: np.ndarray):
    jd0, fr0 = to_jd(base)
    jds = np.full(len(offsets_s), jd0)
    frs = fr0 + offsets_s / 86400.0
    return SatrecArray(sats).sgp4(jds, frs)


def rtn_basis(r, v):
    R = r / np.linalg.norm(r)
    N = np.cross(r, v); N /= np.linalg.norm(N)
    T = np.cross(N, R)
    return np.vstack([R, T, N])        # rows: R, T, N


def _state(sat, base, s):
    e, r, v = sat.sgp4(*to_jd(_t(base, s)))
    if e != 0:
        return None
    return np.array(r), np.array(v)


def _rel(sa, sb, base, s):
    A, B = _state(sa, base, s), _state(sb, base, s)
    if A is None or B is None:
        return np.inf, None
    dr = B[0] - A[0]
    return float(np.linalg.norm(dr)), (A, B, dr, B[1] - A[1])


@dataclass
class Conjunction:
    primary_id: int
    secondary_id: int
    tca_utc: dt.datetime
    miss_distance_km: float
    relative_speed_kms: float
    rel_pos_teme_km: list
    rel_vel_teme_kms: list
    rel_pos_rtn_km: list
    category: str
    threshold_km: float
    time_to_tca_s: float
    epoch_age_days: dict = field(default_factory=dict)
    uncertainty_available: bool = False
    pc: dict = field(default_factory=lambda: {"available": False,
                     "reason": "GP/TLE data carries no covariance or hard-body radius; Pc not computed."})
    data_label: str = "SGP4-propagated from observed mean elements (simulation, not an operational warning)"

    def to_dict(self):
        d = asdict(self)
        d["tca_utc"] = self.tca_utc.isoformat()
        return d


def _refine(sa, sb, base, s_lo, s_hi):
    res = minimize_scalar(lambda s: _rel(sa, sb, base, s)[0], bounds=(s_lo, s_hi), method="bounded",
                          options={"xatol": 1e-3})
    return float(res.x)


def _make(pa, pb, base, s_tca, cfg, epoch_a, epoch_b, t_ref):
    d, pack = _rel(pa.satrec, pb.satrec, base, s_tca)
    A, B, dr, dv = pack
    basis = rtn_basis(*A)
    tca = _t(base, s_tca)
    return Conjunction(pa.norad_id, pb.norad_id, tca, d, float(np.linalg.norm(dv)), dr.tolist(), dv.tolist(),
                       (basis @ dr).tolist(), category(d, cfg), cfg.threshold_km,
                       (tca - t_ref).total_seconds(),
                       {"primary": abs((tca - epoch_a).total_seconds()) / 86400,
                        "secondary": abs((tca - epoch_b).total_seconds()) / 86400})


def closest_approach(pa, pb, t0: dt.datetime, horizon_s: float, step_s: float = 10.0,
                     cfg: ScreeningConfig | None = None) -> Conjunction | None:
    """Global minimum separation of two CatObject-like objects (needs .satrec, .norad_id, .epoch_utc)."""
    cfg = cfg or ScreeningConfig()
    offs = np.arange(0.0, horizon_s + step_s, step_s)
    ea, ra, _ = _propagate_grid([pa.satrec], t0, offs)
    eb, rb, _ = _propagate_grid([pb.satrec], t0, offs)
    ok = (ea[0] == 0) & (eb[0] == 0)
    d = np.where(ok, np.linalg.norm(rb[0] - ra[0], axis=1), np.inf)
    if not np.isfinite(d).any():
        return None
    best = None
    idx = [i for i in range(len(d)) if d[i] < np.inf and
           (i == 0 or d[i] <= d[i - 1]) and (i == len(d) - 1 or d[i] <= d[i + 1])]
    for i in sorted(idx, key=lambda i: d[i])[:5]:
        lo, hi = max(0.0, offs[i] - step_s), min(offs[-1], offs[i] + step_s)
        s = _refine(pa.satrec, pb.satrec, t0, lo, hi)
        dist = _rel(pa.satrec, pb.satrec, t0, s)[0]
        if best is None or dist < best[1]:
            best = (s, dist)
    return _make(pa, pb, t0, best[0], cfg, pa.epoch_utc, pb.epoch_utc, t0)


def distance_series(pa, pb, center: dt.datetime, half_window_s: float, step_s: float):
    offs = np.arange(-half_window_s, half_window_s + step_s, step_s)
    ea, ra, va = _propagate_grid([pa.satrec], center, offs)
    eb, rb, vb = _propagate_grid([pb.satrec], center, offs)
    ok = (ea[0] == 0) & (eb[0] == 0)
    dist = np.where(ok, np.linalg.norm(rb[0] - ra[0], axis=1), np.nan)
    rel_speed = np.where(ok, np.linalg.norm(vb[0] - va[0], axis=1), np.nan)
    return offs, ra[0], rb[0], dist, rel_speed, ok


def altitude_prefilter(cat, primary, margin_km: float, threshold_km: float):
    """Indices of objects whose [perigee, apogee] shell overlaps the primary's within slack."""
    gap = np.maximum(cat.perigee, primary.perigee_km) - np.minimum(cat.apogee, primary.apogee_km)
    return np.where(gap <= threshold_km + margin_km)[0]


def screen_primary(cat, primary, t0: dt.datetime, cfg: ScreeningConfig | None = None,
                   chunk: int = 1500, progress=None) -> dict:
    cfg = cfg or ScreeningConfig()
    cand_idx = [i for i in altitude_prefilter(cat, primary, cfg.altitude_margin_km, cfg.threshold_km)
                if cat.objects[i].norad_id != primary.norad_id]
    offs = np.arange(0.0, cfg.horizon_h * 3600 + cfg.coarse_step_s, cfg.coarse_step_s)
    ep, rp, _ = _propagate_grid([primary.satrec], t0, offs)
    coarse_limit = cfg.threshold_km + V_REL_MAX_KMS * cfg.coarse_step_s / 2
    found: list[Conjunction] = []
    for c0 in range(0, len(cand_idx), chunk):
        sub = [cat.objects[i] for i in cand_idx[c0:c0 + chunk]]
        es, rs, _ = _propagate_grid([o.satrec for o in sub], t0, offs)
        d = np.linalg.norm(rs - rp[0][None], axis=2)
        d[(es != 0) | (ep[0][None] != 0)] = np.inf
        mid = d[:, 1:-1]
        mins = (mid <= d[:, :-2]) & (mid <= d[:, 2:]) & (mid < coarse_limit)
        for si, ti in zip(*np.nonzero(mins)):
            ti += 1
            o = sub[si]
            s = _refine(primary.satrec, o.satrec, t0, offs[ti - 1], offs[ti + 1])
            dist = _rel(primary.satrec, o.satrec, t0, s)[0]
            if dist <= cfg.threshold_km:
                found.append(_make(primary, o, t0, s, cfg, primary.epoch_utc, o.epoch_utc, t0))
        if progress:
            progress(min(1.0, (c0 + chunk) / max(1, len(cand_idx))))
    found.sort(key=lambda c: c.miss_distance_km)
    # keep one (the closest) event per secondary within the horizon
    seen, uniq = set(), []
    for c in found:
        key = (c.secondary_id, int(c.time_to_tca_s // 3600))
        if key not in seen:
            seen.add(key); uniq.append(c)
    return {"config": asdict(cfg), "n_catalogue": len(cat), "n_candidates_after_altitude_filter": len(cand_idx),
            "conjunctions": [c.to_dict() for c in uniq[:cfg.max_results]]}


# ---------------- Probability of collision (only with covariance + HBR) ----------------
def encounter_frame(r_rel, v_rel):
    z = v_rel / np.linalg.norm(v_rel)
    miss = r_rel - np.dot(r_rel, z) * z
    mn = np.linalg.norm(miss)
    if mn < 1e-12:
        helper = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
        x = helper - np.dot(helper, z) * z; x /= np.linalg.norm(x)
    else:
        x = miss / mn
    y = np.cross(z, x)
    return np.vstack([x, y, z]), mn


def encounter_plane_pc(r_rel, v_rel, cov_a, cov_b, hbr_km: float, n_r: int = 48, n_t: int = 96) -> float:
    """2-D encounter-plane Pc (linear relative motion, short-encounter assumption, Gaussian position errors).
    cov_* are 3x3 position covariances [km^2] in the SAME inertial frame as r_rel."""
    U, miss = encounter_frame(np.asarray(r_rel, float), np.asarray(v_rel, float))
    C = U @ (np.asarray(cov_a, float) + np.asarray(cov_b, float)) @ U.T
    C2 = C[:2, :2]
    det = np.linalg.det(C2)
    if det <= 0:
        raise ValueError("combined 2x2 encounter-plane covariance must be positive definite")
    inv = np.linalg.inv(C2)
    gx, gw = np.polynomial.legendre.leggauss(n_r)
    rr, wr = 0.5 * hbr_km * (gx + 1), 0.5 * hbr_km * gw
    th = np.linspace(0, 2 * np.pi, n_t, endpoint=False)
    X = miss + rr[:, None] * np.cos(th)[None]
    Y = rr[:, None] * np.sin(th)[None]
    q = inv[0, 0] * X ** 2 + 2 * inv[0, 1] * X * Y + inv[1, 1] * Y ** 2
    pdf = np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(det))
    return float(np.sum(pdf * rr[:, None] * wr[:, None]) * (2 * np.pi / n_t))


def pc_or_unavailable(conj: Conjunction, cov_a=None, cov_b=None, hbr_km=None) -> dict:
    if cov_a is None or cov_b is None or hbr_km is None:
        return {"available": False, "reason": "Covariance and/or hard-body radius not provided. "
                                              "GP/TLE data has neither; Pc is not estimated."}
    pc = encounter_plane_pc(conj.rel_pos_teme_km, conj.rel_vel_teme_kms, cov_a, cov_b, hbr_km)
    return {"available": True, "pc": pc, "hbr_km": hbr_km,
            "assumptions": ["Gaussian, uncorrelated position errors", "linear relative motion through encounter",
                            "covariances given in TEME, km^2", "user-supplied inputs (not derived from GP data)"]}
