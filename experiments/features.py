"""Feature tables built from REAL element sets. Column provenance:
OBSERVED (published OMM fields) vs DERIVED (computed). Targets are labelled likewise."""
from __future__ import annotations
import numpy as np, pandas as pd
from orbital_engine.elements import MU, R_EARTH

OBSERVED = ["mean_motion", "eccentricity", "inclination", "raan", "arg_perigee", "mean_anomaly", "bstar"]
DERIVED = ["a_km", "period_min", "perigee_alt", "apogee_alt", "mean_alt", "log_abs_bstar", "raan_sin", "raan_cos"]
DEFAULT_FEATURES = ["mean_motion", "eccentricity", "inclination", "bstar", "log_abs_bstar", "a_km", "perigee_alt",
                    "apogee_alt", "raan_sin", "raan_cos"]


def feature_frame(cat) -> pd.DataFrame:
    rows = []
    for o in cat.objects:
        f = o.fields
        n = float(f["MEAN_MOTION"]); e = float(f["ECCENTRICITY"]); b = float(f["BSTAR"])
        raan = float(f["RA_OF_ASC_NODE"])
        rows.append(dict(norad_id=o.norad_id, name=o.name, group=o.group, obj_class=o.obj_class, epoch=o.epoch_utc,
                         mean_motion=n, eccentricity=e, inclination=float(f["INCLINATION"]), raan=raan,
                         arg_perigee=float(f["ARG_OF_PERICENTER"]), mean_anomaly=float(f["MEAN_ANOMALY"]),
                         bstar=b, log_abs_bstar=np.log10(abs(b) + 1e-12), a_km=o.a_km, period_min=o.period_min,
                         perigee_alt=o.perigee_km, apogee_alt=o.apogee_km, mean_alt=0.5 * (o.perigee_km + o.apogee_km),
                         raan_sin=np.sin(np.radians(raan)), raan_cos=np.cos(np.radians(raan)),
                         mean_motion_dot=_num(f.get("MEAN_MOTION_DOT")), mean_motion_ddot=_num(f.get("MEAN_MOTION_DDOT"))))
    return pd.DataFrame(rows)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def add_cross_sectional_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Single-snapshot targets.
    mean_motion_dot       : OBSERVED field as published (rev/day^2).
    decay_alt_rate_km_day : DERIVED. Assumes the published value is ndot/2 (TLE convention) so dn/dt = 2*value;
                            da/dt = -(2/3) a (dn/dt)/n.  Treat as a derived indicator, not a measurement."""
    d = df.copy()
    dn_dt = 2.0 * d["mean_motion_dot"]
    d["decay_alt_rate_km_day"] = -(2.0 / 3.0) * d["a_km"] * dn_dt / d["mean_motion"]
    return d


def history_targets(hist: pd.DataFrame, min_gap_days: float = 0.5) -> pd.DataFrame:
    """Chronological supervised pairs from REAL accumulated snapshots.
    Features at snapshot k, targets = change to snapshot k+1 per day. Requires >=2 epochs per object."""
    h = hist.sort_values(["norad_id", "epoch_utc"]).copy()
    n = h["mean_motion"].to_numpy()
    h["a_km"] = (MU / (n * 2 * np.pi / 86400.0) ** 2) ** (1 / 3)
    h["mean_alt"] = h["a_km"] - R_EARTH
    g = h.groupby("norad_id")
    h["next_epoch"] = g["epoch_utc"].shift(-1)
    h["gap_days"] = (h["next_epoch"] - h["epoch_utc"]).dt.total_seconds() / 86400.0
    h["d_mean_motion_per_day"] = (g["mean_motion"].shift(-1) - h["mean_motion"]) / h["gap_days"]
    h["d_altitude_km_per_day"] = (g["mean_alt"].shift(-1) - h["mean_alt"]) / h["gap_days"]
    h["perigee_alt"] = h["a_km"] * (1 - h["eccentricity"]) - R_EARTH
    h["apogee_alt"] = h["a_km"] * (1 + h["eccentricity"]) - R_EARTH
    h["log_abs_bstar"] = np.log10(h["bstar"].abs() + 1e-12)
    h["raan_sin"] = np.sin(np.radians(h["raan"])); h["raan_cos"] = np.cos(np.radians(h["raan"]))
    h = h.rename(columns={"epoch_utc": "epoch"})
    return h[(h["gap_days"] >= min_gap_days)].dropna(subset=["d_mean_motion_per_day", "d_altitude_km_per_day"])


def training_table(cat, hist: pd.DataFrame | None = None, min_snapshots: int = 2):
    """-> (df, mode, targets, data_label, note). Uses real history when available, else cross-sectional."""
    if hist is not None and len(hist):
        pairs = history_targets(hist)
        if len(pairs) >= 200 and pairs["norad_id"].nunique() >= 20:
            return (pairs, "history", ["d_altitude_km_per_day", "d_mean_motion_per_day"], "observed history",
                    "Chronological targets from consecutive REAL snapshots (future change in altitude / mean motion).")
    df = add_cross_sectional_targets(feature_frame(cat)).dropna(subset=["mean_motion_dot"])
    return (df, "cross_sectional", ["mean_motion_dot", "decay_alt_rate_km_day"], "observed + derived",
            "Only one real snapshot per object is available (CelesTrak serves current elements). Targets are the published "
            "drag-rate field and a derived altitude-decay rate. This is cross-sectional regression, NOT future forecasting. "
            "Re-run ingestion over days to accumulate history for true chronological targets.")
