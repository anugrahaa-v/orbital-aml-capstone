"""OMM / TLE parsing and validation -> sgp4 Satrec objects."""
from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from sgp4.api import Satrec, WGS72
from sgp4 import omm as sgp4_omm

REQUIRED = ("OBJECT_NAME", "NORAD_CAT_ID", "EPOCH", "MEAN_MOTION", "ECCENTRICITY",
            "INCLINATION", "RA_OF_ASC_NODE", "ARG_OF_PERICENTER", "MEAN_ANOMALY", "BSTAR")


@dataclass
class ParsedObject:
    norad_id: int
    name: str
    object_id: str | None       # international designator (launch-derived)
    object_type: str | None     # only present if the source provides it
    epoch_utc: dt.datetime
    fields: dict                # raw OMM fields (real observations)
    satrec: Satrec


@dataclass
class Rejected:
    norad_id: str | None
    reason: str


def _parse_epoch(s: str) -> dt.datetime:
    d = dt.datetime.fromisoformat(s)
    return d.replace(tzinfo=dt.timezone.utc) if d.tzinfo is None else d.astimezone(dt.timezone.utc)


def validate_fields(f: dict) -> str | None:
    """Return a rejection reason, or None if the record is physically plausible."""
    for k in REQUIRED:
        if f.get(k) in (None, ""):
            return f"missing {k}"
    try:
        e, i, n = float(f["ECCENTRICITY"]), float(f["INCLINATION"]), float(f["MEAN_MOTION"])
        _parse_epoch(f["EPOCH"])
    except (ValueError, TypeError) as exc:
        return f"unparseable value: {exc}"
    if not 0.0 <= e < 1.0:
        return f"eccentricity out of range: {e}"
    if not 0.0 <= i <= 180.0:
        return f"inclination out of range: {i}"
    if n <= 0.0:
        return f"non-positive mean motion: {n}"
    return None


def parse_omm_records(records: list[dict]) -> tuple[list[ParsedObject], list[Rejected]]:
    """Parse OMM JSON records; keep the newest epoch per NORAD ID; report rejects."""
    best: dict[int, ParsedObject] = {}
    rejected: list[Rejected] = []
    for f in records:
        reason = validate_fields(f)
        if reason:
            rejected.append(Rejected(str(f.get("NORAD_CAT_ID")), reason))
            continue
        sat = Satrec()
        try:
            # sgp4's initializer indexes OBJECT_ID unconditionally; the stored
            # record (`f`) is left untouched, only the init copy gets a blank.
            sgp4_omm.initialize(sat, {"OBJECT_ID": "", **f})
        except Exception as exc:  # sgp4init failure on malformed element sets
            rejected.append(Rejected(str(f.get("NORAD_CAT_ID")), f"sgp4 init failed: {exc}"))
            continue
        obj = ParsedObject(int(f["NORAD_CAT_ID"]), f["OBJECT_NAME"], f.get("OBJECT_ID"),
                           f.get("OBJECT_TYPE"), _parse_epoch(f["EPOCH"]), f, sat)
        old = best.get(obj.norad_id)
        if old is None or obj.epoch_utc > old.epoch_utc:
            best[obj.norad_id] = obj
    return list(best.values()), rejected


def parse_tle(line1: str, line2: str) -> Satrec:
    """Legacy TLE -> Satrec (WGS72, as SGP4 requires)."""
    return Satrec.twoline2rv(line1, line2, WGS72)
