"""In-memory object catalogue built from REAL CelesTrak GP records.

Object class is NOT in GP data. It is assigned from the CelesTrak group the
record came from (documented, defensible) and, only as a fallback, from the
object name (flagged `name_heuristic`).
"""
from __future__ import annotations
import datetime as dt, json, re
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from sgp4.api import SatrecArray

from backend.ingestion.omm import parse_omm_records, ParsedObject
from backend.ingestion.registry import read_all
from .elements import derive
from .propagate import to_jd

_CACHE_RE = re.compile(r"^gp_(.+)_(\d{4}-\d{2}-\d{2}T\d{6}Z)\.json$")


def classify(name: str, group: str) -> tuple[str, str]:
    """-> (class, class_source). class in PAYLOAD | DEBRIS | ROCKET_BODY."""
    if "debris" in group.lower():
        return "DEBRIS", "celestrak_group"
    n = f" {name.upper()} "
    if " R/B" in n:
        return "ROCKET_BODY", "name_heuristic"
    if " DEB " in n or n.rstrip().endswith(" DEB"):
        return "DEBRIS", "name_heuristic"
    return "PAYLOAD", "celestrak_group"


@dataclass
class CatObject:
    norad_id: int
    name: str
    object_id: str | None
    group: str
    obj_class: str
    class_source: str
    epoch_utc: dt.datetime
    fields: dict
    satrec: object
    a_km: float
    period_min: float
    perigee_km: float
    apogee_km: float
    inc_deg: float

    def summary(self) -> dict:
        return {"norad_id": self.norad_id, "name": self.name, "object_id": self.object_id,
                "group": self.group, "class": self.obj_class, "class_source": self.class_source,
                "epoch": self.epoch_utc.isoformat(), "perigee_km": round(self.perigee_km, 1),
                "apogee_km": round(self.apogee_km, 1), "period_min": round(self.period_min, 2),
                "inc_deg": round(self.inc_deg, 3)}


@dataclass
class Catalogue:
    objects: list[CatObject]
    provenance: list[dict] = field(default_factory=list)
    rejected: int = 0
    _index: dict = field(default_factory=dict, repr=False)
    _arr: SatrecArray | None = field(default=None, repr=False)

    def __post_init__(self):
        self._index = {o.norad_id: i for i, o in enumerate(self.objects)}
        self.perigee = np.array([o.perigee_km for o in self.objects])
        self.apogee = np.array([o.apogee_km for o in self.objects])

    def __len__(self): return len(self.objects)

    @classmethod
    def from_records(cls, records_by_group: dict[str, list[dict]], provenance=None) -> "Catalogue":
        best: dict[int, CatObject] = {}
        rejected = 0
        for group, recs in records_by_group.items():
            parsed, rej = parse_omm_records(recs)
            rejected += len(rej)
            for p in parsed:
                a = derive(float(p.fields["MEAN_MOTION"]), float(p.fields["ECCENTRICITY"]))
                cls_, src = classify(p.name, group)
                o = CatObject(p.norad_id, p.name, p.object_id, group, cls_, src, p.epoch_utc, p.fields,
                              p.satrec, a.semi_major_axis_km, a.period_min, a.perigee_alt_km,
                              a.apogee_alt_km, float(p.fields["INCLINATION"]))
                old = best.get(o.norad_id)
                # prefer debris-group label when an object appears in several groups; else newest epoch
                if old is None or (o.group != "active" and old.group == "active") or \
                   (o.group == old.group and o.epoch_utc > old.epoch_utc):
                    best[o.norad_id] = o
        return cls(sorted(best.values(), key=lambda o: o.norad_id), provenance or [], rejected)

    @classmethod
    def from_cache(cls, data_dir: Path = Path("data")) -> "Catalogue":
        latest: dict[str, Path] = {}
        for f in sorted((data_dir / "cache").glob("gp_*.json")):
            m = _CACHE_RE.match(f.name)
            if m:
                latest[m.group(1)] = f
        recs = {g: json.loads(f.read_text()) for g, f in latest.items()}
        recs = {g: r for g, r in recs.items() if isinstance(r, list)}
        prov = read_all(data_dir / "registry" / "source_registry.jsonl")
        return cls.from_records(recs, prov)

    # ---- access ----
    def get(self, norad_id: int) -> CatObject:
        return self.objects[self._index[norad_id]]

    def has(self, norad_id: int) -> bool:
        return norad_id in self._index

    def filter(self, q: str | None = None, obj_class: str | None = None, alt_min=None, alt_max=None,
               group: str | None = None, limit: int | None = None) -> list[CatObject]:
        out = []
        ql = q.lower() if q else None
        for o in self.objects:
            if obj_class and o.obj_class != obj_class: continue
            if group and o.group != group: continue
            if ql and ql not in o.name.lower() and ql != str(o.norad_id): continue
            mean_alt = 0.5 * (o.perigee_km + o.apogee_km)
            if alt_min is not None and mean_alt < alt_min: continue
            if alt_max is not None and mean_alt > alt_max: continue
            out.append(o)
            if limit and len(out) >= limit: break
        return out

    def positions(self, objs: list[CatObject], times: list[dt.datetime]):
        """TEME km, km/s. Returns (err[N,T], r[N,T,3], v[N,T,3]); err != 0 means invalid state."""
        arr = SatrecArray([o.satrec for o in objs])
        jd, fr = zip(*(to_jd(t) for t in times))
        return arr.sgp4(np.array(jd), np.array(fr))
