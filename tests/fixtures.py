"""TEST-ONLY fixtures. Everything generated here is synthetic and is NEVER written to data/ or presented
as observation. Real data comes exclusively from CelesTrak via backend.ingestion."""
import datetime as dt, math
import numpy as np
from orbital_engine.elements import MU

EPOCH = dt.datetime(2026, 11, 17, 0, 0, 0, tzinfo=dt.timezone.utc)


def mean_motion_for_alt(alt_km: float) -> float:
    a = 6378.137 + alt_km
    return 86400.0 / (2 * math.pi * math.sqrt(a ** 3 / MU))


def make_omm(norad, alt_km=550.0, ecc=0.0005, inc=53.0, raan=0.0, argp=0.0, ma=0.0, bstar=1e-4,
             name=None, epoch=EPOCH, ndot=1e-5):
    return {"OBJECT_NAME": name or f"TESTSAT {norad}", "OBJECT_ID": "2000-001A", "NORAD_CAT_ID": norad,
            "EPOCH": epoch.strftime("%Y-%m-%dT%H:%M:%S.%f"), "MEAN_MOTION": mean_motion_for_alt(alt_km),
            "ECCENTRICITY": ecc, "INCLINATION": inc, "RA_OF_ASC_NODE": raan, "ARG_OF_PERICENTER": argp,
            "MEAN_ANOMALY": ma, "EPHEMERIS_TYPE": 0, "CLASSIFICATION_TYPE": "U", "ELEMENT_SET_NO": 999,
            "REV_AT_EPOCH": 1000, "BSTAR": bstar, "MEAN_MOTION_DOT": ndot, "MEAN_MOTION_DDOT": 0.0}


def synthetic_records(n_payload=400, n_debris=300, seed=0):
    rng = np.random.default_rng(seed)
    out = {"active": [], "cosmos-2251-debris": []}
    for i in range(n_payload):
        alt = rng.choice([550, 780, 1200, 20200, 35786], p=[.55, .2, .1, .1, .05]) + rng.normal(0, 15)
        e = EPOCH - dt.timedelta(hours=float(rng.uniform(0, 72)))
        out["active"].append(make_omm(10000 + i, alt, abs(rng.normal(0.0008, 0.0005)), rng.uniform(0, 99),
                                      rng.uniform(0, 360), rng.uniform(0, 360), rng.uniform(0, 360),
                                      abs(rng.normal(1e-4, 5e-5)), epoch=e, ndot=abs(rng.normal(2e-5, 1e-5))))
    for i in range(n_debris):
        alt = rng.uniform(450, 1000)
        e = EPOCH - dt.timedelta(hours=float(rng.uniform(0, 72)))
        out["cosmos-2251-debris"].append(make_omm(50000 + i, alt, abs(rng.normal(0.01, 0.006)), rng.normal(74, 1),
                                                  rng.uniform(0, 360), rng.uniform(0, 360), rng.uniform(0, 360),
                                                  abs(rng.normal(8e-4, 4e-4)), name=f"COSMOS 2251 DEB",
                                                  epoch=e, ndot=abs(rng.normal(2e-4, 1e-4))))
    return out


def synthetic_catalogue(**kw):
    from orbital_engine.catalog import Catalogue
    return Catalogue.from_records(synthetic_records(**kw))
