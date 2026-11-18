"""Phase 2-3 tests.

Fixtures: the SGP4 verification TLE for object 00005 (Vallado et al., 'Revisiting
Spacetrack Report #3', public test case) and its published t=0 state vector.
These are published TEST VECTORS for algorithm verification, not dataset content.
HTTP is mocked ONLY to test cache/error logic; no mocked payload is ever used as data.
"""
import datetime as dt, json, time
import numpy as np, pytest

from backend.ingestion.omm import parse_tle, parse_omm_records, validate_fields
from backend.ingestion import celestrak
from backend.ingestion.registry import read_all
from orbital_engine.propagate import propagate, propagate_many, ground_point, to_jd
from orbital_engine.frames import (gmst, teme_to_ecef, ecef_to_geodetic, geodetic_to_ecef,
                                   OMEGA_EARTH)
from orbital_engine.elements import derive

L1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
L2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"
EPOCH = dt.datetime(2000, 6, 27, 18, 50, 19, 733568, tzinfo=dt.timezone.utc)
OMM = {"OBJECT_NAME": "VANGUARD 1", "OBJECT_ID": "1958-002B", "NORAD_CAT_ID": 5,
       "EPOCH": "2000-06-27T18:50:19.733568", "MEAN_MOTION": 10.82419157,
       "ECCENTRICITY": 0.1859667, "INCLINATION": 34.2682, "RA_OF_ASC_NODE": 348.7242,
       "ARG_OF_PERICENTER": 331.7664, "MEAN_ANOMALY": 19.3264, "EPHEMERIS_TYPE": 0,
       "CLASSIFICATION_TYPE": "U", "ELEMENT_SET_NO": 475, "REV_AT_EPOCH": 41366,
       "BSTAR": 0.28098e-4, "MEAN_MOTION_DOT": 0.00000023, "MEAN_MOTION_DDOT": 0}


# ---- TLE parsing + SGP4 ----
def test_tle_parse_fields():
    s = parse_tle(L1, L2)
    assert s.satnum == 5
    assert s.ecco == pytest.approx(0.1859667)
    assert np.degrees(s.inclo) == pytest.approx(34.2682)


def test_sgp4_published_vector_t0():
    r, v = propagate(parse_tle(L1, L2), EPOCH).r_km, propagate(parse_tle(L1, L2), EPOCH).v_kms
    # published SGP4 verification state at tsince=0 (TEME)
    assert r == pytest.approx([7022.46529266, -1400.08296755, 0.03995155], abs=1e-3)
    assert v == pytest.approx([1.893841015, 6.405893759, 4.534807250], abs=1e-5)


def test_sgp4_energy_sane_over_time():
    s = parse_tle(L1, L2)
    radii = [np.linalg.norm(propagate(s, EPOCH + dt.timedelta(minutes=m)).r_km) for m in range(0, 600, 30)]
    assert 6378 < min(radii) and max(radii) < 12000


def test_sgp4_failure_raises_with_reason():
    # extreme drag term on a very low orbit drives SGP4 to its error state
    decaying = {**OMM, "NORAD_CAT_ID": 9, "MEAN_MOTION": 16.3, "ECCENTRICITY": 0.001,
                "INCLINATION": 51.6, "BSTAR": 0.05}
    objs, rej = parse_omm_records([decaying])
    assert objs and not rej
    with pytest.raises(ValueError, match="SGP4 error"):
        propagate(objs[0].satrec, objs[0].epoch_utc + dt.timedelta(days=5))


def test_record_without_object_id_is_still_parsed():
    f = {k: v for k, v in OMM.items() if k != "OBJECT_ID"}
    objs, rej = parse_omm_records([f])
    assert objs and not rej and objs[0].object_id is None


def test_vectorised_matches_scalar():
    s = parse_tle(L1, L2)
    times = [EPOCH + dt.timedelta(minutes=m) for m in (0, 45, 90)]
    err, r, v = propagate_many([s], times)
    for i, t in enumerate(times):
        assert err[0, i] == 0
        assert r[0, i] == pytest.approx(propagate(s, t).r_km, abs=1e-9)


# ---- OMM parsing / validation ----
def test_omm_matches_tle():
    objs, rej = parse_omm_records([OMM])
    assert not rej and objs[0].norad_id == 5
    a = propagate(objs[0].satrec, EPOCH + dt.timedelta(hours=3)).r_km
    b = propagate(parse_tle(L1, L2), EPOCH + dt.timedelta(hours=3)).r_km
    assert np.linalg.norm(a - b) < 0.5  # km; differences = text rounding between formats


@pytest.mark.parametrize("patch,frag", [({"ECCENTRICITY": 1.2}, "eccentricity"),
                                        ({"INCLINATION": 200}, "inclination"),
                                        ({"MEAN_MOTION": 0}, "mean motion"),
                                        ({"EPOCH": "garbage"}, "unparseable"),
                                        ({"BSTAR": None}, "missing BSTAR")])
def test_validation_rejects(patch, frag):
    assert frag in validate_fields({**OMM, **patch})


def test_dedupe_keeps_newest_epoch():
    newer = {**OMM, "EPOCH": "2000-06-28T00:00:00.000000"}
    objs, _ = parse_omm_records([OMM, newer])
    assert len(objs) == 1 and objs[0].epoch_utc.day == 28


# ---- frames ----
def test_gmst_vallado_example():
    # Vallado Ex. 3-5: 1992-08-20 12:14:00 UT1 -> GMST = 152.578787810 deg
    jd = 2448855.009722222
    assert np.degrees(gmst(jd)) % 360 == pytest.approx(152.578787810, abs=1e-4)


def test_geodetic_roundtrip():
    for lat, lon, alt in [(0, 0, 400), (51.5, -0.12, 550), (-33.9, 151.2, 35786), (89.0, 10, 800)]:
        l2, o2, a2 = ecef_to_geodetic(geodetic_to_ecef(lat, lon, alt))
        assert (l2, o2, a2) == pytest.approx((lat, lon, alt), abs=1e-6)


def test_teme_ecef_preserves_radius_and_velocity_frame_effect():
    s = parse_tle(L1, L2); st = propagate(s, EPOCH)
    jd, fr = to_jd(EPOCH)
    r, v = teme_to_ecef(st.r_km, st.v_kms, jd + fr)
    assert np.linalg.norm(r) == pytest.approx(np.linalg.norm(st.r_km), abs=1e-9)
    # Earth-fixed speed differs from inertial by the rotation term (<= omega*r)
    assert abs(np.linalg.norm(v) - np.linalg.norm(st.v_kms)) <= OMEGA_EARTH * np.linalg.norm(r) + 1e-9


def test_ground_point_altitude_consistent_with_elements():
    s = parse_tle(L1, L2)
    d = derive(10.82419157, 0.1859667)
    alts = [ground_point(propagate(s, EPOCH + dt.timedelta(minutes=m)))[2] for m in range(0, 133, 3)]
    assert min(alts) > d.perigee_alt_km - 60 and max(alts) < d.apogee_alt_km + 60


# ---- elements ----
def test_circular_leo_derivation():
    d = derive(15.5, 0.0)           # ~ ISS-like mean motion
    assert 92 < d.period_min < 94
    assert d.perigee_alt_km == pytest.approx(d.apogee_alt_km)
    assert 300 < d.perigee_alt_km < 450


# ---- ingestion: cache policy, errors, provenance ----
class FakeResp:
    def __init__(self, content, ok=True): self.content, self.ok = content, ok
    def raise_for_status(self):
        if not self.ok: raise celestrak.requests.HTTPError("403")


class FakeSession:
    def __init__(self, resp): self.resp, self.calls = resp, 0
    def get(self, *a, **k): self.calls += 1; return self.resp


def test_fetch_caches_and_respects_fair_use(tmp_path):
    sess = FakeSession(FakeResp(json.dumps([OMM]).encode()))
    recs, info = celestrak.fetch_group("stations", tmp_path, session=sess)
    assert len(recs) == 1 and not info.from_cache and sess.calls == 1
    recs2, info2 = celestrak.fetch_group("stations", tmp_path, session=sess)
    assert info2.from_cache and sess.calls == 1            # no second download in window
    log = read_all(tmp_path / "registry" / "source_registry.jsonl")
    assert len(log) == 2 and log[0]["dataset_version"] == log[1]["dataset_version"]


def test_network_failure_without_cache_raises_never_fabricates(tmp_path):
    with pytest.raises(celestrak.CelesTrakError):
        celestrak.fetch_group("stations", tmp_path, session=FakeSession(FakeResp(b"", ok=False)))


def test_non_json_payload_rejected(tmp_path):
    with pytest.raises(celestrak.CelesTrakError):
        celestrak.fetch_group("stations", tmp_path, session=FakeSession(FakeResp(b"GP data has not updated")))


def test_stale_cache_fallback_is_flagged(tmp_path):
    good = FakeSession(FakeResp(json.dumps([OMM]).encode()))
    celestrak.fetch_group("stations", tmp_path, session=good)
    _, info = celestrak.fetch_group("stations", tmp_path, min_interval_s=0,
                                    session=FakeSession(FakeResp(b"", ok=False)))
    assert info.from_cache


def test_history_counts_query_valid_on_sqlite(tmp_path):
    """Regression: nested aggregate max(count(*)) is invalid SQL (found on first real ingestion)."""
    from tests.fixtures import synthetic_catalogue
    from database import store
    import datetime as dt
    eng = store.get_engine(None, tmp_path); cat = synthetic_catalogue(n_payload=5, n_debris=3)
    assert store.history_snapshot_counts(eng)["max_snapshots_per_object"] == 0
    store.upsert_objects(eng, cat); store.append_history(eng, cat, dt.datetime.now(dt.timezone.utc), "x" * 64)
    c = store.history_snapshot_counts(eng)
    assert c["objects"] == 8 and c["max_snapshots_per_object"] == 1
