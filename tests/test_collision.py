import datetime as dt, numpy as np, pytest
from tests.fixtures import make_omm, EPOCH, synthetic_catalogue
from orbital_engine.catalog import Catalogue
from collision_engine.screening import (closest_approach, screen_primary, ScreeningConfig, category,
                                        rtn_basis, encounter_plane_pc, pc_or_unavailable, distance_series,
                                        altitude_prefilter)


def two_crossing():
    # circular orbits, different inclination, sharing an ascending node; B arrives ~1 s after A
    a = make_omm(1, 700, 0.0, 51.6, 0, 0, 0, bstar=0, ndot=0)
    b = make_omm(2, 700, 0.0, 98.0, 0, 0, -0.05, bstar=0, ndot=0)
    return Catalogue.from_records({"active": [a, b]})


def test_closest_approach_matches_bruteforce():
    cat = two_crossing(); A, B = cat.get(1), cat.get(2)
    t0 = EPOCH - dt.timedelta(minutes=5)
    c = closest_approach(A, B, t0, 600, step_s=20)
    offs, ra, rb, dist, rs, ok = distance_series(A, B, EPOCH, 60, 0.05)
    brute = np.nanmin(dist)
    assert c.miss_distance_km == pytest.approx(brute, abs=0.05)
    assert abs((c.tca_utc - EPOCH).total_seconds()) < 10
    assert c.relative_speed_kms > 5          # crossing orbits => km/s relative speed
    assert c.uncertainty_available is False and c.pc["available"] is False


def test_rtn_basis_orthonormal():
    r = np.array([7000., 100, 50]); v = np.array([0.1, 7.5, 1.0])
    M = rtn_basis(r, v)
    assert M @ M.T == pytest.approx(np.eye(3), abs=1e-12)


def test_category_bins_and_threshold_config():
    cfg = ScreeningConfig(threshold_km=8, category_edges_km=(0.5, 2.0))
    assert category(0.3, cfg) == "CLOSE_APPROACH" and category(1.0, cfg) == "ELEVATED" and category(5, cfg) == "WATCH"


def test_screen_primary_finds_planted_conjunction():
    a = make_omm(1, 700, 0.0, 51.6, 0, 0, 0, bstar=0, ndot=0)
    b = make_omm(2, 700, 0.0, 98.0, 0, 0, -0.05, bstar=0, ndot=0)
    far = make_omm(3, 20200, 0.0, 55, 40, 0, 0)               # MEO: removed by altitude filter
    cat = Catalogue.from_records({"active": [a, b, far]})
    cfg = ScreeningConfig(threshold_km=20, horizon_h=0.2, coarse_step_s=30)
    out = screen_primary(cat, cat.get(1), EPOCH - dt.timedelta(minutes=5), cfg)
    assert out["n_candidates_after_altitude_filter"] == 1
    ids = [c["secondary_id"] for c in out["conjunctions"]]
    assert ids == [2] and out["conjunctions"][0]["miss_distance_km"] < 20


def test_altitude_prefilter_excludes_disjoint_shells():
    cat = synthetic_catalogue(n_payload=100, n_debris=50)
    p = cat.filter(obj_class="PAYLOAD")[0]
    idx = altitude_prefilter(cat, p, 30, 10)
    for i in idx:
        o = cat.objects[i]
        assert max(o.perigee_km, p.perigee_km) - min(o.apogee_km, p.apogee_km) <= 40 + 1e-9


# ---- Pc ----
def test_pc_zero_miss_isotropic_closed_form():
    s, R = 0.5, 0.01
    cov = np.eye(3) * s ** 2 / 2          # a + b => sigma^2 = s^2 per axis
    pc = encounter_plane_pc([0, 0, 0.0], [0, 7.5, 0], cov, cov, R)
    assert pc == pytest.approx(1 - np.exp(-R ** 2 / (2 * s ** 2)), rel=1e-4)


def test_pc_offset_small_hbr_approximation():
    s, R, d = 1.0, 0.005, 1.2
    cov = np.eye(3) * s ** 2 / 2
    pc = encounter_plane_pc([d, 0, 0.0], [0, 7.5, 0], cov, cov, R)
    approx = R ** 2 / (2 * s ** 2) * np.exp(-d ** 2 / (2 * s ** 2))
    assert pc == pytest.approx(approx, rel=2e-3)


def test_pc_unavailable_without_covariance():
    cat = two_crossing()
    c = closest_approach(cat.get(1), cat.get(2), EPOCH - dt.timedelta(minutes=5), 600)
    assert pc_or_unavailable(c)["available"] is False
    r = pc_or_unavailable(c, np.eye(3) * 0.25, np.eye(3) * 0.25, 0.01)
    assert r["available"] and 0 <= r["pc"] <= 1


def test_pc_rejects_singular_covariance():
    with pytest.raises(ValueError):
        encounter_plane_pc([1, 0, 0], [0, 7, 0], np.zeros((3, 3)), np.zeros((3, 3)), 0.01)
