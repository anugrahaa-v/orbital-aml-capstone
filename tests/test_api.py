import warnings; warnings.filterwarnings("ignore")
import time, base64, numpy as np, pytest
from fastapi.testclient import TestClient
from tests.fixtures import synthetic_catalogue, EPOCH
import backend.main as M
from database import store


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    M.S.data_dir = d; M.S.cat = synthetic_catalogue(n_payload=250, n_debris=200); M.S.eng = store.get_engine(None, d)
    import experiments.common as C; C.RUNS_DIR = d / "runs"
    return TestClient(M.app)


def wait(client, jid, timeout=120):
    t = time.time()
    while time.time() - t < timeout:
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] != "running": return j
        time.sleep(0.3)
    raise TimeoutError


def test_status_and_catalogue(client):
    s = client.get("/api/status").json(); assert s["n_objects"] == 450 and s["classes"]["DEBRIS"] == 200
    c = client.get("/api/catalogue", params={"cls": "DEBRIS", "limit": 5}).json(); assert c["count"] == 5
    nid = c["objects"][0]["norad_id"]
    o = client.get(f"/api/object/{nid}").json(); assert "derived" in o and o["observed_label"].startswith("OBSERVED")
    assert client.get("/api/object/1").status_code == 404


def test_positions_binary_roundtrip(client):
    p = client.get("/api/positions", params={"start": EPOCH.isoformat(), "duration_min": 10, "step_s": 60, "limit": 50}).json()
    a = np.frombuffer(base64.b64decode(p["data_b64"]), np.float32).reshape(p["n_frames"], p["n_objects"], 3)
    r = np.linalg.norm(a, axis=2); assert (r > 6378).all() and (r < 50000).all() and len(p["gmst_rad"]) == p["n_frames"]


def test_orbit_path_closed(client):
    nid = client.get("/api/catalogue", params={"cls": "PAYLOAD", "limit": 1}).json()["objects"][0]["norad_id"]
    p = client.get(f"/api/orbit/{nid}/path", params={"start": EPOCH.isoformat()}).json()
    r = np.array(p["r_teme"]); assert np.linalg.norm(r[0] - r[-1]) < 200 and p["valid"]


def test_collision_endpoints_and_pc_unavailable(client):
    ids = [o["norad_id"] for o in client.get("/api/catalogue", params={"limit": 2}).json()["objects"]]
    c = client.get("/api/collision/closest", params={"a": ids[0], "b": ids[1], "start": EPOCH.isoformat(), "hours": 3}).json()
    assert c["conjunction"]["pc"]["available"] is False and "not an operational" in c["warning"]
    r = client.post("/api/collision/pc", json={"a": ids[0], "b": ids[1], "start": EPOCH.isoformat(), "hours": 3}).json()
    assert r["pc"]["available"] is False
    pid = client.get("/api/catalogue", params={"cls": "DEBRIS", "limit": 1}).json()["objects"][0]["norad_id"]
    s = client.post("/api/collision/screen", json={"primary": pid, "start": EPOCH.isoformat(), "threshold_km": 50, "horizon_h": 2}).json()
    assert "conjunctions" in s and s["n_catalogue"] == 450
    assert client.post("/api/collision/screen", json={"primary": pid, "threshold_km": -1}).status_code == 422


def test_environment_and_nn(client):
    e = client.get("/api/environment/stats").json(); assert "catalogue counts" in e["label"]
    n = client.post("/api/nn/neuron", json={"inputs": [1, 2], "weights": [.5, -1], "bias": .5, "activation": "relu"}).json()
    assert n["output"] == 0
    assert client.post("/api/nn/neuron", json={"inputs": [1], "weights": [1, 2], "activation": "relu"}).status_code == 422
    t = client.post("/api/nn/train", json={"dataset": "xor", "hidden": [6], "lr": 0.5, "epochs": 1500}).json()
    assert t["loss"][-1] < t["loss"][0]


def test_ml_job_and_experiment_logging(client):
    j = client.post("/api/ml/run", json={"task": "regression", "models": ["ridge", "knn"]}).json()["job"]
    r = wait(client, j); assert r["status"] == "done", r
    assert r["result"]["mode"] == "cross_sectional" and "NOT future forecasting" in r["result"]["data_note"]
    assert client.get("/api/experiments", params={"kind": "regression"}).json()["experiments"]
    k = wait(client, client.post("/api/ml/run", json={"task": "kmeans"}).json()["job"]); assert k["status"] == "done"
    assert client.get("/api/analytics/summary").json()["n_runs"] >= 2


def test_rl_episode(client):
    r = client.post("/api/rl/episode", json={"policy": "mpc_explicit", "seed": 2}).json()
    assert r["fuel_used_ms"] <= 2.0 + 1e-9 and r["trajectory"]["rel"]
    assert client.post("/api/rl/episode", json={"policy": "nope"}).status_code == 422


def test_empty_catalogue_gives_actionable_error(tmp_path):
    saved = (M.S.cat, M.S.data_dir, M.S.eng)
    try:
        M.S.cat = None; M.S.data_dir = tmp_path
        c = TestClient(M.app); r = c.get("/api/catalogue"); assert r.status_code == 409 and "ingest" in r.json()["detail"]
        assert c.get("/api/status").json()["catalogue_loaded"] is False
    finally:
        M.S.cat, M.S.data_dir, M.S.eng = saved


def test_built_frontend_is_served_when_present():
    from pathlib import Path
    if not (Path(M.__file__).resolve().parent.parent / "frontend" / "dist" / "index.html").exists():
        pytest.skip("frontend not built")
    import importlib.util
    spec = importlib.util.spec_from_file_location('backend_main_static_copy', M.__file__)
    fresh = importlib.util.module_from_spec(spec); spec.loader.exec_module(fresh)   # separate copy: leaves M.S untouched
    c = TestClient(fresh.app)
    assert c.get("/").status_code == 200 and "ORBITALX" in c.get("/").text
    assert c.get("/textures/day.jpg").headers["content-type"].startswith("image/")


def test_forecast_and_anomaly_jobs_need_torch(client):
    pytest.importorskip("torch")
    ids = [o["norad_id"] for o in client.get("/api/catalogue", params={"cls": "DEBRIS", "limit": 1}).json()["objects"]]
    j = client.post("/api/forecast/run", json={"norad_id": ids[0], "days": 3, "epochs": 4, "models": ["gru"], "step_s": 120}).json()["job"]
    r = wait(client, j, 240); assert r["status"] == "done", r
    res = r["result"]; assert "persistence" in res["models"] and "gru" in res["models"] and "SIMULATED" in res["label"]
    a = wait(client, client.post("/api/dl/anomaly", params={"epochs": 40}).json()["job"], 240)
    assert a["status"] == "done" and a["result"]["roc_auc"] > 0.5


def test_nan_in_payload_does_not_break_json(client):
    """A decayed/invalid element set yields NaN positions; the API must still answer (null), not 500."""
    from tests.fixtures import make_omm
    from orbital_engine.catalog import Catalogue
    bad = make_omm(777, 400, 0.001, 51.6, bstar=0.05); bad["MEAN_MOTION"] = 16.3
    saved = M.S.cat
    try:
        M.S.cat = Catalogue.from_records({"active": [bad]})
        r2 = client.get("/api/orbit/777/path", params={"start": "2026-12-30T00:00:00+00:00"})
        assert r2.status_code == 200 and r2.json()["valid"] is False     # NaN positions serialised as null
    finally:
        M.S.cat = saved


def test_safe_json_replaces_nan():
    from backend.main import SafeJSONResponse
    assert SafeJSONResponse({"a": float("nan"), "b": [float("inf"), 1.0]}).body == b'{"a":null,"b":[null,1.0]}'


def test_dl_endpoints_give_actionable_error_without_torch(client, monkeypatch):
    import builtins, sys
    real = builtins.__import__
    def fake(name, *a, **k):
        if name == "torch": raise ImportError("simulated: no torch")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    r = client.post("/api/dl/anomaly")
    assert r.status_code == 503 and "pip install torch" in r.json()["detail"]


def test_catalogue_is_picked_up_after_late_ingestion(tmp_path):
    """Regression: API started before data existed kept an empty catalogue until restart."""
    import json
    from tests.fixtures import synthetic_records
    saved = (M.S.cat, M.S.data_dir, M.S.eng)
    try:
        M.S.cat = None; M.S.data_dir = tmp_path
        c = TestClient(M.app)
        assert c.get("/api/status").json()["catalogue_loaded"] is False
        (tmp_path / "cache").mkdir()
        (tmp_path / "cache" / "gp_active_20261117T000000Z.json").write_text(json.dumps(synthetic_records(20, 0)["active"]))
        assert c.get("/api/status").json()["n_objects"] == 20          # no restart needed
    finally:
        M.S.cat, M.S.data_dir, M.S.eng = saved
