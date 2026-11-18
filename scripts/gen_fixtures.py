"""DEV TOOL: dump real API responses (computed on a SYNTHETIC test catalogue) to frontend/test/fixtures/api.json so the
frontend contract tests render every page against genuine backend payloads. Never used at runtime; never shown as data.
Run with PyTorch installed to include the forecast/anomaly payloads."""
import json, sys, time, tempfile, pathlib, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from fastapi.testclient import TestClient
import backend.main as M
from database import store
import experiments.common as C
from tests.fixtures import synthetic_catalogue, EPOCH

d = pathlib.Path(tempfile.mkdtemp()); M.S.data_dir = d; C.RUNS_DIR = d / "runs"
M.S.cat = synthetic_catalogue(n_payload=500, n_debris=300); M.S.eng = store.get_engine(None, d)
M.S.cat.provenance = [{"source": "TEST", "retrieved_utc": "2026-11-17T00:00:00Z", "n_records_raw": 800, "dataset_version": "a" * 64, "from_cache": False}]
c = TestClient(M.app); F = {}


def wait(jid, to=600):
    t = time.time()
    while time.time() - t < to:
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] != "running": return j
        time.sleep(0.5)

F["status"] = c.get("/api/status").json()
objs = c.get("/api/catalogue", params={"limit": 6}).json(); F["catalogue"] = objs
deb = c.get("/api/catalogue", params={"cls": "DEBRIS", "limit": 1}).json()["objects"][0]
pay = c.get("/api/catalogue", params={"cls": "PAYLOAD", "limit": 1}).json()["objects"][0]
F["debris"], F["payload"] = deb, pay
F["object_detail"] = c.get(f"/api/object/{pay['norad_id']}").json()
F["positions"] = c.get("/api/positions", params={"start": EPOCH.isoformat(), "limit": 40, "duration_min": 10, "step_s": 60}).json()
F["orbit_path"] = c.get(f"/api/orbit/{pay['norad_id']}/path", params={"start": EPOCH.isoformat()}).json()
F["env_stats"] = c.get("/api/environment/stats").json()
F["collision_closest"] = c.get("/api/collision/closest", params={"a": deb["norad_id"], "b": pay["norad_id"], "start": EPOCH.isoformat(), "hours": 3}).json()
F["collision_screen"] = c.post("/api/collision/screen", json={"primary": deb["norad_id"], "start": EPOCH.isoformat(), "threshold_km": 60, "horizon_h": 3}).json()
F["pc"] = c.post("/api/collision/pc", json={"a": deb["norad_id"], "b": pay["norad_id"], "start": EPOCH.isoformat(), "hours": 3, "cov_a_km2": [[.25, 0, 0], [0, .25, 0], [0, 0, .25]], "cov_b_km2": [[.25, 0, 0], [0, .25, 0], [0, 0, .25]], "hbr_km": 0.01}).json()
F["history"] = {"label": "x", "n_snapshots": 1, "epochs": [], "mean_motion": [], "backtest": None, "note": "Only one snapshot stored."}
F["nn_curve"] = c.get("/api/nn/curve/sigmoid").json()
F["nn_neuron"] = c.post("/api/nn/neuron", json={"inputs": [.5, -1, 2], "weights": [.8, .4, -.3], "bias": .2, "activation": "sigmoid"}).json()
F["nn_train"] = c.post("/api/nn/train", json={"dataset": "xor", "hidden": [4], "lr": .5, "epochs": 800}).json()
for task in ["regression", "classification", "feature_selection", "dimred", "kmeans", "hierarchical"]:
    r = wait(c.post("/api/ml/run", json={"task": task, "models": ["ridge", "random_forest", "knn"] if task == "regression" else None}).json()["job"])
    assert r["status"] == "done", (task, r); F[f"ml_{task}"] = r["result"]
F["rl_episode"] = c.post("/api/rl/episode", json={"policy": "mpc_explicit", "seed": 2}).json()
r = wait(c.post("/api/rl/compare", json={"n_scenarios": 2, "seed": 1, "train_model": False}).json()["job"]); assert r["status"] == "done", r; F["rl_compare"] = r["result"]
try:
    import torch  # noqa
    r = wait(c.post("/api/forecast/run", json={"norad_id": deb["norad_id"], "days": 3, "epochs": 3, "models": ["gru", "mlp"]}).json()["job"]); assert r["status"] == "done", r; F["forecast"] = r["result"]
    r = wait(c.post("/api/dl/anomaly", params={"epochs": 30}).json()["job"]); assert r["status"] == "done", r; F["anomaly"] = r["result"]
except ImportError:
    print("torch missing: forecast/anomaly fixtures skipped")
F["analytics"] = c.get("/api/analytics/summary").json()
out = pathlib.Path("frontend/test/fixtures/api.json"); out.write_text(json.dumps(F)); print("wrote", out, round(out.stat().st_size / 1e6, 2), "MB", sorted(F))
