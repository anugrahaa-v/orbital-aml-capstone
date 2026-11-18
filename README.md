# ORBITALX AI

Orbital debris monitoring, geometric conjunction screening, ML/DL forecasting and model-based RL research platform
with a 3D observatory. **Single dataset: CelesTrak GP** (public, no key). See `documentation/ARCHITECTURE.md` and
`documentation/VERIFICATION.md` (what was and was not verified).

## Quick start

### Windows (PowerShell)
```powershell
cd orbitalx
.\scripts\setup_windows.ps1          # venv + core packages (add -WithTorch for deep learning + XGBoost)
.\scripts\run.ps1                    # downloads CelesTrak data on first run, then serves http://localhost:8000
```
Manual equivalent:
```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
python -m backend.ingest_cli --groups active cosmos-2251-debris iridium-33-debris fengyun-1c-debris cosmos-1408-debris
uvicorn backend.main:app --port 8000
```

### Linux / macOS
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m backend.ingest_cli --groups active cosmos-2251-debris iridium-33-debris fengyun-1c-debris cosmos-1408-debris
uvicorn backend.main:app --port 8000
```

Optional: `pip install -r requirements-optional.txt` (PyTorch for the Forecast & DL page, XGBoost). The app runs without them
and tells you what is missing. Open **http://localhost:8000** (first load propagates thousands of objects: allow 10-20 s).

Tests: `python -m pytest -q` (backend) and `cd frontend && npm install --legacy-peer-deps && npm test` (frontend contract tests).
Frontend dev: `cd frontend && npm run dev` (http://localhost:5173, proxies /api). After edits `npm run build` refreshes what port 8000 serves.
Run `scripts/ingest_daily.sh` (or `.ps1`) once a day to accumulate real history: this unlocks chronological forecasting
targets and the observed-vs-SGP4 back-test.

## Screens
Mission (landing) · Observatory (3D, real catalogue, SGP4) · Collision Lab · Debris Environment · Forecast & DL ·
ML Lab (13 regressors, classification, feature selection, PCA/LDA/ICA, K-Means, hierarchical) · Neural Nets · RL Planner · Analytics.

## Data honesty rules (enforced in code and UI)
* Every value is labelled OBSERVED, DERIVED, SIMULATED or ML PREDICTION.
* No fabricated data: failed downloads raise errors; with no catalogue the UI shows an empty state, not fake objects.
* Collision probability is **unavailable** (GP has no covariance); only screening geometry is reported. Never operational.
* Object classes come from CelesTrak group membership (name heuristic for rocket bodies), not ground truth.
* Earth textures: NASA imagery via the three-globe package (MIT), see `frontend/public/textures/ATTRIBUTION.txt`.

## Layout
`backend/` API + ingestion · `orbital_engine/` SGP4, frames, catalogue, back-test · `collision_engine/` screening, Pc ·
`experiments/` ML, DL, tracking · `reinforcement_learning/` simulator, residual model, planners · `database/` models + Postgres DDL ·
`frontend/` React app (dist prebuilt) · `tests/` · `Dockerfile`, `docker-compose.yml` (PostgreSQL), `.github/workflows/ci.yml`.
