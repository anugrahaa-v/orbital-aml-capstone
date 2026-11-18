# ORBITALX AI: Phase 1 architecture, data decision, roadmap, risks

## 1. Dataset decision: ONE source, CelesTrak GP

The brief lists five sources. This project uses **only CelesTrak GP** (OMM JSON).

| Why CelesTrak GP | Detail |
|---|---|
| Real observations | Mean orbital elements published from the US Space Force catalogue (via Space-Track) |
| No credentials | Public HTTP queries, no API key |
| SGP4-ready | OMM keywords map directly onto SGP4 initialisation; supports 9-digit catalogue numbers |
| Satellites and debris | Group queries (`active`, plus debris groups such as `cosmos-2251-debris`, `iridium-33-debris`, `fengyun-1c-debris`, `cosmos-1408-debris`) |
| Fair use | One download per ~2 h update cycle; the client is cache-first and logs every retrieval |

Not used: Space-Track, ESA DISCOSweb, NASA ORDEM 3.2, NASA TLE API.

### What the single-source choice changes (stated, not hidden)

| Brief item | Consequence | Valid alternative implemented |
|---|---|---|
| Historical trajectory reconstruction, GP_History | CelesTrak serves **current** element sets only | Each ingestion is stored as a dated snapshot (`orbital_element_history`). History grows with every scheduled run. Chronological targets are built only from real snapshots. Until enough exist, experiments report "insufficient history" instead of inventing data. |
| Object type labels, fragmentation context (DISCOS) | Not in GP | Object class is taken from the CelesTrak group the object came from (payload/debris-group membership). This is a *defensible, documented label*, not ground truth. Name-based `DEB`/`R/B` tagging is flagged as heuristic. |
| Debris flux (ORDEM) | Unavailable | Debris Environment screen shows **catalogue** distributions (altitude density, inclination, type), labelled as catalogue counts, never flux. |
| Probability of collision | GP/TLE carry **no covariance** and no hard-body radius | Pc is always shown as **unavailable**. Screening reports miss distance, relative velocity, time of closest approach. |
| LDA | Needs real labels | Uses group-derived class labels only, with the caveat above. |

## 2. Data lineage categories (enforced in code and UI)

1. **Observed**: OMM fields exactly as downloaded (`fields` dict kept untouched).
2. **Derived**: computed from observed (a, period, perigee/apogee via Kepler; `elements.derive`).
3. **Physics simulation**: SGP4 propagated states (TEME), RL environment dynamics.
4. **ML prediction**: model outputs, always shown beside SGP4 and error.

Every retrieval is logged to `data/registry/source_registry.jsonl` (URL, UTC time, SHA-256 dataset version, record count, access terms, attribution, cache flag).

## 3. Module layout

```
backend/ingestion/   registry.py, celestrak.py (cache-first client), omm.py (parse+validate)
orbital_engine/      propagate.py (SGP4, TEME), frames.py (TEME->ECEF->geodetic), elements.py
collision_engine/    (Phase 4) screening, TCA refinement, relative state
experiments/         (Phase 5-6) sklearn + PyTorch runs, seeds, MLflow
reinforcement_learning/ (Phase 7) dynamics env + model-based planner
database/            (Phase 2b) PostgreSQL schema
frontend/            (Phase 8) React + R3F + Drei + Tailwind + Plotly
tests/
```

Frames: SGP4 output is **TEME**. Conversion to ECEF uses GMST (IAU-1982); UT1≈UTC and polar motion are ignored (documented error budget below 0.5 km Earth-fixed).

## 4. Phase status (what is built and how it was verified)

| Phase | Scope | Status / verification |
|---|---|---|
| 1 | Architecture, source verification, risks | This document |
| 2 | Ingestion, registry, validation, cache, DB history snapshots | Code + tests. **Live CelesTrak download not verified in the build sandbox (no network)**; run `python -m backend.ingest_cli` locally |
| 3 | SGP4, frames (TEME/ECEF/geodetic), derived elements | Tested against published SGP4 verification vector + Vallado GMST example |
| 4 | Conjunction screening (altitude filter, coarse grid, TCA refinement, RTN), Pc only with user covariance | Tested vs brute force, closed-form Pc limits |
| 5 | Regression (13 models), classification, feature selection, PCA/LDA/ICA, K-Means, hierarchical, NN fundamentals, from-scratch MLP | Tested (gradient check, XOR, chronological-split guarantees) |
| 6 | Deep learning (MLP, 1D-CNN, LSTM, GRU, CNN-LSTM, Transformer, autoencoder) | Tested in a CPU PyTorch environment |
| 7 | Model-based RL (J2 truth simulator, explicit + learned-residual planner) | Tested; policy comparison reported honestly |
| 8-9 | React/R3F frontend (9 screens) + FastAPI | API integration-tested; **frontend build-verified only, not browser-tested** |
| 10 | Docker, PostgreSQL schema | Files provided, **not executed** in the build sandbox |

## 4b. Targets (what is predicted, honestly)

* **history mode** (after ingesting on several days): features at snapshot *k*, targets = per-day change in altitude and
  mean motion to snapshot *k+1*. Chronological split.
* **cross-sectional mode** (single snapshot, the situation right after first ingestion): target is the published
  `MEAN_MOTION_DOT` (observed) and a derived altitude-decay rate. This is regression on drag indicators, **not future
  forecasting**; the UI says so.
* **Deep-learning forecasts** run on the SGP4-generated altitude series of a real element set (label SIMULATED). SGP4 is
  the generator, hence the exact reference; the neural models are learned surrogates compared with persistence and a
  linear AR baseline. The real SGP4-vs-observation comparison is the back-test (needs >= 2 snapshots).
* **Anomaly autoencoder**: trained on older PAYLOAD objects; newer DEBRIS-group objects are the "anomalies".

## 5. Main scientific risks

1. **No history in the single source.** Forecast targets (future altitude and mean-motion change) need dated snapshots. Mitigation: scheduled ingestion; report data volume honestly; compare ML against SGP4 rather than assuming ML wins.
2. **TLE accuracy.** Mean elements have km-level error that grows with age. Screening distances are *screening* values, not conjunction assessments.
3. **No covariance, so no Pc.** Hard limit; Pc stays unavailable.
4. **O(N²) screening at catalogue scale.** Needs apogee/perigee filtering and a spatial index (Phase 4).
5. **Label leakage and temporal leakage** in ML. Chronological splits, scalers fit on train only.
6. **Group-derived labels are not ground truth** (affects classification, LDA).
7. **Fair-use limits** on CelesTrak. Cache-first client plus registry.
