# Verification report (what was actually run)

| Area | Evidence |
|---|---|
| SGP4 / frames / elements | Published Vallado verification vector (object 00005, t = 0), GMST worked example, geodetic round trips |
| Ingestion | Cache policy, fair-use window, non-JSON and network-failure handling, stale-cache flagging, registry hashes (HTTP mocked; **no live download was possible in the build sandbox**) |
| Screening | TCA vs 0.05 s brute force, altitude pre-filter, planted conjunction found, Pc vs closed-form limits, Pc unavailable without covariance |
| ML | Chronological split no-leakage, all model families run, baseline reported, insufficient-data errors, history targets, MLP gradient check (analytic vs numeric), XOR learned |
| DL (PyTorch) | All six architectures shape-checked, seeded reproducibility, beats persistence on a periodic signal, window/target leakage test, autoencoder anomaly AUC, API jobs end to end |
| RL | Designed miss reproduced without manoeuvre, budget constraint, invalid action, determinism, energy conservation of RK4, residual model lowers held-out error |
| API | 10 endpoint tests via TestClient incl. empty-catalogue message and static frontend serving |
| Frontend | `vite build` succeeds; 20 vitest tests: every page rendered against REAL backend payloads (fixtures from `scripts/gen_fixtures.py`), GMST/sun/axis maths, SatCloud interpolation in the real R3F reconciler. 3-D shaders and WebGL drawing are **not** exercised (no GPU/browser in the build environment) |
| Scale | 19.7k-object synthetic catalogue: positions 0.5 s, 24 h screening 5 s, every ML task completes (regression suite ~100 s) |
| First real run | The user's real CelesTrak download (16,917 active + 2,682 debris objects, 0 rejected) exposed a SQL bug (nested aggregate) that mock-based tests could not; fixed with a regression test |

Tests are synthetic-fixture based (see `tests/fixtures.py`, never written to `data/`). No ML result in this repository
was produced from real CelesTrak data, because none could be downloaded here. Run ingestion, then the ML Lab, and
quote your own numbers.

RL numbers measured on synthetic designed encounters (8 scenarios, seeds 1 and 2, mean minimum separation, km):
no manoeuvre 1.29 / 1.19; naive along-track burns 4.91 / 5.30; explicit-model MPC 1.87 / 2.09; explicit + learned
residual MPC (100 training episodes / 400 epochs; the API trains 90 episodes / 600 epochs) 5.88 / 5.95. With only 24 training episodes the learned planner scored 4.11 / 4.38,
below the naive baseline, so planner quality depends on model fidelity. Small samples; indicative only.
