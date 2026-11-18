import numpy as np, pytest
from reinforcement_learning.env import ConjunctionEnv, RLConfig, accel, rk4, MU, RE, ACTION_NAMES, sma
from reinforcement_learning.policies import NoManeuver, NaiveAlongTrack, ModelBasedMPC
from reinforcement_learning.run import run_episode, compare_policies
from reinforcement_learning.model import ResidualModel


def test_scenario_designed_miss_is_reproduced_without_maneuver():
    env = ConjunctionEnv()
    for seed in range(5):
        ep = run_episode(env, NoManeuver(), seed, record=False)
        assert ep["final_separation_km"] == pytest.approx(ep["designed_miss_km"], abs=0.05)
        assert ep["fuel_used_ms"] == 0


def test_reset_is_deterministic():
    e1, e2 = ConjunctionEnv(), ConjunctionEnv()
    e1.reset(3); e2.reset(3)
    assert np.allclose(e1.chaser, e2.chaser) and np.allclose(e1.target, e2.target)


def test_energy_conserved_by_two_body_rk4():
    s = np.array([7000.0, 0, 0, 0, np.sqrt(MU / 7000.0), 0])
    E0 = 0.5 * s[3:] @ s[3:] - MU / np.linalg.norm(s[:3])
    for _ in range(600): s = rk4(s, 10.0, j2=False)
    E1 = 0.5 * s[3:] @ s[3:] - MU / np.linalg.norm(s[:3])
    assert abs(E1 - E0) / abs(E0) < 1e-9


def test_budget_constraint_enforced():
    cfg = RLConfig(budget_ms=0.5, dv_step_ms=0.25)
    env = ConjunctionEnv(cfg); env.reset(1)
    infos = [env.step(1)[3] for _ in range(4)]
    assert env.fuel == pytest.approx(0.5)                  # never exceeds budget
    assert [i["constraint_applied"] for i in infos] == [False, False, True, True]
    assert all(i["executed_action"] == 0 for i in infos[2:])


def test_invalid_action_rejected():
    env = ConjunctionEnv(); env.reset(0)
    with pytest.raises(ValueError): env.step(9)


def test_action_set_matches_spec():
    assert ACTION_NAMES[0] == "maintain" and any("along" in n for n in ACTION_NAMES) \
        and any("radial" in n for n in ACTION_NAMES) and any("cross" in n for n in ACTION_NAMES)


def test_along_track_burn_changes_sma_and_fuel():
    env = ConjunctionEnv(); env.reset(2); a0 = sma(env.chaser)
    env.step(1)
    assert sma(env.chaser) > a0 and env.fuel == pytest.approx(env.cfg.dv_step_ms)


def test_observation_contains_required_state():
    o = ConjunctionEnv().reset(4)
    for k in ["chaser_pos_km", "chaser_vel_kms", "rel_pos_km", "rel_vel_kms", "separation_km", "rel_speed_kms",
              "time_to_encounter_s", "budget_remaining_ms"]:
        assert k in o


def test_mpc_beats_no_maneuver_on_average():
    cfg = RLConfig()
    res = compare_policies(cfg, n_scenarios=4, seed=1, policies=["no_maneuver", "mpc_explicit"])
    by = {r["policy"]: r for r in res["policies"]}
    assert by["mpc_explicit"]["mean_final_separation_km"] > by["no_maneuver"]["mean_final_separation_km"]
    assert by["mpc_explicit"]["mean_fuel_ms"] <= cfg.budget_ms + 1e-9


def test_residual_model_reduces_one_step_error():
    m = ResidualModel(RLConfig(), seed=0)
    rep = m.fit(n_episodes=12, epochs=150, seed=0)
    assert rep["holdout_rmse_with_residual"] < rep["holdout_rmse_explicit_only"]
