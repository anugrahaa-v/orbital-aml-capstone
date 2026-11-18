"""Episode runner, policy comparison and visualisation payloads."""
from __future__ import annotations
import numpy as np
from .env import ConjunctionEnv, RLConfig, ACTION_NAMES
from .policies import NoManeuver, RandomPolicy, NaiveAlongTrack, ModelBasedMPC
from .model import ResidualModel


def build_policies(cfg: RLConfig, residual: ResidualModel | None, seed: int = 0) -> dict:
    p = {"no_maneuver": NoManeuver(), "random": RandomPolicy(seed), "naive_along_track": NaiveAlongTrack(),
         "mpc_explicit": ModelBasedMPC(cfg, None, seed=seed)}
    if residual is not None and residual.net is not None:
        p["mpc_explicit_plus_learned"] = ModelBasedMPC(cfg, residual, seed=seed)
    return p


def run_episode(env: ConjunctionEnv, policy, seed: int, record: bool = True) -> dict:
    obs = env.reset(seed); policy.reset(env)
    total, steps, done = 0.0, [], False
    traj = {"t": [0.0], "rel": [(env.chaser[:3] - env.target[:3]).tolist()]}
    while not done:
        a, pinfo = policy.act(env, obs)
        obs, r, done, info = env.step(a); total += r
        steps.append({"t": obs["time_s"], "requested": ACTION_NAMES[info["requested_action"]],
                      "executed": ACTION_NAMES[info["executed_action"]], "reward": r,
                      "separation_km": obs["separation_km"], "rel_speed_kms": obs["rel_speed_kms"],
                      "time_to_encounter_s": obs["time_to_encounter_s"], "predicted_miss_km": obs["predicted_miss_km"],
                      "budget_remaining_ms": obs["budget_remaining_ms"], "plan": pinfo})
        if record:
            traj["t"].append(obs["time_s"]); traj["rel"].append(obs["rel_pos_km"].tolist())
    return {"policy": policy.name, "seed": seed, "designed_miss_km": env.designed_miss_km,
            "final_separation_km": info["final_separation_km"], "success": info["success"],
            "fuel_used_ms": info["min_sep_km"] and env.fuel, "total_reward": total,
            "sma_change_km": info["sma_change_km"], "steps": steps, "trajectory": traj if record else None}


def compare_policies(cfg: RLConfig, n_scenarios: int = 10, seed: int = 0, residual: ResidualModel | None = None,
                     policies: list[str] | None = None) -> dict:
    env = ConjunctionEnv(cfg); pols = build_policies(cfg, residual, seed)
    names = policies or list(pols)
    table = {n: [] for n in names}
    for k in range(n_scenarios):
        for n in names:
            ep = run_episode(env, pols[n], seed=1000 + seed * 100 + k, record=False)
            table[n].append((ep["final_separation_km"], env.fuel, ep["total_reward"], ep["success"], ep["designed_miss_km"]))
    rows = []
    for n, v in table.items():
        a = np.array(v, float)
        rows.append({"policy": n, "mean_final_separation_km": float(a[:, 0].mean()), "median_final_separation_km": float(np.median(a[:, 0])),
                     "mean_fuel_ms": float(a[:, 1].mean()), "mean_reward": float(a[:, 2].mean()),
                     "success_rate": float(a[:, 3].mean()), "episodes": len(a),
                     "per_episode": {"final_separation_km": a[:, 0].tolist(), "reward": a[:, 2].tolist()}})
    return {"config": cfg.__dict__, "n_scenarios": n_scenarios, "seed": seed, "policies": rows,
            "note": "Simulation on designed encounters (two-body+J2 truth). Not an operational controller."}
