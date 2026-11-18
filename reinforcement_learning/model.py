"""Learned residual transition model.

Planner model = explicit two-body RK4 (no J2) + a LEARNED residual correction fitted to transitions from the
truth simulator (two-body + J2). The residual MLP is the 'learned' part of the model-based method."""
from __future__ import annotations
import numpy as np
from experiments.mlp import MLP
from .env import ConjunctionEnv, RLConfig, rk4, rtn, ACTION_DIRS

R_SCALE, V_SCALE = 7000.0, 8.0
Y_SCALE = np.array([1e-1] * 3 + [1e-4] * 3)      # residual magnitudes (km, km/s) over one decision interval


def features(states, dv_rtn_kms):
    return np.hstack([states[:, :3] / R_SCALE, states[:, 3:] / V_SCALE, dv_rtn_kms * 1e3])


class ResidualModel:
    def __init__(self, cfg: RLConfig, seed=0):
        self.cfg = cfg; self.net = None; self.seed = seed; self.report = None

    def explicit_step(self, states):
        """two-body only, one decision interval, states (K,6)."""
        s = states.copy()
        for _ in range(int(round(self.cfg.decision_dt_s / self.cfg.substep_s))):
            s = rk4(s, self.cfg.substep_s, j2=False)
        return s

    def collect(self, n_episodes=30, seed=0):
        env, rng = ConjunctionEnv(self.cfg), np.random.default_rng(seed)
        X, Y = [], []
        for ep in range(n_episodes):
            env.reset(seed=10_000 + seed * 1000 + ep)
            for _ in range(self.cfg.n_steps):
                a = int(rng.integers(0, 7)) if rng.random() < 0.5 else 0
                ch0, tg0 = env.chaser.copy(), env.target.copy()
                a_exec = a if env.allowed(a) else 0
                ch_after_burn = env.apply_dv(ch0, a_exec) if a_exec else ch0
                dv = (ACTION_DIRS[a_exec] * self.cfg.dv_step_ms * 1e-3)       # RTN components, km/s
                env.step(a)
                for pre, post, dvv in ((ch_after_burn, env.chaser, dv), (tg0, env.target, np.zeros(3))):
                    exp = self.explicit_step(pre[None])[0]
                    X.append(features(pre[None], dvv[None])[0]); Y.append((post - exp) / Y_SCALE)
        return np.array(X), np.array(Y)

    def fit(self, n_episodes=90, epochs=600, seed=0):
        X, Y = self.collect(n_episodes, seed)
        n = len(X); cut = int(0.8 * n)                      # transitions are ordered by episode => chronological holdout
        self.net = MLP([X.shape[1], 48, 48, 6], hidden="tanh", task="regression", lr=0.05, seed=seed)
        self.net.fit(X[:cut], Y[:cut], epochs=epochs, batch_size=128, seed=seed)
        pred = self.net.predict(X[cut:])
        rm_model = float(np.sqrt(np.mean(((pred - Y[cut:]) * Y_SCALE) ** 2)))
        rm_expl = float(np.sqrt(np.mean((Y[cut:] * Y_SCALE) ** 2)))
        self.report = {"n_transitions": int(n), "holdout_rmse_explicit_only": rm_expl,
                       "holdout_rmse_with_residual": rm_model,
                       "improvement_ratio": rm_expl / rm_model if rm_model > 0 else None,
                       "note": "RMSE of the one-interval next-state error (mixed km / km/s units), held-out episodes"}
        return self.report

    def step(self, states, dv_rtn_kms=None):
        """Explicit step + learned residual (if fitted). states (K,6) AFTER any burn."""
        out = self.explicit_step(states)
        if self.net is not None:
            dv = np.zeros((len(states), 3)) if dv_rtn_kms is None else dv_rtn_kms
            out = out + self.net.predict(features(states, dv)) * Y_SCALE
        return out
