"""Policies: baselines + receding-horizon model-based planner (random shooting + CEM-style refinement)."""
from __future__ import annotations
import numpy as np
from .env import ConjunctionEnv, RLConfig, rtn, ACTION_DIRS, seg_min, sma
from .model import ResidualModel


class NoManeuver:
    name = "no_maneuver"
    def reset(self, env): pass
    def act(self, env, obs): return 0, {}


class RandomPolicy:
    name = "random"
    def __init__(self, seed=0): self.rng = np.random.default_rng(seed)
    def reset(self, env): pass
    def act(self, env, obs): return int(self.rng.integers(0, 7)), {}


class NaiveAlongTrack:
    """Simple baseline: burn along-track (+T) at every step until the budget is exhausted."""
    name = "naive_along_track"
    def reset(self, env): pass
    def act(self, env, obs): return 1, {}


class ModelBasedMPC:
    """Receding-horizon planner. Scores action sequences by rolling the MODEL forward:
    risk(predicted min separation) + fuel + semi-major-axis deviation. Executes the first action only."""
    def __init__(self, cfg: RLConfig, residual: ResidualModel | None = None, horizon=8, n_candidates=96, iters=2, seed=0):
        self.cfg, self.res, self.H, self.K, self.iters = cfg, residual, horizon, n_candidates, iters
        self.rng = np.random.default_rng(seed)
        self.name = "mpc_explicit_plus_learned" if residual is not None and residual.net is not None else "mpc_explicit"

    def reset(self, env): self.rng = np.random.default_rng(env.seed + 7)

    def _model_step(self, ch, tg):
        if self.res is not None and self.res.net is not None:
            return self.res.step(ch), self.res.step(tg[None])[0]
        return self.res_free(ch), self.res_free(tg[None])[0]

    def res_free(self, s):
        from .env import rk4
        out = s.copy()
        for _ in range(int(round(self.cfg.decision_dt_s / self.cfg.substep_s))):
            out = rk4(out, self.cfg.substep_s, j2=False)
        return out

    def _evaluate(self, env, seqs):
        c, K = self.cfg, len(seqs)
        ch = np.tile(env.chaser, (K, 1)); tg = env.target.copy()
        fuel = np.full(K, env.fuel); used = np.zeros(K)
        best = np.full(K, env.min_sep)
        steps_left = c.n_steps - env.step_i
        for h in range(steps_left):
            acts = seqs[:, h] if h < seqs.shape[1] else np.zeros(K, int)
            ok = (acts != 0) & ((c.budget_ms - fuel) >= c.dv_step_ms - 1e-12)
            acts = np.where(ok, acts, 0)
            dv_rtn = ACTION_DIRS[acts] * c.dv_step_ms * 1e-3
            for k in np.nonzero(acts)[0]:
                ch[k, 3:] += dv_rtn[k] @ rtn(ch[k, :3], ch[k, 3:])
            fuel += np.where(acts != 0, c.dv_step_ms, 0.0); used += np.where(acts != 0, c.dv_step_ms, 0.0)
            # sub-sample min separation inside the interval using the model's (explicit) substeps
            from .env import rk4
            c0, t0 = ch.copy(), np.broadcast_to(tg, ch.shape).copy()
            for _ in range(int(round(c.decision_dt_s / c.substep_s))):
                d, _t = seg_min(c0[:, :3] - t0[:, :3], c0[:, 3:] - t0[:, 3:], c.substep_s)
                best = np.minimum(best, d)
                c0, t0 = rk4(c0, c.substep_s, False), rk4(t0, c.substep_s, False)
            ch_new, tg_new = self._model_step_batch(ch, tg, dv_rtn)
            ch, tg = ch_new, tg_new
        dev = np.abs(sma(ch) - env.a0)
        score = -(c.w_risk * np.exp(-best / c.risk_scale_km) + c.w_fuel * used + c.w_dev * dev)
        return score, best

    def _model_step_batch(self, ch, tg, dv_rtn):
        if self.res is not None and self.res.net is not None:
            return self.res.step(ch, dv_rtn), self.res.step(tg[None])[0]
        return self.res_free(ch), self.res_free(tg[None])[0]

    def act(self, env, obs):
        steps_left = self.cfg.n_steps - env.step_i
        H = min(self.H, steps_left)
        seqs = self.rng.integers(0, 7, size=(self.K, H))
        seqs[0] = 0
        row = 1
        for a in range(1, 7):                    # structured candidates: burn action `a` for m steps, then coast
            for m in range(1, H + 1):
                if row >= self.K: break
                seqs[row, :] = 0; seqs[row, :m] = a; row += 1
        elite = None
        for it in range(self.iters):
            score, pred_min = self._evaluate(env, seqs)
            order = np.argsort(-score); elite = seqs[order[:max(4, self.K // 6)]]
            if it < self.iters - 1:              # CEM-style: resample around the elites
                new = elite[self.rng.integers(0, len(elite), size=self.K)].copy()
                mask = self.rng.random(new.shape) < 0.3
                new[mask] = self.rng.integers(0, 7, size=mask.sum()); new[0] = 0
                seqs = new
        i = int(order[0])
        info = {"predicted_min_sep_km": float(pred_min[i]), "score": float(score[i]),
                "planned_sequence": [int(x) for x in seqs[i]]}
        return int(seqs[i, 0]), info
