import { useState } from 'react'
import { post } from '../api.js'
import { Panel, Notice, Badge, Stat, useJobRunner, Progress } from '../components/ui.jsx'
import Plot from '../components/Plot.jsx'

const POLICIES = ['no_maneuver', 'random', 'naive_along_track', 'mpc_explicit', 'mpc_explicit_plus_learned']

export default function RL() {
  const [cfg, setCfg] = useState({ policy: 'mpc_explicit_plus_learned', seed: 3, budget_ms: 2.0, dv_step_ms: 0.25 })
  const [ep, setEp] = useState(null); const [err, setErr] = useState(null); const [busy, setBusy] = useState(false)
  const [cmp, runCmp] = useJobRunner(); const [k, setK] = useState(0)
  async function episode() { setBusy(true); setErr(null); try { const r = await post('/rl/episode', cfg); setEp(r); setK(r.steps.length) } catch (e) { setErr(e.message) } finally { setBusy(false) } }
  const traj = ep?.trajectory
  return (
    <div className="p-4 grid gap-4 xl:grid-cols-[330px_1fr]">
      <div className="space-y-4">
        <Panel title="Conjunction-avoidance simulator" badge="simulated">
          <div className="text-xs text-slate-400 mb-3">Designed encounters in a two-body + J2 simulator. The planner uses an explicit two-body model, optionally with a <b>learned residual</b> correction fitted to simulator transitions. A research simulation, not an operational controller.</div>
          <div className="grid grid-cols-2 gap-3 text-[13px]">
            <label className="col-span-2"><div className="label">Policy</div><select className="inp w-full" value={cfg.policy} onChange={(e) => setCfg({ ...cfg, policy: e.target.value })}>{POLICIES.map((p) => <option key={p}>{p}</option>)}</select></label>
            <label><div className="label">Scenario seed</div><input className="inp w-full" type="number" value={cfg.seed} onChange={(e) => setCfg({ ...cfg, seed: +e.target.value })} /></label>
            <label><div className="label">Budget m/s</div><input className="inp w-full" type="number" step="0.25" value={cfg.budget_ms} onChange={(e) => setCfg({ ...cfg, budget_ms: +e.target.value })} /></label>
          </div>
          <div className="flex gap-2 mt-3"><button className="btn" disabled={busy} onClick={episode}>{busy ? 'Running… (first learned run trains the model)' : 'Run episode'}</button>
            <button className="btn-ghost" disabled={cmp.running} onClick={() => runCmp(() => post('/rl/compare', { ...cfg, n_scenarios: 8 }))}>Compare policies (8 scenarios)</button></div>
          <div className="mt-2"><Progress s={cmp} /></div>{(err || cmp.error) && <div className="mt-2"><Notice kind="error">{err || cmp.error}</Notice></div>}
        </Panel>
        {ep?.model_report && <Panel title="Learned residual model" badge="ml"><div className="text-[13px]">Held-out one-interval error: explicit only <b>{ep.model_report.holdout_rmse_explicit_only.toExponential(2)}</b> → with residual <b>{ep.model_report.holdout_rmse_with_residual.toExponential(2)}</b> ({ep.model_report.improvement_ratio?.toFixed(1)}× lower), on {ep.model_report.n_transitions} transitions.</div></Panel>}
      </div>
      <div className="space-y-4 min-w-0">
        {!ep && !cmp.result && <Panel><div className="text-sm text-slate-400">Run an episode to watch the chaser's relative trajectory, burns, fuel and separation, or compare policies over seeded scenarios.</div></Panel>}
        {ep && <>
          <Panel title={`Episode: ${ep.policy} (seed ${ep.seed})`} badge="simulated">
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <Stat label="Designed miss (no burn)" value={ep.designed_miss_km.toFixed(2)} unit="km" /><Stat label="Final min separation" value={ep.final_separation_km.toFixed(2)} unit="km" />
              <Stat label="Fuel used" value={ep.fuel_used_ms.toFixed(2)} unit="m/s" /><Stat label="Total reward" value={ep.total_reward.toFixed(2)} /><Stat label="Outcome" value={ep.success ? 'miss ≥ safe threshold' : 'below threshold'} />
            </div>
            <div className="text-[11px] text-slate-500 mt-2">{ep.label}</div>
          </Panel>
          <div className="grid xl:grid-cols-2 gap-4">
            <Panel title="Chaser position relative to target (km, ECI axes)" badge="simulated">
              <label className="text-xs text-slate-400">Step {k}/{ep.steps.length}<input type="range" className="w-full" min="1" max={ep.steps.length} value={k} onChange={(e) => setK(+e.target.value)} /></label>
              <Plot height={360} data={[{ type: 'scatter3d', mode: 'lines+markers', x: traj.rel.slice(0, k + 1).map((p) => p[0]), y: traj.rel.slice(0, k + 1).map((p) => p[1]), z: traj.rel.slice(0, k + 1).map((p) => p[2]), line: { color: '#38bdf8', width: 4 }, marker: { size: 2 }, name: 'chaser' },
                { type: 'scatter3d', mode: 'markers', x: [0], y: [0], z: [0], marker: { size: 6, color: '#fb7185' }, name: 'target' }]} layout={{ margin: { l: 0, r: 0, t: 0, b: 0 }, scene: { aspectmode: 'data' } }} />
            </Panel>
            <Panel title="Separation, predicted miss and budget" badge="simulated">
              <Plot height={360} data={[{ x: ep.steps.map((s) => s.t), y: ep.steps.map((s) => s.separation_km), name: 'separation km', mode: 'lines', yaxis: 'y' }, { x: ep.steps.map((s) => s.t), y: ep.steps.map((s) => s.predicted_miss_km), name: 'predicted miss km (no further burns)', mode: 'lines', line: { dash: 'dot' } },
                { x: ep.steps.map((s) => s.t), y: ep.steps.map((s) => s.budget_remaining_ms), name: 'budget m/s', yaxis: 'y2', mode: 'lines' }]} layout={{ xaxis: { title: 'seconds' }, yaxis: { title: 'km', type: 'log' }, yaxis2: { overlaying: 'y', side: 'right', title: 'm/s' } }} />
            </Panel>
          </div>
          <Panel title="Decision log" badge="simulated"><div className="scroll max-h-60"><table className="t"><thead><tr><th>t s</th><th>Requested</th><th>Executed</th><th>Predicted miss km</th><th>Rel speed km/s</th><th>Budget m/s</th><th>Reward</th></tr></thead><tbody>
            {ep.steps.map((s, i) => <tr key={i}><td>{s.t.toFixed(0)}</td><td>{s.requested}</td><td className={s.executed !== 'maintain' ? 'text-amber-300' : ''}>{s.executed}</td><td>{s.predicted_miss_km.toFixed(2)}</td><td>{s.rel_speed_kms.toFixed(2)}</td><td>{s.budget_remaining_ms.toFixed(2)}</td><td>{s.reward.toFixed(2)}</td></tr>)}</tbody></table></div></Panel>
        </>}
        {cmp.result && <Panel title={`Policy comparison over ${cmp.result.n_scenarios} seeded scenarios`} badge="simulated">
          <table className="t"><thead><tr><th>Policy</th><th>Mean min separation km</th><th>Median km</th><th>Success rate</th><th>Mean fuel m/s</th><th>Mean reward</th></tr></thead><tbody>
            {cmp.result.policies.map((p) => <tr key={p.policy}><td>{p.policy}</td><td>{p.mean_final_separation_km.toFixed(2)}</td><td>{p.median_final_separation_km.toFixed(2)}</td><td>{(p.success_rate * 100).toFixed(0)}%</td><td>{p.mean_fuel_ms.toFixed(2)}</td><td>{p.mean_reward.toFixed(1)}</td></tr>)}</tbody></table>
          <Plot height={260} data={[{ type: 'bar', x: cmp.result.policies.map((p) => p.policy), y: cmp.result.policies.map((p) => p.mean_final_separation_km) }]} layout={{ yaxis: { title: 'mean min separation km' } }} />
          <div className="text-xs text-slate-400">{cmp.result.note} Small sample: treat differences as indicative, not conclusive.</div>
        </Panel>}
      </div>
    </div>
  )
}
