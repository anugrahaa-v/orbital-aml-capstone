import React, { useMemo, useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { Line3 } from '../scene.jsx'
import { post, useJob } from '../api.js'
import { Panel, Badge, Notice, Spinner, Plot, Table, Stat, Field, fmt, COLORS } from '../ui.jsx'

const POLICIES = ['no_maneuver', 'random', 'naive_along_track', 'mpc_explicit', 'mpc_explicit_plus_learned']
const SC = 0.02   // 1 unit = 50 km

function RelScene({ ep, k, safe }) {
  const pts = useMemo(() => ep.trajectory.rel.map(p => [p[0] * SC, p[2] * SC, -p[1] * SC]), [ep])
  const m = useRef()
  useFrame(() => { if (m.current) m.current.position.set(...pts[Math.min(k.current, pts.length - 1)]) })
  return <group>
    <Line3 points={pts} color="#ff8a5b" />
    <mesh><sphereGeometry args={[0.08, 16, 16]} /><meshBasicMaterial color="#4cc2ff" /></mesh>
    <mesh><sphereGeometry args={[safe * SC, 24, 24]} /><meshBasicMaterial color="#4cd6a0" wireframe transparent opacity={0.4} /></mesh>
    <mesh ref={m}><sphereGeometry args={[0.07, 12, 12]} /><meshBasicMaterial color="#ff8a5b" /></mesh>
    <gridHelper args={[40, 40, '#223', '#111']} />
  </group>
}

export default function RL() {
  const [policy, setPolicy] = useState('mpc_explicit_plus_learned'); const [seed, setSeed] = useState(1)
  const [budget, setBudget] = useState(2.0); const [dv, setDv] = useState(0.25)
  const [ep, setEp] = useState(null); const [busy, setBusy] = useState(false); const [err, setErr] = useState(null)
  const [job, run] = useJob(); const [step, setStep] = useState(0); const k = useRef(0)
  const go = async () => { setBusy(true); setErr(null); try { const r = await post('/api/rl/episode', { policy, seed: +seed, budget_ms: +budget, dv_step_ms: +dv }); setEp(r); setStep(0); k.current = 0 } catch (e) { setErr(e.message) } setBusy(false) }
  const t = ep?.steps
  return (
    <div className="h-full overflow-auto p-4 grid grid-cols-12 gap-4 content-start">
      <div className="col-span-12 flex items-center gap-3 flex-wrap"><h2 className="text-lg font-semibold">Autonomous Manoeuvre Simulator</h2><Badge kind="simulated">simulation · designed encounters</Badge></div>
      <div className="col-span-12"><Notice kind="warn">Research simulation only: two-body + J2 truth, scenarios are <b>designed</b> encounters, the planner uses an explicit two-body model plus a learned residual. Not an operational collision-avoidance controller.</Notice></div>
      <div className="col-span-12 flex gap-3 items-end flex-wrap">
        <Field label="Policy"><select className="inp" value={policy} onChange={e => setPolicy(e.target.value)}>{POLICIES.map(p => <option key={p}>{p}</option>)}</select></Field>
        <Field label="Scenario seed"><input className="inp w-20" type="number" value={seed} onChange={e => setSeed(e.target.value)} /></Field>
        <Field label="Budget (m/s)"><input className="inp w-20" value={budget} onChange={e => setBudget(e.target.value)} /></Field>
        <Field label="Δv per burn (m/s)"><input className="inp w-20" value={dv} onChange={e => setDv(e.target.value)} /></Field>
        <button className="btn btn-primary" onClick={go} disabled={busy}>Run episode</button>{busy && <Spinner text="Planning (first learned run also trains the model)" />}
        <button className="btn" onClick={() => run('/api/rl/compare', { n_scenarios: 8, seed: +seed, budget_ms: +budget, dv_step_ms: +dv, train_model: true })} disabled={job.status === 'running'}>Compare all policies (8 scenarios)</button>{job.status === 'running' && <Spinner text="Comparing" />}
      </div>
      {(err || job.error) && <div className="col-span-12"><Notice>{err || job.error}</Notice></div>}
      {ep && <>
        <div className="col-span-12 flex gap-3 flex-wrap">
          <Stat label="Final min separation" value={fmt(ep.final_separation_km, 3)} unit="km" hint={`designed miss ${fmt(ep.designed_miss_km, 2)} km`} />
          <Stat label="Success (≥ safe miss)" value={ep.success ? 'yes' : 'no'} hint={`threshold ${ep.config.safe_miss_km} km`} />
          <Stat label="Fuel used" value={fmt(ep.fuel_used_ms, 2)} unit="m/s" /><Stat label="Total reward" value={fmt(ep.total_reward, 2)} />
          <Stat label="Δ semi-major axis" value={fmt(ep.sma_change_km, 2)} unit="km" />
        </div>
        <Panel className="col-span-5" title="Relative trajectory (chaser around target)">
          <div className="h-[300px] rounded-lg overflow-hidden"><Canvas camera={{ position: [10, 8, 12], fov: 50 }}><color attach="background" args={['#02050f']} /><RelScene ep={ep} k={k} safe={ep.config.safe_miss_km} /><OrbitControls /></Canvas></div>
          <input type="range" className="w-full mt-2" min={0} max={ep.trajectory.rel.length - 1} value={step} onChange={e => { setStep(+e.target.value); k.current = +e.target.value }} />
          <div className="text-[10px] text-slate-500">Grid = 50 km per square · green sphere = safe-miss radius · blue = target</div>
        </Panel>
        <Panel className="col-span-7" title="Episode state over time">
          <Plot height={300} data={[
            { x: t.map(s => s.t), y: t.map(s => s.separation_km), name: 'separation km', mode: 'lines+markers', yaxis: 'y', line: { color: COLORS[0] } },
            { x: t.map(s => s.t), y: t.map(s => s.predicted_miss_km), name: 'predicted miss km', mode: 'lines', yaxis: 'y', line: { color: COLORS[2], dash: 'dot' } },
            { x: t.map(s => s.t), y: t.map(s => s.budget_remaining_ms), name: 'budget m/s', mode: 'lines', yaxis: 'y2', line: { color: COLORS[1] } }]}
            layout={{ xaxis: { title: 'time s' }, yaxis: { title: 'km', type: 'log' }, yaxis2: { overlaying: 'y', side: 'right', title: 'm/s', showgrid: false } }} />
        </Panel>
        <Panel className="col-span-12" title="Decisions">
          <Table max={260} columns={[{ key: 't', label: 't (s)' }, { key: 'requested', label: 'Requested' }, { key: 'executed', label: 'Executed' }, { key: 'sep', label: 'Sep km', render: s => fmt(s.separation_km, 1) }, { key: 'rv', label: 'Rel km/s', render: s => fmt(s.rel_speed_kms, 2) },
            { key: 'tte', label: 'Time to enc. s', render: s => fmt(s.time_to_encounter_s, 0) }, { key: 'pm', label: 'Pred. miss km', render: s => fmt(s.predicted_miss_km, 2) }, { key: 'r', label: 'Reward', render: s => fmt(s.reward, 2) },
            { key: 'p', label: 'Planner pred. min sep', render: s => s.plan?.predicted_min_sep_km ? fmt(s.plan.predicted_min_sep_km, 2) : '–' }]} rows={t} />
        </Panel>
      </>}
      {job.result && <Panel className="col-span-12" title="Policy comparison" right={<Badge kind="simulated">{job.result.n_scenarios} designed scenarios · seed {job.result.seed}</Badge>}>
        <Table columns={[{ key: 'policy', label: 'Policy' }, { key: 'a', label: 'Mean final sep km', render: r => fmt(r.mean_final_separation_km, 3) }, { key: 'b', label: 'Median km', render: r => fmt(r.median_final_separation_km, 3) },
          { key: 'c', label: 'Success rate', render: r => `${fmt(r.success_rate * 100, 0)}%` }, { key: 'd', label: 'Mean fuel m/s', render: r => fmt(r.mean_fuel_ms, 2) }, { key: 'e', label: 'Mean reward', render: r => fmt(r.mean_reward, 2) }]} rows={job.result.policies} />
        {job.result.model_report && <div className="text-xs text-slate-400 mt-2">Learned residual model: held-out one-interval RMSE {fmt(job.result.model_report.holdout_rmse_explicit_only, 5)} → {fmt(job.result.model_report.holdout_rmse_with_residual, 5)} (×{fmt(job.result.model_report.improvement_ratio, 2)} better) on {job.result.model_report.n_transitions} transitions.</div>}
      </Panel>}
    </div>
  )
}
