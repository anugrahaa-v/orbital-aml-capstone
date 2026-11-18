import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { Line3 } from '../scene.jsx'
import { api, post } from '../api.js'
import { Panel, Badge, Notice, Spinner, Stat, Field, ObjectPicker, Plot, Table, fmt, NeedData, CLASS_COLOR } from '../ui.jsx'

const S = 0.05   // 1 scene unit = 20 km
function RelScene({ rel, idx }) {
  const pts = useMemo(() => rel.map(p => [p[0] * S, p[2] * S, -p[1] * S]), [rel])
  const m = useRef()
  useFrame(() => { if (m.current) m.current.position.set(...pts[Math.min(idx.current, pts.length - 1)]) })
  const tca = pts[Math.floor(pts.length / 2)]
  return <group>
    <Line3 points={pts} color="#ff8a5b" />
    <mesh><sphereGeometry args={[0.12, 16, 16]} /><meshBasicMaterial color="#4cc2ff" /></mesh>
    <mesh ref={m}><sphereGeometry args={[0.1, 16, 16]} /><meshBasicMaterial color="#ff8a5b" /></mesh>
    <mesh position={tca}><ringGeometry args={[0.18, 0.2, 32]} /><meshBasicMaterial color="#ffffff" side={THREE.DoubleSide} /></mesh>
    <gridHelper args={[20, 20, '#223', '#112']} />
  </group>
}

export default function Collision({ status, select }) {
  const [A, setA] = useState(select || null)
  const [B, setB] = useState(null)
  const [hours, setHours] = useState(24)
  const [res, setRes] = useState(null)
  const [err, setErr] = useState(null)
  const [busy, setBusy] = useState(false)
  const [screen, setScreen] = useState(null)
  const [thr, setThr] = useState(10)
  const [sbusy, setSbusy] = useState(false)
  const [pcIn, setPcIn] = useState({ sa: 0.5, sb: 0.5, hbr: 0.01 })
  const [pc, setPc] = useState(null)
  const [step, setStep] = useState(0)
  const idx = useRef(0)
  useEffect(() => { if (select && !A) setA(select) }, [select])  // eslint-disable-line

  const compute = async (a = A, b = B) => {
    if (!a || !b) return
    setBusy(true); setErr(null); setPc(null)
    try { const r = await api(`/api/collision/closest?a=${a.norad_id}&b=${b.norad_id}&start=${new Date().toISOString()}&hours=${hours}`); setRes(r); idx.current = 0; setStep(0) }
    catch (e) { setErr(e.message); setRes(null) }
    setBusy(false)
  }
  const runScreen = async () => {
    if (!A) return; setSbusy(true); setErr(null)
    try { setScreen(await post('/api/collision/screen', { primary: A.norad_id, start: new Date().toISOString(), threshold_km: +thr, horizon_h: +hours })) }
    catch (e) { setErr(e.message) }
    setSbusy(false)
  }
  const runPc = async () => {
    const iso = s => [[s * s, 0, 0], [0, s * s, 0], [0, 0, s * s]]
    try { setPc(await post('/api/collision/pc', { a: A.norad_id, b: B.norad_id, start: new Date().toISOString(), hours: +hours, cov_a_km2: iso(+pcIn.sa), cov_b_km2: iso(+pcIn.sb), hbr_km: +pcIn.hbr })) } catch (e) { setErr(e.message) }
  }
  const rel = useMemo(() => res ? res.series.r_b_teme.map((b, i) => b.map((v, k) => v - res.series.r_a_teme[i][k])) : null, [res])
  useEffect(() => { idx.current = step }, [step])
  const c = res?.conjunction
  const tcaTxt = c ? new Date(c.tca_utc).toISOString().replace('T', ' ').slice(0, 19) : ''

  return (
    <div className="h-full overflow-auto p-4 grid grid-cols-12 gap-4 content-start">
      <div className="col-span-12 flex items-end gap-4 flex-wrap">
        <ObjectPicker label="Primary" value={A} onPick={setA} />
        <ObjectPicker label="Secondary" value={B} onPick={o => { setB(o) }} />
        <Field label="Window (h)"><input className="inp w-20" type="number" value={hours} onChange={e => setHours(+e.target.value)} /></Field>
        <button className="btn btn-primary" disabled={!A || !B || busy} onClick={() => compute()}>Closest approach</button>
        {busy && <Spinner />}
        <div className="flex-1" /><Badge kind="simulated">SGP4 screening · public mean elements</Badge>
      </div>
      <div className="col-span-12"><NeedData status={status} />{err && <Notice kind="warn">{err}</Notice>}</div>
      <div className="col-span-12"><Notice kind="warn">Screening from public mean elements, not an operational conjunction assessment. Probability of collision is <b>not computed</b> because GP/TLE data contains no covariance or hard-body radius.</Notice></div>

      {c && <>
        <div className="col-span-12 flex gap-3 flex-wrap">
          <Stat label="Miss distance" value={fmt(c.miss_distance_km, 3)} unit="km" />
          <Stat label="Relative speed" value={fmt(c.relative_speed_kms, 3)} unit="km/s" />
          <Stat label="TCA (UTC)" value={tcaTxt.slice(11)} hint={tcaTxt.slice(0, 10)} />
          <Stat label="Time to TCA" value={fmt(c.time_to_tca_s / 3600, 2)} unit="h" />
          <Stat label="Category" value={c.category.replace('_', ' ')} hint="screening bin, not a warning" />
          <Stat label="Element age @TCA" value={`${fmt(c.epoch_age_days.primary, 1)} / ${fmt(c.epoch_age_days.secondary, 1)}`} unit="d" hint="primary / secondary" />
          <Stat label="RTN offset" value={c.rel_pos_rtn_km.map(v => fmt(v, 2)).join(', ')} unit="km" hint="radial, transverse, normal" />
        </div>
        <Panel className="col-span-7" title="Separation vs time around TCA" right={<Badge kind="simulated">simulated</Badge>}>
          <Plot height={290} data={[{ x: res.series.offset_s.map(s => s / 60), y: res.series.distance_km, mode: 'lines', line: { color: '#5b8dff' }, name: 'distance' }]}
            layout={{ xaxis: { title: 'minutes from TCA' }, yaxis: { title: 'km', type: res.series.distance_km.some(d => d > 0) ? 'log' : 'linear' },
              shapes: [{ type: 'line', x0: 0, x1: 0, yref: 'paper', y0: 0, y1: 1, line: { color: '#ff8a5b', dash: 'dot' } }] }} />
        </Panel>
        <Panel className="col-span-5" title="Relative trajectory (secondary around primary, TEME)">
          <div className="h-[290px] rounded-lg overflow-hidden"><Canvas camera={{ position: [6, 5, 8], fov: 50 }}><color attach="background" args={['#02050f']} />
            <RelScene rel={rel} idx={idx} /><OrbitControls /></Canvas></div>
          <input type="range" className="w-full mt-2" min={0} max={rel.length - 1} value={step} onChange={e => setStep(+e.target.value)} />
          <div className="text-[10px] text-slate-500">Grid cell = 1 km · blue = primary, orange = secondary path, ring = TCA</div>
        </Panel>
        <Panel className="col-span-12" title="Probability of collision" right={<Badge kind="warn">unavailable from GP data</Badge>}>
          <div className="text-xs text-slate-400 mb-3">{c.pc.reason} If you hold covariance and hard-body radius from another source (e.g. a CDM), enter isotropic 1-σ values to evaluate the 2-D encounter-plane Pc. These numbers are <b>your inputs</b>, not derived from the data.</div>
          <div className="flex gap-3 items-end flex-wrap">
            <Field label="σ primary (km)"><input className="inp w-24" value={pcIn.sa} onChange={e => setPcIn({ ...pcIn, sa: e.target.value })} /></Field>
            <Field label="σ secondary (km)"><input className="inp w-24" value={pcIn.sb} onChange={e => setPcIn({ ...pcIn, sb: e.target.value })} /></Field>
            <Field label="Hard-body radius (km)"><input className="inp w-24" value={pcIn.hbr} onChange={e => setPcIn({ ...pcIn, hbr: e.target.value })} /></Field>
            <button className="btn" onClick={runPc}>Evaluate Pc</button>
            {pc?.pc?.available && <div className="text-sm">Pc = <b>{pc.pc.pc.toExponential(3)}</b> <span className="text-slate-500 text-xs">({pc.pc.assumptions.join('; ')})</span></div>}
          </div>
        </Panel>
      </>}

      <Panel className="col-span-12" title="Screen primary against the whole catalogue"
        right={<><Field label="Threshold km"><input className="inp w-20" value={thr} onChange={e => setThr(e.target.value)} /></Field><button className="btn btn-primary" onClick={runScreen} disabled={!A || sbusy}>Screen</button>{sbusy && <Spinner />}</>}>
        {!screen && <div className="text-xs text-slate-500">Pick a primary and press Screen. Altitude-band filtering, coarse grid and TCA refinement are used; runtime depends on catalogue size.</div>}
        {screen && <>
          <div className="text-xs text-slate-400 mb-2">{screen.n_candidates_after_altitude_filter} of {screen.n_catalogue} objects passed the altitude filter · {screen.conjunctions.length} approaches within {screen.config.threshold_km} km</div>
          <Table max={320} columns={[
            { key: 'secondary_name', label: 'Secondary' }, { key: 'secondary_id', label: 'NORAD' },
            { key: 'secondary_class', label: 'Class', render: r => <span style={{ color: CLASS_COLOR[r.secondary_class] }}>{r.secondary_class}</span> },
            { key: 'miss_distance_km', label: 'Miss km', render: r => fmt(r.miss_distance_km, 3) },
            { key: 'relative_speed_kms', label: 'Rel km/s', render: r => fmt(r.relative_speed_kms, 2) },
            { key: 'tca_utc', label: 'TCA (UTC)', render: r => r.tca_utc.replace('T', ' ').slice(0, 19) },
            { key: 'category', label: 'Bin' }]}
            rows={screen.conjunctions} onRow={r => { api(`/api/catalogue?q=${r.secondary_id}&limit=1`).then(x => { setB(x.objects[0]); compute(A, x.objects[0]) }) }} />
        </>}
      </Panel>
    </div>
  )
}
