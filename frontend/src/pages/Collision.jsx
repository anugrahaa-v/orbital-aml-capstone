import { useRef, useState, useEffect, Suspense } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Stars, Line } from '@react-three/drei'
import Earth, { teme2scene } from '../three/Earth.jsx'
import { get, post } from '../api.js'
import { Panel, Notice, Badge, Stat, ObjectPicker } from '../components/ui.jsx'
import Plot from '../components/Plot.jsx'

function Encounter({ res, tRef }) {
  const a = useRef(), b = useRef(), link = useRef(), tracks = res.series
  const pa = tracks.r_a_teme.map((r) => teme2scene(...r)), pb = tracks.r_b_teme.map((r) => teme2scene(...r))
  const mid = pa[Math.floor(pa.length / 2)]
  useFrame(({ camera }, dt) => {
    const i = Math.min(pa.length - 1, Math.max(0, Math.round(tRef.current)))
    a.current.position.set(...pa[i]); b.current.position.set(...pb[i])
    link.current.geometry.setFromPoints([a.current.position, b.current.position])
  })
  return (
    <group>
      <Line points={pa} color="#38bdf8" lineWidth={2} /><Line points={pb} color="#fb7185" lineWidth={2} />
      <mesh ref={a}><sphereGeometry args={[0.012, 12, 12]} /><meshBasicMaterial color="#7dd3fc" /></mesh>
      <mesh ref={b}><sphereGeometry args={[0.012, 12, 12]} /><meshBasicMaterial color="#fda4af" /></mesh>
      <line ref={link}><bufferGeometry /><lineBasicMaterial color="#fde047" /></line>
      <OrbitControls target={mid} enablePan minDistance={0.02} maxDistance={40} />
    </group>
  )
}

export default function Collision({ ctx, status }) {
  const [A, setA] = useState(ctx?.a || null), [B, setB] = useState(null)
  const [hours, setHours] = useState(24); const [res, setRes] = useState(null); const [err, setErr] = useState(null); const [busy, setBusy] = useState(false)
  const [scr, setScr] = useState(null); const [th, setTh] = useState(10)
  const [sig, setSig] = useState({ a: 0.5, b: 0.5, hbr: 0.01 }); const [pc, setPc] = useState(null)
  const [i, setI] = useState(0); const tRef = useRef(0); const simDate = useRef(new Date())
  tRef.current = i
  const start = useRef(new Date().toISOString())
  async function compute(a = A, b = B) {
    if (!a || !b) return
    setBusy(true); setErr(null); setPc(null)
    try { const r = await get('/collision/closest', { a: a.norad_id, b: b.norad_id, hours, start: start.current }); setRes(r); simDate.current = new Date(r.conjunction.tca_utc); setI(Math.floor(r.series.offset_s.length / 2)) }
    catch (e) { setErr(e.message); setRes(null) } finally { setBusy(false) }
  }
  async function screen() {
    if (!A) return; setBusy(true); setErr(null)
    try { setScr(await post('/collision/screen', { primary: A.norad_id, start: start.current, threshold_km: th, horizon_h: Math.min(hours, 48) })) } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }
  async function computePc() {
    const d = (s) => [[s * s, 0, 0], [0, s * s, 0], [0, 0, s * s]]
    try { setPc(await post('/collision/pc', { a: A.norad_id, b: B.norad_id, start: start.current, hours, cov_a_km2: d(sig.a), cov_b_km2: d(sig.b), hbr_km: sig.hbr })) } catch (e) { setErr(e.message) }
  }
  const c = res?.conjunction, s = res?.series
  return (
    <div className="p-4 grid gap-4 lg:grid-cols-[340px_1fr]">
      <div className="space-y-4">
        <Panel title="Select pair" badge="simulated">
          <div className="space-y-3">
            <ObjectPicker label="Primary" value={A} onPick={setA} /><ObjectPicker label="Secondary" value={B} onPick={setB} />
            <label className="block text-[13px]"><div className="label">Look-ahead: {hours} h</div><input type="range" min="1" max="72" value={hours} onChange={(e) => setHours(+e.target.value)} className="w-full" /></label>
            <div className="flex gap-2"><button className="btn" disabled={!A || !B || busy} onClick={() => compute()}>Closest approach</button>
              <button className="btn-ghost" disabled={!A || busy} onClick={screen}>Screen primary vs catalogue</button></div>
            <label className="block text-[13px]"><div className="label">Screening threshold: {th} km</div><input type="range" min="1" max="50" value={th} onChange={(e) => setTh(+e.target.value)} className="w-full" /></label>
            {err && <Notice kind="error">{err}</Notice>}{busy && <div className="text-xs text-slate-400">Computing…</div>}
          </div>
        </Panel>
        <Panel title="Probability of collision" badge="unavailable">
          <Notice kind="warn">{c ? c.pc.reason : 'GP/TLE data has no covariance or hard-body radius, so Pc is not estimated.'}</Notice>
          <details className="mt-3 text-[13px]"><summary className="cursor-pointer text-slate-300">Optional: supply your own assumptions</summary>
            <div className="mt-2 grid grid-cols-3 gap-2">
              {[['σ primary km', 'a'], ['σ secondary km', 'b'], ['HBR km', 'hbr']].map(([l, k]) => <label key={k}><div className="label">{l}</div><input className="inp w-full" type="number" step="0.01" value={sig[k]} onChange={(e) => setSig({ ...sig, [k]: +e.target.value })} /></label>)}
            </div>
            <button className="btn-ghost mt-2" disabled={!res} onClick={computePc}>Compute 2-D Pc</button>
            {pc?.pc?.available && <div className="mt-2"><Badge kind="derived">USER-ASSUMED INPUTS</Badge> Pc = <b>{pc.pc.pc.toExponential(3)}</b><div className="text-[11px] text-slate-400">Isotropic Gaussian σ, linear relative motion. Illustrative only; not derived from data.</div></div>}
          </details>
        </Panel>
      </div>

      <div className="space-y-4 min-w-0">
        {!res && <Panel><div className="text-slate-400 text-sm">Pick two objects (try a debris object and an active payload in a similar altitude shell) and compute their closest approach. {status && !status.catalogue_loaded && 'No catalogue is loaded yet.'}</div></Panel>}
        {c && <>
          <Panel title={`${res.primary.name} ↔ ${res.secondary.name}`} badge="simulated" right={<span className={`text-xs px-2 py-1 rounded ${c.category === 'CLOSE_APPROACH' ? 'bg-rose-500/20 text-rose-200' : c.category === 'ELEVATED' ? 'bg-amber-500/20 text-amber-200' : 'bg-sky-500/20 text-sky-200'}`}>{c.category.replace('_', ' ')}</span>}>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Stat label="Miss distance" value={c.miss_distance_km.toFixed(3)} unit="km" /><Stat label="Relative speed" value={c.relative_speed_kms.toFixed(2)} unit="km/s" />
              <Stat label="TCA (UTC)" value={c.tca_utc.slice(0, 19).replace('T', ' ')} /><Stat label="Time to TCA" value={(c.time_to_tca_s / 3600).toFixed(2)} unit="h" />
              <Stat label="Radial / In-track / Cross" value={c.rel_pos_rtn_km.map((x) => x.toFixed(2)).join(' / ')} unit="km" />
              <Stat label="Element age primary" value={c.epoch_age_days.primary.toFixed(1)} unit="d" /><Stat label="Element age secondary" value={c.epoch_age_days.secondary.toFixed(1)} unit="d" />
              <Stat label="Uncertainty" value="not available" />
            </div>
            <div className="mt-3"><Notice kind="warn">{res.warning} Mean-element positions typically carry km-level error that grows with element age; a small miss distance here is a screening flag, not a prediction of impact.</Notice></div>
          </Panel>
          <div className="grid xl:grid-cols-2 gap-4">
            <Panel title="Separation vs time (±10 min of TCA)" badge="simulated">
              <Plot height={300} data={[{ x: s.offset_s, y: s.distance_km, mode: 'lines', line: { color: '#fde047' }, name: 'separation' }]} layout={{ xaxis: { title: 'seconds from TCA' }, yaxis: { title: 'km', type: 'log' } }} />
            </Panel>
            <Panel title="Encounter view (true TEME positions)" badge="simulated" className="!p-0 overflow-hidden">
              <div className="h-[340px] relative">
                <Canvas camera={{ position: [0, 7.4, 6.3], fov: 50, near: 0.001, far: 500 }}>
                  <color attach="background" args={['#02040a']} /><Stars radius={100} count={2000} />
                  <Suspense fallback={null}><Earth simDate={simDate} clouds={false} /><Encounter res={res} tRef={tRef} /></Suspense>
                </Canvas>
                <div className="absolute bottom-2 left-3 right-3"><input type="range" className="w-full" min="0" max={s.offset_s.length - 1} value={i} onChange={(e) => setI(+e.target.value)} /></div>
              </div>
            </Panel>
          </div>
        </>}
        {scr && <Panel title={`Screening: ${scr.primary.name}`} badge="simulated">
          <div className="text-xs text-slate-400 mb-2">{scr.n_candidates_after_altitude_filter.toLocaleString()} of {scr.n_catalogue.toLocaleString()} objects passed the perigee/apogee shell filter · threshold {scr.config.threshold_km} km · {scr.config.horizon_h} h</div>
          {scr.conjunctions.length === 0 ? <Notice>No approaches closer than the threshold in this window.</Notice> :
            <div className="scroll max-h-72"><table className="t"><thead><tr><th>Secondary</th><th>Class</th><th>TCA (UTC)</th><th>Miss km</th><th>Rel. km/s</th><th>Bin</th><th /></tr></thead><tbody>
              {scr.conjunctions.map((x, k) => <tr key={k}><td>{x.secondary_name} <span className="text-slate-500">#{x.secondary_id}</span></td><td>{x.secondary_class}</td><td>{x.tca_utc.slice(5, 19).replace('T', ' ')}</td><td>{x.miss_distance_km.toFixed(2)}</td><td>{x.relative_speed_kms.toFixed(1)}</td><td>{x.category.replace('_', ' ')}</td>
                <td><button className="btn-ghost" onClick={() => { const o = { norad_id: x.secondary_id, name: x.secondary_name, class: x.secondary_class }; setB(o); compute(A, o) }}>View</button></td></tr>)}
            </tbody></table></div>}
        </Panel>}
      </div>
    </div>
  )
}
