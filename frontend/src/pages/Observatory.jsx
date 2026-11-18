import { useRef, useState, useEffect, Suspense, useCallback } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Stars, Line } from '@react-three/drei'
import Earth, { Shells, teme2scene } from '../three/Earth.jsx'
import SatCloud from '../three/SatCloud.jsx'
import { get, decodePositions } from '../api.js'
import { Panel, Notice, Badge, Stat } from '../components/ui.jsx'
import Plot from '../components/Plot.jsx'

function Clock({ simDate, baseRef, tRef, playing, speed, dur, onTick }) {
  const acc = useRef(0)
  useFrame((_, dt) => {
    if (playing.current) { tRef.current = (tRef.current + dt * speed.current) % dur.current }
    simDate.current = new Date(baseRef.current + tRef.current * 1000)
    acc.current += dt; if (acc.current > 0.25) { acc.current = 0; onTick(tRef.current) }
  })
  return null
}

export default function Observatory({ status, setCtx, go }) {
  const [f, setF] = useState({ cls: '', alt: 'all', limit: 3000, q: '' })
  const [pos, setPos] = useState(null); const [err, setErr] = useState(null); const [loading, setLoading] = useState(false)
  const [sel, setSel] = useState(null); const [detail, setDetail] = useState(null); const [path, setPath] = useState(null)
  const [layers, setLayers] = useState({ shells: true, orbit: true, clouds: true })
  const [t, setT] = useState(0); const [speed, setSpeed] = useState(60); const [play, setPlay] = useState(true)
  const tRef = useRef(0), speedRef = useRef(60), playing = useRef(true), dur = useRef(5400)
  const base = useRef(Date.now()); const simDate = useRef(new Date())
  speedRef.current = speed; playing.current = play

  const load = useCallback(() => {
    setLoading(true); setErr(null)
    const alts = { all: [], leo: [0, 2000], meo: [2000, 35000], geo: [35000, 36500] }[f.alt]
    const start = new Date().toISOString()
    get('/positions', { cls: f.cls, limit: f.limit, q: f.q, alt_min: alts[0], alt_max: alts[1], duration_min: 90, step_s: 60, start })
      .then((p) => { base.current = Date.parse(p.start); simDate.current = new Date(base.current); tRef.current = 0; dur.current = (p.n_frames - 1) * p.step_s; setPos({ meta: p, data: decodePositions(p) }) })
      .catch((e) => setErr(e.message)).finally(() => setLoading(false))
  }, [f])
  useEffect(() => { if (status?.catalogue_loaded) load() }, [status, f.cls, f.alt, f.limit]) // eslint-disable-line
  useEffect(() => {
    if (sel == null) return
    get(`/object/${sel}`).then(setDetail).catch(() => setDetail(null))
    get(`/orbit/${sel}/path`, { start: new Date(base.current).toISOString() }).then(setPath).catch(() => setPath(null))
  }, [sel])

  const pathPts = path?.valid ? path.r_teme.map((r) => teme2scene(...r)) : null
  return (
    <div className="relative h-full min-h-[680px]">
      <Canvas camera={{ position: [0, 8, 24], fov: 45, near: 0.1, far: 2000 }} raycaster={{ params: { Points: { threshold: 0.15 } } }} className="!absolute inset-0">
        <color attach="background" args={['#02040a']} /><Stars radius={300} depth={80} count={4000} factor={4} fade />
        <Suspense fallback={null}>
          <Earth simDate={simDate} clouds={layers.clouds} /><Shells show={layers.shells} />
          <SatCloud pos={pos} tRef={tRef} selected={sel} onPick={setSel} />
          {layers.orbit && pathPts && <Line points={pathPts} color="#fde68a" lineWidth={1.6} />}
        </Suspense>
        <Clock simDate={simDate} baseRef={base} tRef={tRef} playing={playing} speed={speedRef} dur={dur} onTick={setT} />
        <OrbitControls enablePan={false} minDistance={8} maxDistance={120} />
      </Canvas>

      <div className="absolute left-4 top-4 w-72 space-y-3 z-10">
        <Panel title="Filters" badge="simulated">
          {!status?.catalogue_loaded && <Notice kind="warn">No catalogue ingested yet. Run the ingest command (see Mission page).</Notice>}
          <div className="grid grid-cols-2 gap-2 text-[13px]">
            <label className="col-span-2"><div className="label">Object class</div>
              <select className="inp w-full" value={f.cls} onChange={(e) => setF({ ...f, cls: e.target.value })}><option value="">All</option><option>PAYLOAD</option><option>DEBRIS</option><option>ROCKET_BODY</option></select></label>
            <label className="col-span-2"><div className="label">Altitude regime</div>
              <select className="inp w-full" value={f.alt} onChange={(e) => setF({ ...f, alt: e.target.value })}><option value="all">All</option><option value="leo">LEO (&lt;2000 km)</option><option value="meo">MEO</option><option value="geo">GEO belt</option></select></label>
            <label className="col-span-2"><div className="label">Max objects drawn: {f.limit}</div>
              <input type="range" min="500" max="8000" step="500" value={f.limit} onChange={(e) => setF({ ...f, limit: +e.target.value })} className="w-full" /></label>
            <input className="inp col-span-2" placeholder="Name filter, Enter to apply" value={f.q} onChange={(e) => setF({ ...f, q: e.target.value })} onKeyDown={(e) => e.key === 'Enter' && load()} />
          </div>
          <div className="flex flex-wrap gap-3 mt-3 text-xs">
            {Object.keys(layers).map((k) => <label key={k} className="flex items-center gap-1"><input type="checkbox" checked={layers[k]} onChange={(e) => setLayers({ ...layers, [k]: e.target.checked })} />{k}</label>)}
          </div>
          <div className="mt-3 text-xs text-slate-400 flex gap-3"><span><i className="inline-block w-2 h-2 rounded-full bg-sky-400" /> payload</span><span><i className="inline-block w-2 h-2 rounded-full bg-rose-400" /> debris</span><span><i className="inline-block w-2 h-2 rounded-full bg-amber-400" /> rocket body</span></div>
          {loading && <div className="text-xs text-slate-400 mt-2">Propagating…</div>}{err && <Notice kind="error">{err}</Notice>}
          {pos && <div className="text-xs text-slate-400 mt-2">{pos.meta.n_objects.toLocaleString()} objects · TEME frame · SGP4</div>}
        </Panel>
      </div>

      {detail && (
        <div className="absolute right-4 top-4 w-80 z-10">
          <Panel title={detail.name} right={<button className="btn-ghost" onClick={() => { setSel(null); setDetail(null); setPath(null) }}>×</button>}>
            <div className="flex gap-1.5 mb-2"><Badge kind="observed" /><Badge kind="derived" /></div>
            <div className="grid grid-cols-2 gap-3">
              <Stat label="NORAD" value={detail.norad_id} /><Stat label="Class" value={detail.class} />
              <Stat label="Perigee" value={detail.derived.perigee_alt_km.toFixed(0)} unit="km" /><Stat label="Apogee" value={detail.derived.apogee_alt_km.toFixed(0)} unit="km" />
              <Stat label="Inclination" value={(+detail.observed_elements.INCLINATION).toFixed(2)} unit="°" /><Stat label="Period" value={detail.derived.period_min.toFixed(1)} unit="min" />
              <Stat label="Eccentricity" value={(+detail.observed_elements.ECCENTRICITY).toFixed(5)} /><Stat label="Element age" value={detail.epoch_age_days.toFixed(1)} unit="d" />
            </div>
            <div className="text-[11px] text-slate-400 mt-2">Class label: {detail.class_source.replace('_', ' ')} (not ground truth). Element sets older than a few days lose accuracy.</div>
            <button className="btn mt-3 w-full" onClick={() => { setCtx({ a: detail }); go('collision') }}>Use as primary in Collision Lab</button>
          </Panel>
        </div>
      )}

      <div className="absolute left-1/2 -translate-x-1/2 bottom-4 w-[min(720px,92vw)] z-10">
        <Panel>
          <div className="flex items-center gap-3 text-[13px]">
            <button className="btn" onClick={() => setPlay(!play)}>{play ? 'Pause' : 'Play'}</button>
            <select className="inp" value={speed} onChange={(e) => setSpeed(+e.target.value)}>{[1, 10, 60, 300, 900].map((s) => <option key={s} value={s}>{s}×</option>)}</select>
            <input type="range" className="flex-1" min="0" max={dur.current} step="30" value={t} onChange={(e) => { tRef.current = +e.target.value; setT(+e.target.value) }} />
            <span className="tabular-nums text-slate-300 w-44 text-right">{new Date(base.current + t * 1000).toISOString().replace('T', ' ').slice(0, 19)} UTC</span>
          </div>
          <div className="text-[11px] text-slate-500 mt-1">Positions: SIMULATED (SGP4 from observed mean elements), linearly interpolated between 60 s keyframes. Click a point to inspect.</div>
        </Panel>
      </div>
    </div>
  )
}
