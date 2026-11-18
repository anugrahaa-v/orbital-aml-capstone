import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { Earth, Shells, Line3, Stars, toScene, sunDirection, RE, KM } from '../scene.jsx'
import { api, useApi, b64ToF32 } from '../api.js'
import { Panel, Badge, Notice, Spinner, Stat, Field, ObjectPicker, Plot, CLASS_COLOR, fmt, NeedData } from '../ui.jsx'

const SPEEDS = [1, 10, 60, 300, 1200]

function Sats({ pack, clock, onPick, selected, hidden }) {
  const geoRef = useRef(), pts = useRef()
  const { data, n, T, step, classes } = pack
  const colors = useMemo(() => {
    const c = new Float32Array(n * 3), col = new THREE.Color()
    classes.forEach((k, i) => { col.set(CLASS_COLOR[k] || '#fff'); c.set([col.r, col.g, col.b], i * 3) }); return c
  }, [classes, n])
  const pos = useMemo(() => new Float32Array(n * 3), [n])
  useFrame(() => {
    const f = Math.min(T - 1.0001, Math.max(0, clock.t / step)), i0 = Math.floor(f), a = f - i0
    for (let i = 0; i < n; i++) {
      const o0 = (i0 * n + i) * 3, o1 = ((i0 + 1) * n + i) * 3
      const x = data[o0] + (data[o1] - data[o0]) * a, y = data[o0 + 1] + (data[o1 + 1] - data[o0 + 1]) * a, z = data[o0 + 2] + (data[o1 + 2] - data[o0 + 2]) * a
      pos[i * 3] = x * KM; pos[i * 3 + 1] = z * KM; pos[i * 3 + 2] = -y * KM
    }
    const g = geoRef.current; if (g) { g.attributes.position.needsUpdate = true; g.computeBoundingSphere() }
    clock.pos = pos
  })
  return (
    <points ref={pts} onClick={e => { e.stopPropagation(); if (e.index !== undefined) onPick(e.index) }}>
      <bufferGeometry ref={geoRef}>
        <bufferAttribute attach="attributes-position" args={[pos, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial vertexColors size={2.6} sizeAttenuation={false} transparent opacity={0.95} depthWrite={false} />
    </points>
  )
}

function SelectedMarker({ clock, index }) {
  const m = useRef()
  useFrame(() => { if (m.current && clock.pos && index != null) m.current.position.set(clock.pos[index * 3], clock.pos[index * 3 + 1], clock.pos[index * 3 + 2]) })
  return index == null ? null : <mesh ref={m}><sphereGeometry args={[0.09, 16, 16]} /><meshBasicMaterial color="#ffffff" /></mesh>
}

function Clock({ clock, playing, speed, total, setT, startDate, view }) {
  useFrame((_, dt) => {
    if (playing) { clock.t += dt * speed; if (clock.t > total) clock.t = 0 }
    clock.date = new Date(startDate.getTime() + clock.t * 1000)
    view.current = clock.t
  })
  return null
}

export default function Observatory({ status, select, setSelect, go }) {
  const [filters, setFilters] = useState({ cls: '', alt_min: '', alt_max: '', limit: 3000 })
  const [pack, setPack] = useState(null)
  const [err, setErr] = useState(null)
  const [loading, setLoading] = useState(false)
  const [playing, setPlaying] = useState(true)
  const [speed, setSpeed] = useState(60)
  const [layers, setLayers] = useState({ shells: true, orbit: true, plane: false })
  const [selIdx, setSelIdx] = useState(null)
  const [tUi, setTUi] = useState(0)
  const clock = useRef({ t: 0, pos: null, date: new Date() }).current
  const viewT = useRef(0)
  const startRef = useRef(new Date())

  const load = async () => {
    setLoading(true); setErr(null)
    try {
      const start = new Date(); startRef.current = start
      const qs = new URLSearchParams({ start: start.toISOString(), duration_min: 120, step_s: 60, limit: filters.limit })
      if (filters.cls) qs.set('cls', filters.cls)
      if (filters.alt_min !== '') qs.set('alt_min', filters.alt_min)
      if (filters.alt_max !== '') qs.set('alt_max', filters.alt_max)
      const p = await api(`/api/positions?${qs}`)
      setPack({ data: b64ToF32(p.data_b64), n: p.n_objects, T: p.n_frames, step: p.step_s, ids: p.ids, names: p.names, classes: p.classes, gmst: p.gmst_rad, start })
      clock.t = 0; setSelIdx(null)
    } catch (e) { setErr(e.message); setPack(null) }
    setLoading(false)
  }
  useEffect(() => { if (status?.catalogue_loaded) load() /* eslint-disable-next-line */ }, [status?.catalogue_loaded])
  useEffect(() => { const i = setInterval(() => setTUi(viewT.current), 250); return () => clearInterval(i) }, [])

  const pick = (i) => { setSelIdx(i); const id = pack.ids[i]; api(`/api/catalogue?q=${id}&limit=1`).then(r => r.objects[0] && setSelect(r.objects[0])) }
  const selId = selIdx != null && pack ? pack.ids[selIdx] : select?.norad_id
  const detail = useApi(selId ? `/api/object/${selId}` : null)
  const path = useApi(selId && layers.orbit ? `/api/orbit/${selId}/path?start=${(pack?.start || new Date()).toISOString()}` : null, [pack])
  const gtrack = useApi(selId ? `/api/ground-track/${selId}?start=${(pack?.start || new Date()).toISOString()}&hours=3` : null, [pack])

  const total = pack ? (pack.T - 1) * pack.step : 1
  const gm = () => pack ? (() => { const f = Math.min(pack.T - 1.0001, clock.t / pack.step), i = Math.floor(f); return pack.gmst[i] + (pack.gmst[i + 1] - pack.gmst[i]) * (f - i) })() : 0
  const sun = () => sunDirection(clock.date)
  const orbitPts = useMemo(() => path.data?.valid ? path.data.r_teme.map(p => toScene(...p)) : null, [path.data])
  const normal = useMemo(() => {
    if (!orbitPts || orbitPts.length < 3) return null
    const a = new THREE.Vector3(...orbitPts[0]), b = new THREE.Vector3(...orbitPts[Math.floor(orbitPts.length / 4)])
    return a.clone().cross(b).normalize()
  }, [orbitPts])
  const d = detail.data
  const timeLabel = new Date(startRef.current.getTime() + tUi * 1000).toISOString().replace('T', ' ').slice(0, 19)

  return (
    <div className="h-full relative">
      <Canvas camera={{ position: [0, 9, 26], fov: 45, near: 0.05, far: 2000 }} raycaster={{ params: { Points: { threshold: 0.25 } } }} gl={{ antialias: true }}>
        <color attach="background" args={['#01030a']} />
        <Stars />
        {pack && <Clock clock={clock} playing={playing} speed={speed} total={total} startDate={pack.start} view={viewT} />}
        <Earth gmst={gm} sun={sun} />
        <Shells show={layers.shells} />
        {pack && <Sats pack={pack} clock={clock} onPick={pick} />}
        {pack && <SelectedMarker clock={clock} index={selIdx} />}
        {orbitPts && layers.orbit && <Line3 points={orbitPts} color="#ffffff" opacity={0.75} />}
        {normal && layers.plane && <mesh quaternion={new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal)}>
          <circleGeometry args={[Math.max(...orbitPts.map(p => Math.hypot(...p))), 64]} /><meshBasicMaterial color="#5b8dff" transparent opacity={0.08} side={THREE.DoubleSide} depthWrite={false} /></mesh>}
        <OrbitControls enablePan={false} minDistance={8} maxDistance={120} rotateSpeed={0.6} />
      </Canvas>

      <div className="absolute top-3 left-3 w-72 space-y-3 max-h-[calc(100%-100px)] overflow-auto pr-1">
        <Panel title="Orbital Observatory" right={<Badge kind="simulated">SGP4 · TEME</Badge>}>
          <NeedData status={status} />
          {err && <Notice kind="warn">{err}</Notice>}
          <div className="grid grid-cols-2 gap-2 mt-2">
            <Field label="Class"><select className="inp" value={filters.cls} onChange={e => setFilters({ ...filters, cls: e.target.value })}>
              <option value="">All</option><option>PAYLOAD</option><option>DEBRIS</option><option>ROCKET_BODY</option></select></Field>
            <Field label="Max objects"><select className="inp" value={filters.limit} onChange={e => setFilters({ ...filters, limit: +e.target.value })}>
              {[500, 1500, 3000, 6000].map(v => <option key={v}>{v}</option>)}</select></Field>
            <Field label="Alt min km"><input className="inp" value={filters.alt_min} onChange={e => setFilters({ ...filters, alt_min: e.target.value })} /></Field>
            <Field label="Alt max km"><input className="inp" value={filters.alt_max} onChange={e => setFilters({ ...filters, alt_max: e.target.value })} /></Field>
          </div>
          <div className="flex gap-2 mt-3 items-center"><button className="btn btn-primary" onClick={load} disabled={loading}>Load objects</button>{loading && <Spinner />}</div>
          <div className="flex flex-wrap gap-3 mt-3 text-xs text-slate-300">
            {[['shells', 'Altitude shells'], ['orbit', 'Orbit path'], ['plane', 'Orbital plane']].map(([k, l]) => (
              <label key={k} className="flex items-center gap-1"><input type="checkbox" checked={layers[k]} onChange={e => setLayers({ ...layers, [k]: e.target.checked })} />{l}</label>))}
          </div>
          <div className="mt-3 flex gap-3 text-[11px]">{Object.entries(CLASS_COLOR).map(([k, c]) => <span key={k}><span style={{ color: c }}>●</span> {k}</span>)}</div>
          <div className="mt-3"><ObjectPicker label="Find object" value={null} onPick={o => { setSelect(o); setSelIdx(pack?.ids.indexOf(o.norad_id) >= 0 ? pack.ids.indexOf(o.norad_id) : null) }} /></div>
        </Panel>

        {selId && <Panel title="Selected object" right={d && <Badge kind="observed">observed</Badge>}>
          {detail.loading && <Spinner />}{detail.error && <Notice>{detail.error}</Notice>}
          {d && <div className="text-xs space-y-1">
            <div className="text-sm font-semibold">{d.name}</div>
            <div className="text-slate-400">NORAD {d.norad_id} · {d.object_id || 'n/a'} · <span style={{ color: CLASS_COLOR[d.class] }}>{d.class}</span> <span className="text-slate-500">({d.class_source})</span></div>
            <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 mt-2">
              <span className="text-slate-500">Inclination</span><span>{fmt(d.observed_elements.INCLINATION, 3)}°</span>
              <span className="text-slate-500">Eccentricity</span><span>{fmt(d.observed_elements.ECCENTRICITY, 5)}</span>
              <span className="text-slate-500">Mean motion</span><span>{fmt(d.observed_elements.MEAN_MOTION, 5)} rev/d</span>
              <span className="text-slate-500">B*</span><span>{fmt(d.observed_elements.BSTAR, 6)}</span>
              <span className="text-slate-500">Epoch age</span><span>{fmt(d.epoch_age_days, 2)} d</span>
            </div>
            <div className="mt-2 flex items-center gap-2"><Badge kind="derived">derived</Badge></div>
            <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
              <span className="text-slate-500">Period</span><span>{fmt(d.derived.period_min, 2)} min</span>
              <span className="text-slate-500">Perigee alt</span><span>{fmt(d.derived.perigee_alt_km, 1)} km</span>
              <span className="text-slate-500">Apogee alt</span><span>{fmt(d.derived.apogee_alt_km, 1)} km</span>
            </div>
            <div className="flex gap-2 mt-3"><button className="btn" onClick={() => go('collision')}>Collision Lab →</button><button className="btn" onClick={() => go('forecast')}>Forecast →</button></div>
          </div>}
          {gtrack.data && <div className="mt-3"><Plot height={150} data={[{ type: 'scattergeo', mode: 'lines', lat: gtrack.data.lat, lon: gtrack.data.lon, line: { color: '#ffd84d', width: 1.5 } }]}
            layout={{ margin: { l: 0, r: 0, t: 0, b: 0 }, geo: { projection: { type: 'equirectangular' }, bgcolor: 'rgba(0,0,0,0)', showland: true, landcolor: '#18233f', showcoastlines: true, coastlinecolor: '#3b4c7a', lonaxis: { range: [-180, 180] }, lataxis: { range: [-90, 90] } } }} /></div>}
          {gtrack.data && <div className="text-[10px] text-slate-500">3-hour ground track (simulated, UT1≈UTC)</div>}
        </Panel>}
      </div>

      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 w-[min(780px,92%)]">
        <Panel>
          <div className="flex items-center gap-3 text-xs">
            <button className="btn" onClick={() => setPlaying(!playing)}>{playing ? '❚❚ Pause' : '▶ Play'}</button>
            <input type="range" className="flex-1" min={0} max={total} step={1} value={tUi} onChange={e => { clock.t = +e.target.value; setTUi(+e.target.value) }} />
            <select className="inp" value={speed} onChange={e => setSpeed(+e.target.value)}>{SPEEDS.map(s => <option key={s} value={s}>{s}×</option>)}</select>
            <span className="font-mono text-slate-300 w-40 text-right">{timeLabel} UTC</span>
          </div>
          <div className="text-[10px] text-slate-500 mt-1 flex justify-between"><span>{pack ? `${pack.n} objects · positions: SGP4 from observed mean elements (simulated)` : 'no objects loaded'}</span><span>Sun/GMST computed in browser (approx.)</span></div>
        </Panel>
      </div>
    </div>
  )
}
