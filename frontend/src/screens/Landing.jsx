import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { Earth, Stars, sunDirection, gmstRad, KM, RE } from '../scene.jsx'
import { api, b64ToF32 } from '../api.js'

function RealSats({ pack }) {
  const g = useRef(), clock = useRef(0)
  const pos = useMemo(() => new Float32Array(pack.n * 3), [pack])
  const col = useMemo(() => { const c = new Float32Array(pack.n * 3), k = new THREE.Color(); pack.classes.forEach((x, i) => { k.set(x === 'DEBRIS' ? '#ff7a59' : x === 'ROCKET_BODY' ? '#ffd84d' : '#4cc2ff'); c.set([k.r, k.g, k.b], i * 3) }); return c }, [pack])
  useFrame((_, dt) => {
    clock.current += dt * 40
    const f = Math.min(pack.T - 1.0001, (clock.current / pack.step) % (pack.T - 1)), i0 = Math.floor(f), a = f - i0
    for (let i = 0; i < pack.n; i++) {
      const o0 = (i0 * pack.n + i) * 3, o1 = ((i0 + 1) * pack.n + i) * 3, d = pack.data
      pos[i * 3] = (d[o0] + (d[o1] - d[o0]) * a) * KM; pos[i * 3 + 1] = (d[o0 + 2] + (d[o1 + 2] - d[o0 + 2]) * a) * KM; pos[i * 3 + 2] = -(d[o0 + 1] + (d[o1 + 1] - d[o0 + 1]) * a) * KM
    }
    if (g.current) { g.current.attributes.position.needsUpdate = true; g.current.computeBoundingSphere() }
  })
  return <points><bufferGeometry ref={g}><bufferAttribute attach="attributes-position" args={[pos, 3]} /><bufferAttribute attach="attributes-color" args={[col, 3]} /></bufferGeometry>
    <pointsMaterial vertexColors size={2.2} sizeAttenuation={false} transparent opacity={0.9} depthWrite={false} /></points>
}

/** Reference orbital shells: geometry only (NOT satellites, no object data). */
function ReferenceShells() {
  return <group>{[[RE + 0.55, '#4cc2ff'], [RE + 20.2, '#7a8cff'], [RE + 35.786, '#ffd84d']].map(([r, c], i) =>
    <mesh key={i} rotation={[Math.PI / 2 + i * 0.3, 0, 0]}><ringGeometry args={[r - 0.01, r + 0.01, 128]} /><meshBasicMaterial color={c} transparent opacity={0.35} side={THREE.DoubleSide} /></mesh>)}</group>
}

function Spin({ children }) {
  const r = useRef(); useFrame((s, dt) => { if (r.current) r.current.rotation.y += dt * 0.02 })
  return <group ref={r}>{children}</group>
}
function Drift() {
  useFrame(({ camera, clock }) => { const t = clock.elapsedTime * 0.08; camera.position.set(Math.sin(t) * 22, 6 + Math.sin(t * 0.7) * 3, Math.cos(t) * 22); camera.lookAt(0, 0, 0) })
  return null
}

export default function Landing({ status, go }) {
  const [pack, setPack] = useState(null)
  useEffect(() => {
    if (!status?.catalogue_loaded) return
    api(`/api/positions?start=${new Date().toISOString()}&duration_min=100&step_s=100&limit=2000`).then(p => setPack({ data: b64ToF32(p.data_b64), n: p.n_objects, T: p.n_frames, step: p.step_s, classes: p.classes })).catch(() => {})
  }, [status?.catalogue_loaded])
  const now = useMemo(() => new Date(), [])
  return (
    <div className="h-full relative overflow-hidden">
      <Canvas camera={{ position: [0, 6, 22], fov: 45, near: 0.05, far: 2000 }}>
        <color attach="background" args={['#000207']} />
        <Stars /><Drift />
        <Earth gmst={() => gmstRad(new Date())} sun={() => sunDirection(new Date())} />
        {pack ? <RealSats pack={pack} /> : <ReferenceShells />}
      </Canvas>
      <div className="absolute inset-0 pointer-events-none bg-gradient-to-t from-black/80 via-transparent to-black/50" />
      <div className="absolute left-10 bottom-16 max-w-2xl">
        <div className="text-[11px] tracking-[0.5em] text-sky-300/80 mb-3">ORBITAL DEBRIS INTELLIGENCE</div>
        <h1 className="text-6xl font-bold tracking-tight bg-gradient-to-r from-white via-sky-200 to-violet-300 bg-clip-text text-transparent">ORBITALX&nbsp;AI</h1>
        <p className="mt-4 text-slate-300 text-lg">Monitoring the orbital environment, screening close approaches and testing learned and planned responses, from real public element sets.</p>
        <div className="mt-6 flex gap-3 pointer-events-auto">
          <button className="btn btn-primary text-base px-6 py-2" onClick={() => go('observatory')}>Enter Observatory</button>
          <button className="btn text-base px-6 py-2" onClick={() => go('collision')}>Collision Lab</button>
        </div>
        <div className="mt-6 text-xs text-slate-400">
          {status?.catalogue_loaded
            ? <>Showing <b className="text-sky-300">{pack?.n ?? '…'}</b> real catalogued objects (SGP4-propagated from CelesTrak GP data) of {status.n_objects} loaded.</>
            : <>No catalogue ingested yet. The rings show reference LEO / MEO / GEO altitudes only (geometry, not satellites).</>}
        </div>
      </div>
    </div>
  )
}
