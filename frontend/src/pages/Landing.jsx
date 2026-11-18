import { useRef, useState, useEffect, Suspense } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Stars } from '@react-three/drei'
import Earth, { Shells } from '../three/Earth.jsx'
import SatCloud from '../three/SatCloud.jsx'
import { get, decodePositions } from '../api.js'
import { Notice } from '../components/ui.jsx'

function Clock({ simDate, tRef, speed }) {
  useFrame((_, dt) => { tRef.current += dt * speed; simDate.current = new Date(simDate.current.getTime() + dt * speed * 1000) })
  return null
}

export default function Landing({ status, go }) {
  const simDate = useRef(new Date()); const tRef = useRef(0); const [pos, setPos] = useState(null)
  useEffect(() => {
    if (!status?.catalogue_loaded) return
    get('/positions', { duration_min: 90, step_s: 60, limit: 1800, start: new Date().toISOString() }).then((p) => { tRef.current = 0; setPos({ meta: p, data: decodePositions(p) }) }).catch(() => {})
  }, [status])
  const loaded = status?.catalogue_loaded
  return (
    <div className="relative h-full min-h-[640px]">
      <Canvas camera={{ position: [0, 6, 22], fov: 45 }} className="!absolute inset-0">
        <color attach="background" args={['#02040a']} />
        <Stars radius={200} depth={60} count={5000} factor={4} fade speed={0.4} />
        <Suspense fallback={null}>
          <Earth simDate={simDate} />
          <Shells show />
          {pos && <SatCloud pos={pos} tRef={tRef} size={1.8} />}
        </Suspense>
        <Clock simDate={simDate} tRef={tRef} speed={loaded ? 90 : 0} />
        <OrbitControls autoRotate autoRotateSpeed={0.35} enablePan={false} minDistance={9} maxDistance={60} />
      </Canvas>
      <div className="relative z-10 p-8 md:p-14 max-w-3xl pointer-events-none">
        <div className="label text-sky-300">Research platform · MSAIML</div>
        <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight mt-2 text-white">ORBITAL<span className="text-sky-400">X</span> AI</h1>
        <p className="mt-4 text-lg text-slate-300">Orbital debris monitoring, geometric conjunction screening and reinforcement-learning manoeuvre research on public GP element sets.</p>
        <div className="mt-6 flex gap-3 pointer-events-auto">
          <button className="btn" onClick={() => go('observatory')}>Open observatory</button>
          <button className="btn-ghost" onClick={() => go('collision')}>Collision lab</button>
        </div>
        <div className="mt-6 glass p-4 pointer-events-auto text-sm space-y-2">
          <div><b>Data source:</b> CelesTrak GP (one dataset). Orbits are SGP4-propagated from observed mean elements.</div>
          {loaded ? <div className="text-slate-300">{status.n_objects.toLocaleString()} objects loaded · epochs {status.epoch_range?.[0]?.slice(0, 10)} to {status.epoch_range?.[1]?.slice(0, 10)}. Points above are the real catalogue (sampled, simulated positions).</div>
            : <Notice kind="warn">No catalogue ingested, so no objects are shown (the wireframe shells are reference geometry only). Run <code>python -m backend.ingest_cli --groups active cosmos-2251-debris</code>, then restart the API.{status?.error && <> API: {status.error}</>}</Notice>}
          <div className="text-xs text-slate-400">Collision probability is never estimated: GP data carries no covariance. Results are screening values, not operational warnings.</div>
        </div>
      </div>
    </div>
  )
}
