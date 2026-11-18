import { useRef, useMemo, useEffect } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { CLASS_COLORS } from '../components/Plot.jsx'

// Interpolates between server-propagated keyframes (TEME km). Positions come ONLY from API data.
export default function SatCloud({ pos, tRef, onPick, selected, size = 2.2 }) {
  const ref = useRef(); const sel = useRef()
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry(); const n = pos ? pos.meta.n_objects : 0
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3))
    const col = new Float32Array(n * 3)
    for (let i = 0; i < n; i++) { const c = new THREE.Color(CLASS_COLORS[pos.meta.classes[i]] || '#ffffff'); col.set([c.r, c.g, c.b], i * 3) }
    g.setAttribute('color', new THREE.BufferAttribute(col, 3)); g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e3)
    return g
  }, [pos])
  useFrame(() => {
    if (!pos) return
    const { n_frames: F, n_objects: N, step_s } = pos.meta; const f = Math.min(Math.max(tRef.current / step_s, 0), F - 1.0001)
    const i0 = Math.floor(f), a = f - i0, A = pos.data, o0 = i0 * N * 3, o1 = (i0 + 1) * N * 3, out = geom.attributes.position.array
    for (let i = 0; i < N; i++) {
      const k = i * 3
      const x = A[o0 + k] * (1 - a) + A[o1 + k] * a, y = A[o0 + k + 1] * (1 - a) + A[o1 + k + 1] * a, z = A[o0 + k + 2] * (1 - a) + A[o1 + k + 2] * a
      out[k] = x / 1000; out[k + 1] = z / 1000; out[k + 2] = -y / 1000
    }
    geom.attributes.position.needsUpdate = true
    if (sel.current && selected != null) {
      const i = pos.meta.ids.indexOf(selected); if (i >= 0) { sel.current.visible = true; sel.current.position.set(out[i * 3], out[i * 3 + 1], out[i * 3 + 2]) } else sel.current.visible = false
    }
  })
  useEffect(() => () => geom.dispose(), [geom])
  if (!pos) return null
  return (
    <group>
      <points ref={ref} geometry={geom} onClick={(e) => { e.stopPropagation(); if (e.index != null) onPick && onPick(pos.meta.ids[e.index]) }}>
        <pointsMaterial vertexColors size={size} sizeAttenuation={false} transparent opacity={0.95} depthWrite={false} />
      </points>
      <mesh ref={sel} visible={false}><sphereGeometry args={[0.13, 16, 16]} /><meshBasicMaterial color="#ffffff" /></mesh>
    </group>
  )
}
