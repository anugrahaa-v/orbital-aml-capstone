import React, { useMemo, useRef } from 'react'
import { useFrame, useLoader } from '@react-three/fiber'
import * as THREE from 'three'

export const KM = 1 / 1000                    // scene unit = 1000 km
export const RE = 6.378137
// TEME (x,y,z) km  ->  three (x, z, -y) scene units
export const toScene = (x, y, z) => [x * KM, z * KM, -y * KM]

/** Approximate Sun direction (inertial/TEME ≈ equatorial) from a JS Date. Accuracy ~1°. */
export function sunDirection(date) {
  const jd = date.getTime() / 86400000 + 2440587.5, n = jd - 2451545.0, rad = Math.PI / 180
  const L = (280.460 + 0.9856474 * n) % 360, g = ((357.528 + 0.9856003 * n) % 360) * rad
  const lam = (L + 1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g)) * rad, eps = (23.439 - 0.0000004 * n) * rad
  const v = [Math.cos(lam), Math.cos(eps) * Math.sin(lam), Math.sin(eps) * Math.sin(lam)]
  const s = toScene(v[0], v[1], v[2]); return new THREE.Vector3(...s).normalize()
}
export function gmstRad(date) {            // IAU-82 GMST, radians
  const jd = date.getTime() / 86400000 + 2440587.5, d = jd - 2451545.0, T = d / 36525
  const deg = 280.46061837 + 360.98564736629 * d + 0.000387933 * T * T
  return ((deg % 360 + 360) % 360) * Math.PI / 180
}

const earthVert = `varying vec2 vUv; varying vec3 vN; void main(){ vUv=uv; vN=normalize(mat3(modelMatrix)*normal); gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);} `
const earthFrag = `uniform sampler2D dayMap; uniform sampler2D nightMap; uniform vec3 sunDir; varying vec2 vUv; varying vec3 vN;
void main(){ float d=dot(normalize(vN),normalize(sunDir)); float k=smoothstep(-0.12,0.18,d);
 vec3 day=texture2D(dayMap,vUv).rgb*(0.25+0.9*max(d,0.0)); vec3 night=texture2D(nightMap,vUv).rgb*1.1;
 vec3 c=mix(night,day,k); float rim=pow(1.0-abs(dot(normalize(vN),vec3(0.,0.,1.))),3.0)*0.0; gl_FragColor=vec4(c+rim,1.0);} `
const atmFrag = `varying vec3 vN; varying vec3 vP; uniform vec3 sunDir; void main(){ float i=pow(0.72-dot(normalize(vN),normalize(-vP)),3.0); float lit=0.35+0.65*clamp(dot(normalize(vN),normalize(sunDir))+0.4,0.0,1.0); gl_FragColor=vec4(0.25,0.55,1.0,1.0)*i*lit*1.6; } `
const atmVert = `varying vec3 vN; varying vec3 vP; void main(){ vN=normalize(normalMatrix*normal); vec4 mv=modelViewMatrix*vec4(position,1.0); vP=mv.xyz; gl_Position=projectionMatrix*mv; } `

export function Earth({ gmst = 0, sun, clouds = true }) {
  const [day, night, cl] = useLoader(THREE.TextureLoader, ['/textures/earth_day.jpg', '/textures/earth_night.jpg', '/textures/clouds.png'])
  const mesh = useRef()
  const uniforms = useMemo(() => ({ dayMap: { value: day }, nightMap: { value: night }, sunDir: { value: new THREE.Vector3(1, 0, 0) } }), [day, night])
  const atmU = useMemo(() => ({ sunDir: { value: new THREE.Vector3(1, 0, 0) } }), [])
  useFrame(() => {
    const g = typeof gmst === 'function' ? gmst() : gmst
    if (mesh.current) mesh.current.rotation.y = g
    const s = typeof sun === 'function' ? sun() : sun
    if (s) { uniforms.sunDir.value.copy(s); atmU.sunDir.value.copy(s) }
  })
  return (
    <group>
      <group ref={mesh}>
        <mesh><sphereGeometry args={[RE, 96, 96]} /><shaderMaterial uniforms={uniforms} vertexShader={earthVert} fragmentShader={earthFrag} /></mesh>
        {clouds && <mesh><sphereGeometry args={[RE * 1.008, 64, 64]} /><meshBasicMaterial map={cl} transparent opacity={0.28} depthWrite={false} /></mesh>}
      </group>
      <mesh scale={1.06}><sphereGeometry args={[RE, 64, 64]} />
        <shaderMaterial uniforms={atmU} vertexShader={atmVert} fragmentShader={atmFrag} side={THREE.BackSide} blending={THREE.AdditiveBlending} transparent depthWrite={false} /></mesh>
    </group>
  )
}

export function Shells({ show }) {
  if (!show) return null
  const shells = [[(RE + 2.0), '#4cc2ff', 'LEO limit 2,000 km'], [(RE + 35.786), '#ffd84d', 'GEO 35,786 km']]
  return <group>{shells.map(([r, c], i) => (
    <mesh key={i}><sphereGeometry args={[r, 48, 32]} /><meshBasicMaterial color={c} wireframe transparent opacity={0.07} /></mesh>))}</group>
}

export function Line3({ points, color = '#6aa5ff', opacity = 0.9 }) {
  const obj = useMemo(() => {
    const g = new THREE.BufferGeometry().setFromPoints(points.map(p => new THREE.Vector3(...p)))
    return new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity }))
  }, [points, color, opacity])
  return <primitive object={obj} />
}

export function Stars() {
  const geo = useMemo(() => {
    const a = new Float32Array(3000 * 3)
    for (let i = 0; i < 3000; i++) { const u = Math.random() * 2 - 1, t = Math.random() * 6.283, r = 400 + Math.random() * 100, s = Math.sqrt(1 - u * u)
      a[i * 3] = r * s * Math.cos(t); a[i * 3 + 1] = r * u; a[i * 3 + 2] = r * s * Math.sin(t) }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(a, 3)); return g
  }, [])
  return <points geometry={geo}><pointsMaterial color="#9fb4ff" size={1.2} sizeAttenuation={false} transparent opacity={0.7} /></points>
}
