import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { useTexture } from '@react-three/drei'
import * as THREE from 'three'

export const KM = 1 / 1000                 // scene unit = 1000 km
export const RE = 6.378137
export const teme2scene = (x, y, z) => [x * KM, z * KM, -y * KM]   // TEME z-up -> three y-up (right-handed)

export function gmstRad(date) {            // IAU-82 GMST from a JS Date (UT1 ~ UTC)
  const jd = date.getTime() / 86400000 + 2440587.5
  const T = (jd - 2451545.0) / 36525
  let g = 67310.54841 + (876600 * 3600 + 8640184.812866) * T + 0.093104 * T * T - 6.2e-6 * T ** 3
  return ((g % 86400) / 240) * Math.PI / 180 % (2 * Math.PI)
}
export function sunDirScene(date) {        // low-precision solar position (~0.01 deg), equatorial frame
  const n = date.getTime() / 86400000 + 2440587.5 - 2451545.0
  const L = (280.46 + 0.9856474 * n) % 360, g = ((357.528 + 0.9856003 * n) % 360) * Math.PI / 180
  const lam = (L + 1.915 * Math.sin(g) + 0.02 * Math.sin(2 * g)) * Math.PI / 180, eps = (23.439 - 4e-7 * n) * Math.PI / 180
  const [x, y, z] = [Math.cos(lam), Math.cos(eps) * Math.sin(lam), Math.sin(eps) * Math.sin(lam)]
  return new THREE.Vector3(x, z, -y).normalize()
}

const vert = `varying vec2 vUv; varying vec3 vN; void main(){ vUv=uv; vN=normalize(mat3(modelMatrix)*normal); gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);} `
const frag = `uniform sampler2D dayMap; uniform sampler2D nightMap; uniform vec3 sunDir; varying vec2 vUv; varying vec3 vN;
void main(){ float d=dot(normalize(vN),normalize(sunDir)); float k=smoothstep(-0.12,0.18,d);
 vec3 day=texture2D(dayMap,vUv).rgb*(0.25+0.85*max(d,0.0)); vec3 night=texture2D(nightMap,vUv).rgb*1.25;
 vec3 col=mix(night,day,k); gl_FragColor=vec4(col,1.0);} `
const atmV = `varying vec3 vN; varying vec3 vP; void main(){ vN=normalize(normalMatrix*normal); vP=(modelViewMatrix*vec4(position,1.0)).xyz; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);} `
const atmF = `varying vec3 vN; varying vec3 vP; void main(){ float f=pow(max(0.72-dot(vN,normalize(-vP)),0.0),3.0); gl_FragColor=vec4(0.25,0.55,1.0,1.0)*f*1.6;} `

export default function Earth({ simDate, gmst, clouds = true, atmosphere = true }) {
  const g = useRef(), cl = useRef()
  const [day, night, cloud] = useTexture(['/textures/day.jpg', '/textures/night.jpg', '/textures/clouds.png'])
  ;[day, night].forEach((t) => { t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8 })
  const uniforms = useMemo(() => ({ dayMap: { value: day }, nightMap: { value: night }, sunDir: { value: new THREE.Vector3(1, 0, 0) } }), [day, night])
  useFrame(() => {
    const d = simDate.current instanceof Date ? simDate.current : new Date()
    uniforms.sunDir.value.copy(sunDirScene(d))
    g.current.rotation.y = gmst ? gmst.current : gmstRad(d)
    if (cl.current) cl.current.rotation.y += 0.00002
  })
  return (
    <group>
      <mesh ref={g}><sphereGeometry args={[RE, 96, 96]} /><shaderMaterial uniforms={uniforms} vertexShader={vert} fragmentShader={frag} /></mesh>
      {clouds && <mesh ref={cl}><sphereGeometry args={[RE * 1.006, 64, 64]} /><meshBasicMaterial map={cloud} transparent opacity={0.22} depthWrite={false} /></mesh>}
      {atmosphere && <mesh scale={1.045}><sphereGeometry args={[RE, 64, 64]} /><shaderMaterial vertexShader={atmV} fragmentShader={atmF} side={THREE.BackSide} blending={THREE.AdditiveBlending} transparent depthWrite={false} /></mesh>}
    </group>
  )
}

export function Shells({ show }) {         // reference altitude shells (geometry only, no objects)
  if (!show) return null
  const shells = [[2000, '#38bdf8'], [35786, '#a78bfa']]
  return shells.map(([alt, c]) => <mesh key={alt}><sphereGeometry args={[(6378.137 + alt) * KM, 48, 24]} /><meshBasicMaterial color={c} wireframe transparent opacity={0.07} /></mesh>)
}
