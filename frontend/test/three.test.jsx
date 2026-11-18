import { describe, it, expect, vi } from 'vitest'
vi.unmock('@react-three/fiber')            // the setup file stubs it for page tests; here we want the real reconciler
vi.unmock('@react-three/drei')
import ReactThreeTestRenderer from '@react-three/test-renderer'
import { gmstRad, sunDirScene, teme2scene } from '../src/three/Earth.jsx'
import SatCloud from '../src/three/SatCloud.jsx'

const deg = (r) => ((r * 180) / Math.PI + 360) % 360

describe('3-D math', () => {
  it('GMST matches Vallado Example 3-5 (1992-08-20 12:14 UT1 -> 152.5788 deg)', () => {
    expect(deg(gmstRad(new Date('1992-08-20T12:14:00Z')))).toBeCloseTo(152.5788, 2)
  })
  it('sun direction: ~+x at the March equinox, y(up) = declination sign in June', () => {
    const eq = sunDirScene(new Date('2026-03-20T14:46:00Z')); expect(eq.x).toBeGreaterThan(0.99); expect(Math.abs(eq.y)).toBeLessThan(0.02)
    expect(sunDirScene(new Date('2026-06-21T09:00:00Z')).y).toBeGreaterThan(0.39)       // declination ~ +23.4 deg
    expect(sunDirScene(new Date('2026-12-21T15:00:00Z')).y).toBeLessThan(-0.39)
  })
  it('TEME z-up maps to three y-up, right-handed (x,y,z) -> (x,z,-y)', () => {
    expect(teme2scene(7000, 1000, 500)).toEqual([7, 0.5, -1])
  })
})

describe('SatCloud interpolation (real R3F reconciler, no WebGL)', () => {
  const meta = { n_frames: 3, n_objects: 2, step_s: 60, ids: [11, 22], classes: ['PAYLOAD', 'DEBRIS'] }
  // (frame, object, xyz) in TEME km
  const data = new Float32Array([7000, 0, 0, 0, 8000, 0, /*f1*/ 7000, 1000, 0, 0, 8000, 2000, /*f2*/ 7000, 2000, 0, 0, 8000, 4000])
  it('writes interpolated positions (km/1000, axis-mapped) into the point buffer', async () => {
    const tRef = { current: 90 }                       // halfway between frame 1 and frame 2
    const r = await ReactThreeTestRenderer.create(<SatCloud pos={{ meta, data }} tRef={tRef} selected={22} />)
    await r.advanceFrames(1, 0.016)
    const pts = r.scene.children[0].children.find((c) => c.type === 'Points')
    const a = pts.instance.geometry.attributes.position.array
    // object 0 at t=90s: TEME y = 1500 -> scene z = -1.5 ; object 1: TEME z = 3000 -> scene y = 3
    expect(Array.from(a.slice(0, 3))).toEqual([7, 0, -1.5].map((v) => expect.closeTo(v, 4)))
    expect(Array.from(a.slice(3, 6))).toEqual([0, 3, -8].map((v) => expect.closeTo(v, 4)))
  })
  it('clamps time outside the window instead of extrapolating', async () => {
    const tRef = { current: 99999 }
    const r = await ReactThreeTestRenderer.create(<SatCloud pos={{ meta, data }} tRef={tRef} />)
    await r.advanceFrames(1, 0.016)
    const pts = r.scene.children[0].children.find((c) => c.type === 'Points')
    expect(pts.instance.geometry.attributes.position.array[2]).toBeCloseTo(-2, 1)    // last frame y=2000 km
  })
})
