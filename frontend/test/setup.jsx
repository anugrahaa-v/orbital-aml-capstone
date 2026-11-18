import '@testing-library/jest-dom/vitest'
import { vi } from 'vitest'

// jsdom has no WebGL/ResizeObserver: stub the 3-D stack and Plotly. Everything else (pages, hooks, API contract) is real.
globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} }
vi.mock('@react-three/fiber', () => ({ Canvas: () => null, useFrame: () => {}, useThree: () => ({}) }))
vi.mock('@react-three/drei', () => ({ OrbitControls: () => null, Stars: () => null, Line: () => null, useTexture: () => [{}, {}, {}] }))
vi.mock('react-plotly.js/factory', () => ({ default: () => (props) => <div data-testid="plot" data-traces={(props.data || []).length} /> }))
vi.mock('plotly.js-dist-min', () => ({ default: {} }))
