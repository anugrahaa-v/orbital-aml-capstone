import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import fx from './fixtures/api.json'

let lastJob = null
const route = (m, p, body) => {
  if (p === '/status') return fx.status
  if (p === '/catalogue') return fx.catalogue
  if (p.startsWith('/object/')) return fx.object_detail
  if (p === '/positions') return fx.positions
  if (p.startsWith('/orbit/')) return fx.orbit_path
  if (p === '/environment/stats') return fx.env_stats
  if (p === '/collision/closest') return fx.collision_closest
  if (p === '/collision/screen') return fx.collision_screen
  if (p === '/collision/pc') return fx.pc
  if (p.startsWith('/history/')) return fx.history
  if (p.startsWith('/nn/curve/')) return fx.nn_curve
  if (p === '/nn/neuron') return fx.nn_neuron
  if (p === '/nn/train') return fx.nn_train
  if (p === '/rl/episode') return fx.rl_episode
  if (p === '/analytics/summary') return fx.analytics
  if (p === '/ml/run') { lastJob = fx[`ml_${body.task}`]; return { job: 'j' } }
  if (p === '/rl/compare') { lastJob = fx.rl_compare; return { job: 'j' } }
  if (p === '/forecast/run') { lastJob = fx.forecast; return { job: 'j' } }
  if (p === '/dl/anomaly') { lastJob = fx.anomaly; return { job: 'j' } }
  throw new Error(`unmocked ${m} ${p}`)
}
vi.mock('../src/api.js', async (orig) => {
  const actual = await orig()
  return { ...actual, get: vi.fn(async (p, q) => route('GET', p, q)), post: vi.fn(async (p, b) => route('POST', p, b)), runJob: vi.fn(async (start) => { await start(); return lastJob }) }
})

import App from '../src/App.jsx'
import Environment from '../src/pages/Environment.jsx'
import MLLab from '../src/pages/MLLab.jsx'
import NNLab from '../src/pages/NNLab.jsx'
import RL from '../src/pages/RL.jsx'
import Analytics from '../src/pages/Analytics.jsx'
import Forecast from '../src/pages/Forecast.jsx'
import Collision from '../src/pages/Collision.jsx'
import Observatory from '../src/pages/Observatory.jsx'
import Landing from '../src/pages/Landing.jsx'
import { get } from '../src/api.js'

const click = async (name) => userEvent.click(await screen.findByRole('button', { name }))
const noop = () => {}
beforeEach(() => { vi.spyOn(console, 'error').mockImplementation(() => {}) })

describe('every page renders against real API payloads', () => {
  it('App shell + navigation to all 9 screens', async () => {
    render(<App />)
    expect(await screen.findByText(/CELESTRAK GP/)).toBeInTheDocument()
    for (const label of ['Observatory', 'Collision Lab', 'Debris Environment', 'Forecast & DL', 'ML Lab', 'Neural Nets', 'RL Planner', 'Analytics', 'Mission']) {
      await userEvent.click(screen.getByRole('button', { name: label }))
      await waitFor(() => expect(location.hash).toBe('#' + { Mission: 'home', Observatory: 'observatory', 'Collision Lab': 'collision', 'Debris Environment': 'environment', 'Forecast & DL': 'forecast', 'ML Lab': 'ml', 'Neural Nets': 'nn', 'RL Planner': 'rl', Analytics: 'analytics' }[label]))
    }
  })
  it('Landing (with and without catalogue)', async () => {
    const { unmount } = render(<Landing status={fx.status} go={noop} />)
    expect(screen.getByText(/CelesTrak GP \(one dataset\)/)).toBeInTheDocument(); unmount()
    render(<Landing status={{ catalogue_loaded: false }} go={noop} />)
    expect(screen.getByText(/No catalogue ingested/)).toBeInTheDocument()
  })
  it('Observatory loads positions and shows filters', async () => {
    render(<Observatory status={fx.status} setCtx={noop} go={noop} />)
    expect(await screen.findByText('Filters')).toBeInTheDocument()
    await waitFor(() => expect(get).toHaveBeenCalledWith('/positions', expect.anything()))
    expect(await screen.findByText(/objects · TEME frame/)).toBeInTheDocument()
  })
  it('Environment charts', async () => {
    render(<Environment />)
    expect(await screen.findByText(/catalogue counts/)).toBeInTheDocument()
    expect((await screen.findAllByTestId('plot')).length).toBeGreaterThanOrEqual(3)
    expect(screen.getByText('Orbital regimes')).toBeInTheDocument()
  })
  it.each([['Regression', /Model comparison/], ['Classification', /Classifiers/], ['Feature selection', /Selection methods/], ['PCA / LDA / ICA', /PCA/], ['K-Means', /Elbow/], ['Hierarchical', /Dendrogram/]])('ML Lab tab %s', async (tab, expected) => {
    render(<MLLab />)
    await userEvent.click(screen.getByRole('button', { name: tab }))
    await userEvent.click(screen.getByRole('button', { name: /^Run/ }))
    expect((await screen.findAllByText(expected))[0]).toBeInTheDocument()
    expect(screen.queryByText(/is not defined|Cannot read/)).toBeNull()
  })
  it('Neural nets: neuron, backend verify, trainer', async () => {
    render(<NNLab />)
    expect(screen.getByText(/output = sigmoid\(net\) = 0\.40131/)).toBeInTheDocument()   // net = -0.4 with the default inputs
    await click(/Verify with Python backend/)
    expect(await screen.findByText(/matches/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Backprop trainer' }))
    await click(/Train/)
    expect(await screen.findByText(/Accuracy/)).toBeInTheDocument()
  })
  it('RL episode and policy comparison', async () => {
    render(<RL />)
    await click(/Run episode/)
    expect(await screen.findByText('Decision log')).toBeInTheDocument()
    await click(/Compare policies/)
    expect(await screen.findByText(/Policy comparison over/)).toBeInTheDocument()
  })
  it('Analytics', async () => {
    render(<Analytics status={fx.status} />)
    expect(await screen.findByText(/Regression leaderboard/)).toBeInTheDocument()
    expect(screen.getByText('All logged runs')).toBeInTheDocument()
  })
  it('Forecast & DL: train and anomaly', async () => {
    render(<Forecast />)
    await userEvent.type(screen.getByPlaceholderText(/Search name/), 'DEB')
    await userEvent.click(await screen.findByText(fx.catalogue.objects[0].name, { exact: false }))
    await click(/Train and evaluate/)
    expect(await screen.findByText(/Test-set metrics/)).toBeInTheDocument()
    await click('Run')
    expect(await screen.findByText(/ROC-AUC/)).toBeInTheDocument()
  })
  it('Collision lab: closest approach, screening, Pc stays unavailable', async () => {
    render(<Collision ctx={{ a: fx.debris }} status={fx.status} />)
    await userEvent.type(screen.getByPlaceholderText(/Search name/), 'TEST')
    await userEvent.click((await screen.findAllByText(fx.catalogue.objects[1].name, { exact: false }))[0])
    await click(/Closest approach/)
    expect(await screen.findByText(/Separation vs time/)).toBeInTheDocument()
    expect(screen.getByText(/Miss distance/)).toBeInTheDocument()
    expect(screen.getAllByText(/UNAVAILABLE/).length).toBeGreaterThan(0)
    await click(/Screen primary/)
    expect(await screen.findByText(/Screening:/)).toBeInTheDocument()
  })
})
