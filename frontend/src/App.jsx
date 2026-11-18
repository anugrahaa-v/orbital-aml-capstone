import { useState, useEffect } from 'react'
import { get } from './api.js'
import { Badge } from './components/ui.jsx'
import Landing from './pages/Landing.jsx'
import Observatory from './pages/Observatory.jsx'
import Collision from './pages/Collision.jsx'
import Environment from './pages/Environment.jsx'
import Forecast from './pages/Forecast.jsx'
import MLLab from './pages/MLLab.jsx'
import NNLab from './pages/NNLab.jsx'
import RL from './pages/RL.jsx'
import Analytics from './pages/Analytics.jsx'

const PAGES = [
  ['home', 'Mission', Landing], ['observatory', 'Observatory', Observatory], ['collision', 'Collision Lab', Collision],
  ['environment', 'Debris Environment', Environment], ['forecast', 'Forecast & DL', Forecast], ['ml', 'ML Lab', MLLab],
  ['nn', 'Neural Nets', NNLab], ['rl', 'RL Planner', RL], ['analytics', 'Analytics', Analytics],
]

export default function App() {
  const [page, setPage] = useState(() => location.hash.slice(1) || 'home')
  const [status, setStatus] = useState(null)
  const [ctx, setCtx] = useState({})            // cross-page context, e.g. {pair:[a,b]}
  useEffect(() => { location.hash = page }, [page])
  useEffect(() => { get('/status').then(setStatus).catch((e) => setStatus({ error: e.message })) }, [])
  const Page = PAGES.find((p) => p[0] === page)[2]
  return (
    <div className="h-full flex flex-col">
      <nav className="flex items-center gap-1 px-4 py-2 border-b border-white/10 bg-[#070b16]/90 z-30 overflow-x-auto">
        <div className="font-bold tracking-[.2em] text-sky-300 mr-4 whitespace-nowrap">ORBITALX<span className="text-slate-400"> AI</span></div>
        {PAGES.map(([k, label]) => (
          <button key={k} onClick={() => setPage(k)} className={`px-3 py-1.5 rounded-lg text-[13px] whitespace-nowrap ${page === k ? 'bg-blue-600/30 text-sky-200' : 'text-slate-400 hover:text-slate-100 hover:bg-white/5'}`}>{label}</button>
        ))}
        <div className="ml-auto pl-4 whitespace-nowrap text-xs text-slate-400">
          {status?.catalogue_loaded ? <><Badge kind="observed">CELESTRAK GP</Badge> <span className="ml-2">{status.n_objects.toLocaleString()} objects</span></> : <Badge kind="unavailable">NO DATA INGESTED</Badge>}
        </div>
      </nav>
      <main className="flex-1 min-h-0 overflow-auto"><Page status={status} go={setPage} ctx={ctx} setCtx={setCtx} /></main>
    </div>
  )
}
