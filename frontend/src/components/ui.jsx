import { useState, useCallback, useEffect } from 'react'
import { runJob, get } from '../api.js'

const BADGES = {
  observed: ['OBSERVED', 'bg-emerald-500/15 text-emerald-300 border-emerald-400/30'],
  derived: ['DERIVED', 'bg-sky-500/15 text-sky-300 border-sky-400/30'],
  simulated: ['SIMULATED', 'bg-amber-500/15 text-amber-300 border-amber-400/30'],
  ml: ['ML PREDICTION', 'bg-fuchsia-500/15 text-fuchsia-300 border-fuchsia-400/30'],
  unavailable: ['UNAVAILABLE', 'bg-rose-500/15 text-rose-300 border-rose-400/30'],
}
export function Badge({ kind, children }) {
  const [t, c] = BADGES[kind] || BADGES.derived
  return <span className={`inline-block text-[10px] font-bold tracking-wider px-2 py-0.5 rounded-full border ${c}`}>{children || t}</span>
}
export function Panel({ title, badge, right, children, className = '' }) {
  return (
    <section className={`glass p-4 ${className}`}>
      {(title || badge || right) && (
        <header className="flex items-center gap-2 mb-3">
          {title && <h3 className="text-sm font-semibold text-slate-100">{title}</h3>}
          {badge && <Badge kind={badge} />}
          <div className="ml-auto flex gap-2">{right}</div>
        </header>
      )}
      {children}
    </section>
  )
}
export function Notice({ kind = 'info', children }) {
  const c = kind === 'error' ? 'border-rose-400/40 bg-rose-500/10 text-rose-200' : kind === 'warn' ? 'border-amber-400/40 bg-amber-500/10 text-amber-100' : 'border-sky-400/30 bg-sky-500/10 text-sky-100'
  return <div className={`text-[13px] rounded-lg border px-3 py-2 ${c}`}>{children}</div>
}
export function Stat({ label, value, unit }) {
  return <div><div className="label">{label}</div><div className="text-lg font-semibold text-slate-50">{value}{unit && <span className="text-xs text-slate-400 ml-1">{unit}</span>}</div></div>
}
export function useJobRunner() {
  const [state, set] = useState({ running: false, progress: 0, result: null, error: null })
  const run = useCallback(async (start) => {
    set({ running: true, progress: 0, result: null, error: null })
    try {
      const result = await runJob(start, (j) => set((s) => ({ ...s, progress: j.progress })))
      set({ running: false, progress: 1, result, error: null }); return result
    } catch (e) { set({ running: false, progress: 0, result: null, error: e.message }) }
  }, [])
  return [state, run]
}
export function useAsync(fn, deps) {
  const [s, set] = useState({ loading: true, data: null, error: null })
  useEffect(() => {
    let live = true; set((x) => ({ ...x, loading: true }))
    fn().then((d) => live && set({ loading: false, data: d, error: null })).catch((e) => live && set({ loading: false, data: null, error: e.message }))
    return () => { live = false }
    // eslint-disable-next-line
  }, deps)
  return s
}
export function Progress({ s }) {
  if (!s.running) return null
  return <div className="h-1.5 rounded bg-white/10 overflow-hidden"><div className="h-full bg-blue-500 animate-pulse" style={{ width: `${Math.max(8, s.progress * 100)}%` }} /></div>
}
export function ObjectPicker({ label, value, onPick }) {
  const [q, setQ] = useState(''); const [res, setRes] = useState([])
  useEffect(() => {
    if (q.length < 2) { setRes([]); return }
    const h = setTimeout(() => get('/catalogue', { q, limit: 8 }).then((r) => setRes(r.objects)).catch(() => setRes([])), 250)
    return () => clearTimeout(h)
  }, [q])
  return (
    <div className="relative">
      <div className="label mb-1">{label}</div>
      <input className="inp w-full" placeholder={value ? `${value.name} (${value.norad_id})` : 'Search name or NORAD ID'} value={q} onChange={(e) => setQ(e.target.value)} />
      {res.length > 0 && (
        <div className="absolute z-20 mt-1 w-full glass max-h-56 scroll">
          {res.map((o) => <button key={o.norad_id} className="block w-full text-left px-3 py-1.5 text-[13px] hover:bg-white/10" onClick={() => { onPick(o); setQ(''); setRes([]) }}>{o.name} <span className="text-slate-400">#{o.norad_id} · {o.class}</span></button>)}
        </div>
      )}
    </div>
  )
}
