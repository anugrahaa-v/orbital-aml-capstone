import React, { useState } from 'react'
import Plotly from 'plotly.js-dist-min'
import createPlotlyComponent from 'react-plotly.js/factory'
const PlotlyChart = createPlotlyComponent(Plotly)

export const LABEL_STYLE = {
  observed: 'bg-emerald-500/15 text-emerald-300 border-emerald-400/40',
  derived: 'bg-sky-500/15 text-sky-300 border-sky-400/40',
  simulated: 'bg-amber-500/15 text-amber-300 border-amber-400/40',
  ml: 'bg-fuchsia-500/15 text-fuchsia-300 border-fuchsia-400/40',
  warn: 'bg-rose-500/15 text-rose-300 border-rose-400/40',
}
export const Badge = ({ kind = 'observed', children }) => (
  <span className={`inline-block text-[10px] uppercase tracking-wider px-2 py-0.5 rounded-full border ${LABEL_STYLE[kind]}`}>{children}</span>
)
export const Panel = ({ title, right, children, className = '' }) => (
  <section className={`glass p-4 ${className}`}>
    {(title || right) && <div className="flex items-center justify-between mb-3"><h3 className="text-sm font-semibold tracking-wide text-slate-200">{title}</h3><div className="flex gap-2 items-center">{right}</div></div>}
    {children}
  </section>
)
export const Stat = ({ label, value, unit, hint }) => (
  <div className="glass px-3 py-2 min-w-[120px]">
    <div className="text-[10px] uppercase tracking-wider text-slate-400">{label}</div>
    <div className="text-lg font-semibold text-slate-100">{value}<span className="text-xs text-slate-400 ml-1">{unit}</span></div>
    {hint && <div className="text-[10px] text-slate-500">{hint}</div>}
  </div>
)
export const Notice = ({ kind = 'warn', children }) => (
  <div className={`text-xs rounded-lg border px-3 py-2 ${LABEL_STYLE[kind]}`}>{children}</div>
)
export const Spinner = ({ text = 'Working…', progress }) => (
  <div className="flex items-center gap-2 text-xs text-slate-400">
    <div className="w-4 h-4 border-2 border-sky-400 border-t-transparent rounded-full animate-spin" />{text}{progress > 0 && progress < 1 && ` ${(progress * 100).toFixed(0)}%`}
  </div>
)
export const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x)) ? '–' : Math.abs(x) >= 1e5 || (Math.abs(x) < 1e-3 && x !== 0) ? Number(x).toExponential(2) : Number(x).toFixed(d)

const layoutBase = {
  paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: '#c7d3ee', size: 11 },
  margin: { l: 52, r: 16, t: 28, b: 42 }, legend: { orientation: 'h', y: -0.25 },
  xaxis: { gridcolor: 'rgba(255,255,255,.07)', zerolinecolor: 'rgba(255,255,255,.12)' },
  yaxis: { gridcolor: 'rgba(255,255,255,.07)', zerolinecolor: 'rgba(255,255,255,.12)' },
}
export function Plot({ data, layout = {}, height = 280, onClick }) {
  const merged = { ...layoutBase, ...layout, autosize: true, height,
    xaxis: { ...layoutBase.xaxis, ...(layout.xaxis || {}) }, yaxis: { ...layoutBase.yaxis, ...(layout.yaxis || {}) } }
  return <PlotlyChart data={data} layout={merged} config={{ displayModeBar: false, responsive: true }}
    style={{ width: '100%' }} useResizeHandler onClick={onClick} />
}
export const COLORS = ['#5b8dff', '#ff8a5b', '#4cd6a0', '#d46bff', '#ffd45b', '#5bd9ff', '#ff5b8a', '#a3ff5b', '#9d9dff', '#ffb35b']
export const CLASS_COLOR = { PAYLOAD: '#4cc2ff', DEBRIS: '#ff7a59', ROCKET_BODY: '#ffd84d' }

export function Table({ columns, rows, max = 360, onRow, active }) {
  return (
    <div style={{ maxHeight: max }} className="overflow-auto">
      <table className="t"><thead><tr>{columns.map(c => <th key={c.key}>{c.label}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => (
          <tr key={i} onClick={() => onRow && onRow(r)} className={`${onRow ? 'cursor-pointer hover:bg-white/5' : ''} ${active && active(r) ? 'bg-sky-500/15' : ''}`}>
            {columns.map(c => <td key={c.key}>{c.render ? c.render(r) : r[c.key]}</td>)}</tr>))}</tbody></table>
    </div>
  )
}
export function Field({ label, children }) {
  return <label className="flex flex-col gap-1 text-[11px] text-slate-400">{label}{children}</label>
}
export function Tabs({ tabs, value, onChange }) {
  return <div className="flex flex-wrap gap-1 mb-3">{tabs.map(t => (
    <button key={t.id} onClick={() => onChange(t.id)} className={`btn ${value === t.id ? 'btn-primary' : ''}`}>{t.label}</button>))}</div>
}
export function ObjectPicker({ value, onPick, label = 'Object', cls }) {
  const [q, setQ] = useState('')
  const [res, setRes] = useState([])
  const search = async (s) => {
    setQ(s)
    if (s.length < 2) { setRes([]); return }
    try { const r = await fetch(`/api/catalogue?q=${encodeURIComponent(s)}&limit=8${cls ? `&cls=${cls}` : ''}`).then(r => r.json()); setRes(r.objects || []) } catch { setRes([]) }
  }
  return (
    <div className="relative">
      <Field label={label}>
        <input className="inp w-56" placeholder="name or NORAD ID…" value={q} onChange={e => search(e.target.value)} />
      </Field>
      {value && <div className="text-[11px] text-sky-300 mt-1">{value.name} · {value.norad_id} · <span style={{ color: CLASS_COLOR[value.class] }}>{value.class}</span></div>}
      {res.length > 0 && <div className="absolute z-20 mt-1 w-72 glass max-h-56 overflow-auto">
        {res.map(o => <div key={o.norad_id} className="px-3 py-1.5 text-xs cursor-pointer hover:bg-white/10"
          onClick={() => { onPick(o); setRes([]); setQ('') }}>{o.name} <span className="text-slate-500">· {o.norad_id} · {o.class}</span></div>)}
      </div>}
    </div>
  )
}
export const NeedData = ({ status }) => (!status || status.catalogue_loaded) ? null : (
  <Notice kind="warn">
    No real catalogue has been ingested yet, so there is nothing to show (nothing is fabricated). Run
    <code className="mx-1 px-1 bg-black/40 rounded">python -m backend.ingest_cli --groups active cosmos-2251-debris iridium-33-debris fengyun-1c-debris cosmos-1408-debris</code>
    on a machine with internet access, then reload.
  </Notice>
)
