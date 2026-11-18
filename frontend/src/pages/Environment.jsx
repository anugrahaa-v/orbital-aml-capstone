import { get } from '../api.js'
import { useAsync, Panel, Notice } from '../components/ui.jsx'
import Plot, { CLASS_COLORS } from '../components/Plot.jsx'

export default function Environment() {
  const { loading, data: d, error } = useAsync(() => get('/environment/stats'), [])
  if (loading) return <div className="p-6 text-slate-400">Loading catalogue statistics…</div>
  if (error) return <div className="p-6"><Notice kind="warn">{error}</Notice></div>
  const classes = Object.keys(d.classes)
  const bars = (key, x) => classes.map((c) => ({ x, y: d.classes[c][key], type: 'bar', name: c, marker: { color: CLASS_COLORS[c] } }))
  return (
    <div className="p-4 grid gap-4 xl:grid-cols-2">
      <div className="xl:col-span-2"><Notice>All charts show <b>catalogue counts</b> from published element sets (observed + derived mean altitude). They are not debris flux, and untracked small debris is absent.</Notice></div>
      <Panel title="Objects per altitude bin (mean of perigee and apogee)" badge="derived"><Plot data={bars('alt_hist', d.alt_edges_km)} layout={{ barmode: 'stack', xaxis: { title: 'altitude km' }, yaxis: { title: 'objects' } }} /></Panel>
      <Panel title="Catalogue spatial density by shell (objects per 10⁹ km³)" badge="derived"><Plot data={classes.map((c) => ({ x: d.alt_edges_km, y: d.classes[c].density_per_1e9_km3, mode: 'lines', name: c, line: { color: CLASS_COLORS[c] } }))} layout={{ xaxis: { title: 'altitude km' }, yaxis: { title: 'density', type: 'log' } }} /></Panel>
      <Panel title="Inclination distribution" badge="observed"><Plot data={bars('inc_hist', d.inc_edges_deg)} layout={{ barmode: 'stack', xaxis: { title: 'inclination °' } }} /></Panel>
      <Panel title="Orbital regimes" badge="derived">
        <table className="t"><thead><tr><th>Regime</th>{classes.map((c) => <th key={c}>{c}</th>)}</tr></thead><tbody>
          {Object.entries(d.regions).map(([r, v]) => <tr key={r}><td>{r}</td>{classes.map((c) => <td key={c}>{(v[c] || 0).toLocaleString()}</td>)}</tr>)}</tbody></table>
        <div className="text-xs text-slate-400 mt-3">Class: {classes.map((c) => `${c}=${d.classes[c].count.toLocaleString()} (median e=${d.classes[c].ecc_median.toFixed(4)})`).join(' · ')}</div>
        <div className="text-xs text-slate-500 mt-1">Class labels come from CelesTrak group membership (name heuristic only for rocket bodies); they are not ground truth.</div>
      </Panel>
    </div>
  )
}
