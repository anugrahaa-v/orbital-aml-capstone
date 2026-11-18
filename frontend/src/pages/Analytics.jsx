import { get } from '../api.js'
import { useAsync, Panel, Notice, Badge, Stat } from '../components/ui.jsx'
import Plot from '../components/Plot.jsx'

export default function Analytics({ status }) {
  const { loading, data, error } = useAsync(() => get('/analytics/summary'), [])
  if (loading) return <div className="p-6 text-slate-400">Loading…</div>
  if (error) return <div className="p-6"><Notice kind="warn">{error}</Notice></div>
  const kinds = Object.keys(data.by_kind)
  const best = (rows, key, lower = true) => [...rows].filter((r) => r.metrics?.[key] != null).sort((a, b) => (lower ? a.metrics[key] - b.metrics[key] : b.metrics[key] - a.metrics[key]))
  const reg = best(data.by_kind.regression || [], 'rmse'), cls = best(data.by_kind.classification || [], 'roc_auc', false)
  const dl = best(data.by_kind.deep_learning || [], 'rmse'), rl = best(data.by_kind.reinforcement_learning || [], 'mean_final_separation_km', false)
  const h = data.history
  return (
    <div className="p-4 space-y-4">
      <Panel title="Data and history" badge="observed"><div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <Stat label="Objects" value={status?.n_objects?.toLocaleString() ?? '0'} /><Stat label="Stored element rows" value={h.rows.toLocaleString()} /><Stat label="Distinct epochs" value={h.distinct_epochs.toLocaleString()} />
        <Stat label="Max snapshots / object" value={h.max_snapshots_per_object} /><Stat label="Experiment runs" value={data.n_runs} /></div>
        {h.max_snapshots_per_object < 2 && <div className="mt-3"><Notice>Only a single snapshot per object exists, so forecasting targets are cross-sectional. Re-run <code>python -m backend.ingest_cli</code> on later days (at most once per ~2 h) to accumulate real history.</Notice></div>}
        <div className="mt-3 text-xs text-slate-400">{status?.provenance?.slice(-5).map((p, i) => <div key={i}>{p.retrieved_utc} · {p.n_records_raw} records · sha256 {p.dataset_version.slice(0, 12)} {p.from_cache ? '(cache)' : '(downloaded)'}</div>)}</div></Panel>
      {data.n_runs === 0 && <Notice>No experiments logged yet. Run something in ML Lab, Forecast &amp; DL or RL Planner.</Notice>}
      <div className="grid xl:grid-cols-2 gap-4">
        {reg.length > 0 && <Panel title="Regression leaderboard (RMSE, lower is better)" badge="ml"><Plot data={[{ type: 'bar', x: reg.slice(0, 13).map((r) => r.model), y: reg.slice(0, 13).map((r) => r.metrics.rmse) }]} layout={{ yaxis: { title: 'RMSE' } }} /><div className="text-xs text-slate-400">target: {reg[0].target} · {reg[0].data_label}</div></Panel>}
        {cls.length > 0 && <Panel title="Classification (ROC-AUC)" badge="ml"><Plot data={[{ type: 'bar', x: cls.map((r) => r.model), y: cls.map((r) => r.metrics.roc_auc) }]} layout={{ yaxis: { range: [0, 1] } }} /></Panel>}
        {dl.length > 0 && <Panel title="Deep-learning forecasts (MAE km, simulated series)" badge="ml"><Plot data={[{ type: 'bar', x: dl.map((r) => r.model), y: dl.map((r) => r.metrics.mae) }]} /></Panel>}
        {rl.length > 0 && <Panel title="RL policies (mean min separation km)" badge="simulated"><Plot data={[{ type: 'bar', x: rl.map((r) => r.model), y: rl.map((r) => r.metrics.mean_final_separation_km) }]} /></Panel>}
      </div>
      {kinds.length > 0 && <Panel title="All logged runs" badge="ml"><div className="scroll max-h-80"><table className="t"><thead><tr><th>Kind</th><th>Model</th><th>Target</th><th>Data</th><th>Metrics</th></tr></thead><tbody>
        {kinds.flatMap((k) => data.by_kind[k].map((r) => <tr key={r.run_id}><td>{k}</td><td>{r.model}</td><td>{r.target}</td><td>{r.data_label}</td><td className="text-slate-400">{Object.entries(r.metrics || {}).filter(([, v]) => typeof v === 'number').map(([a, v]) => `${a}=${v.toPrecision(4)}`).join('  ')}</td></tr>))}</tbody></table></div></Panel>}
    </div>
  )
}
