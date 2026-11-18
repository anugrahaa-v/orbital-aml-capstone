import { useState } from 'react'
import { get, post } from '../api.js'
import { Panel, Notice, Badge, ObjectPicker, useJobRunner, Progress, useAsync } from '../components/ui.jsx'
import Plot from '../components/Plot.jsx'

const ARCH = ['mlp', 'cnn1d', 'lstm', 'gru', 'cnn_lstm', 'transformer']

export default function Forecast({ status }) {
  const [obj, setObj] = useState(null); const [models, setModels] = useState(['gru', 'lstm', 'mlp']); const [epochs, setEpochs] = useState(20); const [days, setDays] = useState(6)
  const [job, run] = useJobRunner(); const [ae, runAe] = useJobRunner(); const [k, setK] = useState(0)
  const hist = useAsync(() => (obj ? get(`/history/${obj.norad_id}`) : Promise.resolve(null)), [obj])
  const r = job.result
  const toggle = (m) => setModels(models.includes(m) ? models.filter((x) => x !== m) : [...models, m])
  const names = r ? Object.keys(r.models) : []
  return (
    <div className="p-4 grid gap-4 xl:grid-cols-[340px_1fr]">
      <div className="space-y-4">
        {status && !status.torch && <Notice kind="warn">PyTorch is not installed, so the deep-learning models below cannot run. Install it with <code>pip install torch</code> and restart the API. The rest of the app is unaffected.</Notice>}
        <Panel title="Forecast experiment" badge="simulated">
          <ObjectPicker label="Object (eccentric orbits are the most informative)" value={obj} onPick={setObj} />
          <div className="mt-3 grid grid-cols-2 gap-2 text-[13px]">
            <label><div className="label">Days of series: {days}</div><input type="range" min="2" max="14" value={days} onChange={(e) => setDays(+e.target.value)} className="w-full" /></label>
            <label><div className="label">Epochs: {epochs}</div><input type="range" min="5" max="60" value={epochs} onChange={(e) => setEpochs(+e.target.value)} className="w-full" /></label>
          </div>
          <div className="flex flex-wrap gap-2 mt-3">{ARCH.map((m) => <label key={m} className={`text-xs px-2 py-1 rounded border cursor-pointer ${models.includes(m) ? 'border-sky-400 bg-sky-500/10' : 'border-white/15'}`}><input type="checkbox" className="hidden" checked={models.includes(m)} onChange={() => toggle(m)} />{m}</label>)}</div>
          <button className="btn mt-3 w-full" disabled={!obj || job.running || !models.length} onClick={() => run(() => post('/forecast/run', { norad_id: obj.norad_id, days, epochs, models }))}>{job.running ? 'Training…' : 'Train and evaluate'}</button>
          <div className="mt-2"><Progress s={job} /></div>
          {job.error && <div className="mt-2"><Notice kind="error">{job.error}</Notice></div>}
        </Panel>
        <Panel title="Autoencoder anomaly detection" badge="ml">
          <div className="text-xs text-slate-400 mb-2">Trained on older PAYLOAD objects only; DEBRIS-group objects are the "anomalies" (group-derived labels, not ground truth).</div>
          <button className="btn-ghost" disabled={ae.running} onClick={() => runAe(() => post('/dl/anomaly'))}>{ae.running ? 'Training…' : 'Run'}</button>
          {ae.error && <div className="mt-2"><Notice kind="error">{ae.error}</Notice></div>}
          {ae.result && <div className="mt-3 text-[13px] space-y-1">{ae.result.roc_auc != null ? <>ROC-AUC <b>{ae.result.roc_auc.toFixed(3)}</b> · precision {ae.result.precision.toFixed(2)} · recall {ae.result.recall.toFixed(2)} · F1 {ae.result.f1.toFixed(2)}</> : 'Only one class in test split.'}
            <Plot height={200} data={[0, 1].map((l) => ({ type: 'histogram', x: ae.result.scores_sample.score.filter((_, i) => ae.result.scores_sample.label[i] === l), name: l ? 'debris-group' : 'payload', opacity: 0.7 }))} layout={{ barmode: 'overlay', xaxis: { type: 'log', title: 'reconstruction error' } }} /></div>}
        </Panel>
        <Panel title="Real history and SGP4 back-test" badge="observed">
          {!obj ? <div className="text-xs text-slate-400">Select an object.</div> : hist.data?.n_snapshots >= 2 ? (
            <Plot height={200} data={[{ x: hist.data.backtest.pairs.map((p) => p.lead_hours), y: hist.data.backtest.pairs.map((p) => p.position_error_km), mode: 'markers+lines', name: 'error' }]} layout={{ xaxis: { title: 'lead time h' }, yaxis: { title: 'position error km' } }} />
          ) : <Notice>{hist.data?.note || hist.error || 'Loading…'}</Notice>}
        </Panel>
      </div>

      <div className="space-y-4 min-w-0">
        {!r && <Panel><div className="text-sm text-slate-400">Choose an object and train. The series is the SGP4-propagated altitude of the object's real element set. <b>SGP4 generates it, so SGP4 is the reference (exact by construction), not a competitor.</b> The models are learned surrogates compared against persistence and a linear baseline. A true observed-vs-SGP4 comparison appears in the back-test panel once ≥2 snapshots have been ingested.</div></Panel>}
        {r && <>
          <Notice kind="warn">{r.note}</Notice>
          <Panel title="Test-set metrics (chronological split, scaler fit on train only)" badge="ml" right={<Badge kind="simulated" />}>
            <table className="t"><thead><tr><th>Model</th><th>Kind</th><th>MAE km</th><th>RMSE km</th><th>Params</th><th>Epochs</th><th>Seconds</th></tr></thead><tbody>
              {names.map((n) => { const m = r.models[n]; return <tr key={n}><td>{n}</td><td>{m.kind}</td><td>{m.mae.toFixed(3)}</td><td>{m.rmse.toFixed(3)}</td><td>{m.params ?? '-'}</td><td>{m.epochs_run ?? '-'}</td><td>{m.seconds ?? '-'}</td></tr> })}</tbody></table>
          </Panel>
          <div className="grid xl:grid-cols-2 gap-4">
            <Panel title="Forecast vs truth (test window)" badge="ml">
              <label className="text-xs text-slate-400">Forecast origin {k + 1}/{r.examples_truth.length}<input type="range" className="w-full" min="0" max={r.examples_truth.length - 1} value={k} onChange={(e) => setK(+e.target.value)} /></label>
              <Plot height={280} data={[{ y: r.examples_truth[k], mode: 'lines+markers', name: 'SGP4 reference (truth)', line: { color: '#e2e8f0' } },
                ...names.filter((n) => r.models[n].examples).map((n) => ({ y: r.models[n].examples.pred[k], mode: 'lines', name: n }))]} layout={{ xaxis: { title: `steps ahead (×${r.step_s}s)` }, yaxis: { title: 'altitude km' } }} />
            </Panel>
            <Panel title="MAE by forecast step" badge="ml"><Plot height={300} data={names.map((n) => ({ y: r.models[n].mae_by_horizon, mode: 'lines', name: n }))} layout={{ xaxis: { title: 'steps ahead' }, yaxis: { title: 'MAE km', type: 'log' } }} /></Panel>
            <Panel title="Training and validation loss" badge="ml" className="xl:col-span-2"><Plot height={280} data={names.filter((n) => r.models[n].history).flatMap((n) => [{ y: r.models[n].history.train_loss, name: `${n} train`, mode: 'lines' }, { y: r.models[n].history.val_loss, name: `${n} val`, mode: 'lines', line: { dash: 'dot' } }])} layout={{ yaxis: { type: 'log', title: 'MSE (scaled)' }, xaxis: { title: 'epoch' } }} /></Panel>
          </div>
        </>}
      </div>
    </div>
  )
}
