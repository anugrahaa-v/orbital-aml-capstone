import React, { useState } from 'react'
import { useApi, useJob } from '../api.js'
import { Panel, Badge, Notice, Spinner, Field, ObjectPicker, Plot, Table, Stat, fmt, COLORS, NeedData } from '../ui.jsx'

const ARCHS = ['mlp', 'cnn1d', 'lstm', 'gru', 'cnn_lstm', 'transformer']

export default function Forecast({ status, select }) {
  const [obj, setObj] = useState(select || null)
  const [models, setModels] = useState(['gru', 'lstm', 'mlp'])
  const [epochs, setEpochs] = useState(20)
  const [days, setDays] = useState(6)
  const [job, run] = useJob()
  const [ae, runAe] = useJob()
  const hist = useApi(obj ? `/api/history/${obj.norad_id}` : null)
  const r = job.result
  const toggle = m => setModels(models.includes(m) ? models.filter(x => x !== m) : [...models, m])
  const names = r ? Object.keys(r.models) : []

  return (
    <div className="h-full overflow-auto p-4 grid grid-cols-12 gap-4 content-start">
      <div className="col-span-12 flex items-end gap-4 flex-wrap">
        <ObjectPicker label="Object" value={obj} onPick={setObj} />
        <Field label="Series length (days)"><input className="inp w-20" type="number" value={days} onChange={e => setDays(+e.target.value)} /></Field>
        <Field label="Epochs"><input className="inp w-20" type="number" value={epochs} onChange={e => setEpochs(+e.target.value)} /></Field>
        <div className="flex gap-2 flex-wrap text-xs">{ARCHS.map(m => <label key={m} className="flex gap-1 items-center"><input type="checkbox" checked={models.includes(m)} onChange={() => toggle(m)} />{m}</label>)}</div>
        <button className="btn btn-primary" disabled={!obj || job.status === 'running'} onClick={() => run('/api/forecast/run', { norad_id: obj.norad_id, days, epochs, models })}>Run forecast experiment</button>
        {job.status === 'running' && <Spinner text="Training" progress={job.progress} />}
      </div>
      <div className="col-span-12 space-y-2"><NeedData status={status} />
        <Notice kind="warn">The series below is <b>SGP4-generated (simulated)</b>. SGP4 is its generator, so it is the exact reference here, not a competitor to beat. A genuine SGP4-vs-reality comparison needs accumulated element-set history, shown in the backtest panel when available.</Notice>
        {job.error && <Notice>{job.error}</Notice>}</div>

      {r && !r.error && <>
        <Panel className="col-span-8" title="Forecasts on the held-out final segment" right={<><Badge kind="simulated">series: simulated</Badge><Badge kind="ml">models: ML</Badge></>}>
          <Plot height={330} data={[
            { y: r.series_test_target, mode: 'lines', name: 'SGP4 reference (altitude km)', line: { color: '#9fb4ff', width: 1 }, x: r.series_test_target.map((_, i) => i) },
            ...names.filter(n => r.models[n].examples).flatMap((n, k) => r.models[n].examples.idx.slice(0, 3).map((ix, j) => ({
              x: Array.from({ length: r.H }, (_, h) => (r.examples_origin_index[j] - r.split_index.val_end) + h), y: r.models[n].examples.pred[j], mode: 'lines', name: j === 0 ? n : undefined, showlegend: j === 0,
              line: { color: COLORS[k + 1], width: 2 } })))]}
            layout={{ xaxis: { title: 'sample index in test segment' }, yaxis: { title: 'altitude km' } }} />
        </Panel>
        <Panel className="col-span-4" title="Test error (km)">
          <Table max={280} columns={[{ key: 'm', label: 'Model' }, { key: 'mae', label: 'MAE', render: x => fmt(x.mae, 3) }, { key: 'rmse', label: 'RMSE', render: x => fmt(x.rmse, 3) }, { key: 'k', label: 'Type' }]}
            rows={names.map(n => ({ m: n, mae: r.models[n].mae, rmse: r.models[n].rmse, k: r.models[n].kind })).sort((a, b) => a.rmse - b.rmse)} />
          <div className="text-[10px] text-slate-500 mt-2">{r.n_windows.join(' / ')} train/val/test windows · chronological split · seed {r.seed}</div>
        </Panel>
        <Panel className="col-span-6" title="MAE by forecast step">
          <Plot height={250} data={names.map((n, k) => ({ y: r.models[n].mae_by_horizon, x: r.models[n].mae_by_horizon.map((_, i) => i + 1), mode: 'lines+markers', name: n, line: { color: COLORS[k] } }))} layout={{ xaxis: { title: 'steps ahead' }, yaxis: { title: 'km' } }} />
        </Panel>
        <Panel className="col-span-6" title="Training curves">
          <Plot height={250} data={names.filter(n => r.models[n].history).flatMap((n, k) => [
            { y: r.models[n].history.train_loss, name: `${n} train`, mode: 'lines', line: { color: COLORS[k] } },
            { y: r.models[n].history.val_loss, name: `${n} val`, mode: 'lines', line: { color: COLORS[k], dash: 'dot' } }])} layout={{ yaxis: { type: 'log', title: 'MSE (scaled)' }, xaxis: { title: 'epoch' } }} />
        </Panel>
      </>}
      {r?.error && <div className="col-span-12"><Notice>{r.error}</Notice></div>}

      <Panel className="col-span-6" title="Real history and SGP4 backtest" right={<Badge kind="observed">observed</Badge>}>
        {!obj && <div className="text-xs text-slate-500">Pick an object.</div>}
        {hist.data && <>
          <div className="text-xs text-slate-300 mb-2">{hist.data.n_snapshots} stored snapshot(s) for this object.</div>
          {hist.data.note && <Notice kind="derived">{hist.data.note}</Notice>}
          {hist.data.backtest?.n_pairs > 0 && <Plot height={220} data={[{ x: hist.data.backtest.pairs.map(p => p.lead_hours), y: hist.data.backtest.pairs.map(p => p.position_error_km), mode: 'markers+lines', name: 'position error' }]}
            layout={{ xaxis: { title: 'lead time (h)' }, yaxis: { title: 'km (SGP4 from set k vs set k+1)' } }} />}
        </>}
      </Panel>
      <Panel className="col-span-6" title="Autoencoder anomaly detection" right={<Badge kind="ml">ML · group-derived labels</Badge>}>
        <div className="text-xs text-slate-400 mb-2">Trained on older PAYLOAD objects only; scores newer objects by reconstruction error. “Anomaly” = debris-group label (not ground truth).</div>
        <button className="btn" onClick={() => runAe('/api/dl/anomaly', null)} disabled={ae.status === 'running'}>Run</button>{ae.status === 'running' && <Spinner />}
        {ae.error && <Notice>{ae.error}</Notice>}
        {ae.result && <div className="mt-2 text-xs flex gap-3 flex-wrap">
          <Stat label="ROC-AUC" value={fmt(ae.result.roc_auc, 3)} /><Stat label="Precision" value={fmt(ae.result.precision, 3)} /><Stat label="Recall" value={fmt(ae.result.recall, 3)} /><Stat label="F1" value={fmt(ae.result.f1, 3)} /></div>}
      </Panel>
    </div>
  )
}
