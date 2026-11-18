import React from 'react'
import { useApi } from '../api.js'
import { Panel, Badge, Notice, Plot, Table, Stat, fmt, COLORS } from '../ui.jsx'

export default function Analytics({ status }) {
  const { data, error } = useApi('/api/analytics/summary')
  const kinds = data ? Object.keys(data.by_kind) : []
  const reg = data?.by_kind?.regression || []
  const cls = data?.by_kind?.classification || []
  return (
    <div className="h-full overflow-auto p-4 grid grid-cols-12 gap-4 content-start">
      <div className="col-span-12 flex items-center gap-3"><h2 className="text-lg font-semibold">Analytics & Provenance</h2></div>
      {error && <div className="col-span-12"><Notice>{error}</Notice></div>}
      <div className="col-span-12 flex gap-3 flex-wrap">
        <Stat label="Objects loaded" value={status?.n_objects ?? 0} /><Stat label="Rejected records" value={status?.rejected_records ?? 0} />
        <Stat label="Stored experiment runs" value={data?.n_runs ?? 0} />
        <Stat label="History snapshots (max / object)" value={data?.history?.max_snapshots_per_object ?? 0} hint={`${data?.history?.distinct_epochs ?? 0} distinct epochs`} />
        <Stat label="Torch" value={status?.torch ?? 'not installed'} />
      </div>
      <Panel className="col-span-6" title="Data provenance (source registry)" right={<Badge kind="observed">CelesTrak GP only</Badge>}>
        {status?.provenance?.length ? <Table columns={[{ key: 'retrieved_utc', label: 'Retrieved' }, { key: 'n_records_raw', label: 'Records' }, { key: 'dataset_version', label: 'SHA-256', render: r => r.dataset_version.slice(0, 12) }, { key: 'from_cache', label: 'Cache', render: r => r.from_cache ? 'yes' : 'no' }]} rows={status.provenance} /> : <div className="text-xs text-slate-500">No retrievals logged yet.</div>}
        <div className="text-[11px] text-slate-500 mt-2">Source: CelesTrak GP (derived from 18th/19th SDS via Space-Track). Fair-use: one download per update cycle (~2 h).</div>
      </Panel>
      <Panel className="col-span-6" title="Experiments by kind">
        {kinds.length ? <Plot height={240} data={[{ type: 'bar', x: kinds, y: kinds.map(k => data.by_kind[k].length), marker: { color: COLORS[0] } }]} /> : <div className="text-xs text-slate-500">Run experiments in the ML Lab, Forecast or RL screens.</div>}
      </Panel>
      {reg.length > 0 && <Panel className="col-span-6" title="Regression runs: RMSE (lower is better)">
        <Plot height={280} data={[{ type: 'bar', x: reg.map(r => `${r.model}`), y: reg.map(r => r.metrics?.rmse), marker: { color: COLORS[2] } }]} />
      </Panel>}
      {cls.length > 0 && <Panel className="col-span-6" title="Classification runs: ROC-AUC">
        <Plot height={280} data={[{ type: 'bar', x: cls.map(r => r.model), y: cls.map(r => r.metrics?.roc_auc), marker: { color: COLORS[3] } }]} layout={{ yaxis: { range: [0, 1] } }} />
      </Panel>}
      <Panel className="col-span-12" title="All logged runs">
        <Table max={360} columns={[{ key: 'kind', label: 'Kind' }, { key: 'model_name', label: 'Model' }, { key: 'target', label: 'Target' }, { key: 'data_label', label: 'Data label' }, { key: 'seed', label: 'Seed' }, { key: 'metrics', label: 'Metrics', render: r => Object.entries(r.metrics || {}).filter(([, v]) => typeof v === 'number').map(([k, v]) => `${k}=${fmt(v, 4)}`).join('  ') }]}
          rows={kinds.flatMap(k => data.by_kind[k].map(r => ({ kind: k, model_name: r.model, target: r.target, data_label: r.data_label, seed: '', metrics: r.metrics })))} />
      </Panel>
    </div>
  )
}
