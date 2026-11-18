import React, { useState } from 'react'
import { useApi } from '../api.js'
import { Panel, Badge, Notice, Spinner, Plot, Table, Stat, NeedData, CLASS_COLOR } from '../ui.jsx'

export default function Environment({ status }) {
  const [bin, setBin] = useState(50)
  const { data, loading, error } = useApi(status?.catalogue_loaded ? `/api/environment/stats?bin_km=${bin}` : null, [bin])
  const cls = data ? Object.keys(data.classes) : []
  return (
    <div className="h-full overflow-auto p-4 grid grid-cols-12 gap-4 content-start">
      <div className="col-span-12 flex items-center gap-3"><h2 className="text-lg font-semibold">Debris Environment</h2><Badge kind="observed">catalogue counts, not flux</Badge>
        <select className="inp" value={bin} onChange={e => setBin(+e.target.value)}>{[25, 50, 100].map(b => <option key={b} value={b}>{b} km bins</option>)}</select>{loading && <Spinner />}</div>
      <div className="col-span-12"><NeedData status={status} />{error && <Notice>{error}</Notice>}
        <Notice kind="derived">Shows what is in the public catalogue, not true debris flux: small, untracked debris is absent. Object classes come from CelesTrak group membership (or name heuristics), not ground truth.</Notice></div>
      {data && <>
        <div className="col-span-12 flex gap-3 flex-wrap">{cls.map(c => <Stat key={c} label={c} value={data.classes[c].count} hint={`median e = ${data.classes[c].ecc_median.toFixed(4)}`} />)}</div>
        <Panel className="col-span-7" title="Objects by mean altitude (LEO window)">
          <Plot height={300} data={cls.map(c => ({ type: 'bar', x: data.alt_edges_km, y: data.classes[c].alt_hist, name: c, marker: { color: CLASS_COLOR[c] } }))}
            layout={{ barmode: 'stack', xaxis: { title: 'mean altitude km', range: [0, 2100] }, yaxis: { title: 'objects' } }} />
        </Panel>
        <Panel className="col-span-5" title="Regions">
          <Table columns={[{ key: 'r', label: 'Region' }, ...cls.map(c => ({ key: c, label: c }))]}
            rows={Object.entries(data.regions).map(([r, v]) => ({ r, ...v }))} />
        </Panel>
        <Panel className="col-span-7" title="Catalogue spatial density by shell (per 10⁹ km³)">
          <Plot height={280} data={cls.map(c => ({ x: data.alt_edges_km, y: data.classes[c].density_per_1e9_km3, mode: 'lines', name: c, line: { color: CLASS_COLOR[c], shape: 'hv' } }))}
            layout={{ xaxis: { title: 'altitude km', range: [0, 2100] }, yaxis: { title: 'objects / 10⁹ km³', type: 'log' } }} />
        </Panel>
        <Panel className="col-span-5" title="Inclination distribution">
          <Plot height={280} data={cls.map(c => ({ type: 'bar', x: data.inc_edges_deg, y: data.classes[c].inc_hist, name: c, marker: { color: CLASS_COLOR[c] } }))}
            layout={{ barmode: 'stack', xaxis: { title: 'inclination °' }, yaxis: { title: 'objects' } }} />
        </Panel>
      </>}
    </div>
  )
}
