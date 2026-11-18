import React, { useState } from 'react'
import { useJob } from '../api.js'
import { Panel, Badge, Notice, Spinner, Plot, Table, Stat, Tabs, fmt, COLORS, CLASS_COLOR, NeedData } from '../ui.jsx'

const TABS = [['regression', 'Regression'], ['classification', 'Classification'], ['feature_selection', 'Feature selection'],
  ['dimred', 'PCA / LDA / ICA'], ['kmeans', 'K-Means'], ['hierarchical', 'Hierarchical']].map(([id, label]) => ({ id, label }))

function Heat({ z, x, y, title }) {
  return <Plot height={300} data={[{ type: 'heatmap', z, x, y, colorscale: 'RdBu', zmin: -1, zmax: 1, reversescale: true }]} layout={{ title: { text: title, font: { size: 12 } }, margin: { l: 110, b: 90, t: 30, r: 10 }, yaxis: { autorange: 'reversed' } }} />
}

function Body({ task, r }) {
  const [sel, setSel] = useState(0)
  if (r.error) return <Notice>{r.error}</Notice>
  if (task === 'regression') {
    const m = r.models[sel] || r.models[0]
    return <div className="grid grid-cols-12 gap-4">
      <Panel className="col-span-7" title={`Leaderboard · target: ${r.target}`} right={<Badge kind="ml">test = final 20% by epoch</Badge>}>
        <Table active={x => x.model === m.model} onRow={x => setSel(r.models.findIndex(z => z.model === x.model))}
          columns={[{ key: 'model', label: 'Model' }, { key: 'family', label: 'Family' }, { key: 'rmse', label: 'RMSE', render: x => fmt(x.test?.rmse, 4) }, { key: 'mae', label: 'MAE', render: x => fmt(x.test?.mae, 4) },
            { key: 'r2', label: 'R²', render: x => fmt(x.test?.r2, 3) }, { key: 'cv', label: 'CV MAE', render: x => fmt(x.cv_mae, 4) }, { key: 'p', label: 'Best params', render: x => x.error ? x.error : JSON.stringify(x.best_params) }]}
          rows={r.models} />
        <div className="text-xs text-slate-400 mt-2">Baseline (predict train mean): RMSE {fmt(r.baseline_mean.rmse, 4)}, R² {fmt(r.baseline_mean.r2, 3)}. {r.n_train} train+val / {r.n_test} test rows. XGBoost: {r.xgboost_available ? 'included' : 'not installed'}.</div>
      </Panel>
      <Panel className="col-span-5" title="RMSE by model">
        <Plot height={300} data={[{ type: 'bar', orientation: 'h', y: r.models.filter(x => x.test).map(x => x.model), x: r.models.filter(x => x.test).map(x => x.test.rmse), marker: { color: COLORS[0] } }]}
          layout={{ shapes: [{ type: 'line', x0: r.baseline_mean.rmse, x1: r.baseline_mean.rmse, y0: 0, y1: 1, yref: 'paper', line: { color: '#ff8a5b', dash: 'dot' } }], yaxis: { autorange: 'reversed' }, margin: { l: 120 } }} />
      </Panel>
      {m.actual_vs_pred && <>
        <Panel className="col-span-6" title={`${m.model}: actual vs predicted`}><Plot height={280} data={[{ x: m.actual_vs_pred.y_true, y: m.actual_vs_pred.y_pred, mode: 'markers', marker: { size: 4, color: COLORS[2] } }]} layout={{ xaxis: { title: 'actual' }, yaxis: { title: 'predicted' } }} /></Panel>
        <Panel className="col-span-6" title={`${m.model}: residuals`}><Plot height={280} data={[{ type: 'histogram', x: m.residuals, marker: { color: COLORS[3] } }]} layout={{ xaxis: { title: 'residual' } }} /></Panel>
      </>}
    </div>
  }
  if (task === 'classification') {
    const m = r.models[sel]
    return <div className="grid grid-cols-12 gap-4">
      <Panel className="col-span-7" title="Metrics (chronological test set)" right={<Badge kind="warn">labels = group-derived</Badge>}>
        <Table active={x => x.model === m.model} onRow={x => setSel(r.models.findIndex(z => z.model === x.model))}
          columns={['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'].map(k => ({ key: k, label: k, render: x => k === 'model' ? x.model : fmt(x[k], 3) }))} rows={r.models} />
        <div className="text-xs text-slate-400 mt-2">{r.label_note}. Positive class: {r.positive_class}.</div>
      </Panel>
      <Panel className="col-span-5" title="ROC curves"><Plot height={300} data={[...r.models.map((x, k) => ({ x: x.roc.fpr, y: x.roc.tpr, mode: 'lines', name: `${x.model} (${fmt(x.roc_auc, 2)})`, line: { color: COLORS[k] } })), { x: [0, 1], y: [0, 1], mode: 'lines', line: { dash: 'dot', color: '#667' }, showlegend: false }]} layout={{ xaxis: { title: 'FPR' }, yaxis: { title: 'TPR' } }} /></Panel>
      <Panel className="col-span-4" title={`Confusion matrix · ${m.model}`}><Plot height={260} data={[{ type: 'heatmap', z: m.confusion_matrix, x: r.confusion_labels.map(l => `pred ${l}`), y: r.confusion_labels.map(l => `true ${l}`), colorscale: 'Blues', text: m.confusion_matrix, texttemplate: '%{text}' }]} layout={{ yaxis: { autorange: 'reversed' } }} /></Panel>
    </div>
  }
  if (task === 'feature_selection') return <div className="grid grid-cols-12 gap-4">
    <Panel className="col-span-7" title={`Selection methods · target ${r.target}`} right={<Badge kind="derived">fit on train segment</Badge>}>
      <Table columns={[{ key: 'feature', label: 'Feature' }, { key: 'votes', label: 'Votes' }, { key: 'c', label: 'Corr', render: x => fmt(x.corr_with_target, 3) }, { key: 'f', label: 'F', render: x => fmt(x.f_score, 1) }, { key: 'mi', label: 'MI', render: x => fmt(x.mutual_info, 3) },
        { key: 'rfe', label: 'RFE rank', render: x => x.rfe_rank }, { key: 'rf', label: 'RF imp', render: x => fmt(x.rf_importance, 3) }, { key: 'l', label: 'Lasso', render: x => fmt(x.lasso_coef, 3) }]} rows={r.table} />
    </Panel>
    <Panel className="col-span-5" title="Correlation matrix"><Heat z={r.correlation_matrix.values} x={r.correlation_matrix.labels} y={r.correlation_matrix.labels} /></Panel>
  </div>
  if (task === 'dimred') return <div className="grid grid-cols-12 gap-4">
    {Object.entries(r.methods).map(([k, m]) => <Panel key={k} className="col-span-4" title={`${k} (${m.n_components} comp.)`}>
      <Plot height={260} data={[...new Set(m.projection.label)].map(l => ({ x: m.projection.x.filter((_, i) => m.projection.label[i] === l), y: m.projection.y.filter((_, i) => m.projection.label[i] === l), mode: 'markers', name: l, marker: { size: 4, color: CLASS_COLOR[l] } }))} />
      <div className="text-xs text-slate-400">kNN acc {fmt(m.knn_accuracy, 3)} · macro-F1 {fmt(m.knn_macro_f1, 3)}{m.explained_variance_ratio && ` · EVR ${m.explained_variance_ratio.map(v => fmt(v, 2)).join(', ')}`}</div></Panel>)}
    {r.lda_note && <div className="col-span-12"><Notice kind="derived">{r.lda_note}</Notice></div>}
  </div>
  if (task === 'kmeans') return <div className="grid grid-cols-12 gap-4">
    <div className="col-span-12 flex gap-3"><Stat label="Optimal K (silhouette)" value={r.optimal_k_silhouette} /><Stat label="Elbow K" value={r.elbow_k} /><Stat label="Silhouette" value={fmt(r.metrics_at_optimum.silhouette, 3)} /><Stat label="Davies-Bouldin" value={fmt(r.metrics_at_optimum.davies_bouldin, 3)} /><Stat label="Calinski-Harabasz" value={fmt(r.metrics_at_optimum.calinski_harabasz, 0)} /></div>
    <Panel className="col-span-4" title="Elbow (inertia)"><Plot height={240} data={[{ x: r.sweep.map(s => s.k), y: r.sweep.map(s => s.inertia), mode: 'lines+markers' }]} layout={{ xaxis: { title: 'K' } }} /></Panel>
    <Panel className="col-span-4" title="Silhouette vs K"><Plot height={240} data={[{ x: r.sweep.map(s => s.k), y: r.sweep.map(s => s.silhouette), mode: 'lines+markers', line: { color: COLORS[2] } }]} layout={{ xaxis: { title: 'K' } }} /></Panel>
    <Panel className="col-span-4" title="Cluster composition">
      <Table columns={[{ key: 'k', label: 'Cluster' }, { key: 'v', label: 'Class counts' }]} rows={Object.entries(r.cluster_vs_class).map(([k, v]) => ({ k, v: JSON.stringify(v) }))} /></Panel>
    <Panel className="col-span-12" title="3-D PCA projection">
      <Plot height={420} data={[{ type: 'scatter3d', mode: 'markers', x: r.points.x, y: r.points.y, z: r.points.z, text: r.points.name, marker: { size: 2.5, color: r.points.cluster, colorscale: 'Turbo' } }]} layout={{ margin: { l: 0, r: 0, t: 0, b: 0 }, scene: { xaxis: { title: 'PC1' }, yaxis: { title: 'PC2' }, zaxis: { title: 'PC3' } } }} />
    </Panel>
  </div>
  if (task === 'hierarchical') {
    const d = r.dendrogram
    return <div className="grid grid-cols-12 gap-4">
      <Panel className="col-span-7" title={`Dendrogram (${d.linkage}, ${d.n_points} pts, truncated)`}>
        <Plot height={360} data={d.icoord.map((xs, i) => ({ x: xs, y: d.dcoord[i], mode: 'lines', line: { color: COLORS[0], width: 1 }, hoverinfo: 'skip', showlegend: false }))} layout={{ xaxis: { showticklabels: false }, yaxis: { title: 'distance' } }} />
      </Panel>
      <Panel className="col-span-5" title={`Linkage comparison (k=${r.n_clusters})`}>
        <Table columns={[{ key: 'l', label: 'Linkage' }, { key: 's', label: 'Silhouette' }, { key: 'd', label: 'DB' }, { key: 'c', label: 'CH' }]}
          rows={Object.entries(r.linkages).map(([l, m]) => ({ l, s: fmt(m.silhouette, 3), d: fmt(m.davies_bouldin, 3), c: fmt(m.calinski_harabasz, 0) }))} />
        <div className="text-xs text-slate-400 mt-2">Best by silhouette: {r.best_linkage_by_silhouette}</div></Panel>
    </div>
  }
  return null
}

export default function MLLab({ status }) {
  const [task, setTask] = useState('regression')
  const [job, run] = useJob()
  const [res, setRes] = useState({})
  const [seed, setSeed] = useState(42)
  const go = async () => { await run('/api/ml/run', { task, seed }) }
  const r = job.status === 'done' && job.result ? job.result : res[task]
  React.useEffect(() => { if (job.status === 'done') setRes(s => ({ ...s, [task]: job.result })) /* eslint-disable-next-line */ }, [job.status])
  return (
    <div className="h-full overflow-auto p-4 space-y-4">
      <div className="flex items-center gap-3 flex-wrap"><h2 className="text-lg font-semibold">Machine Learning Lab</h2><Badge kind="ml">classical ML · leakage-safe pipelines</Badge></div>
      <NeedData status={status} />
      <Tabs tabs={TABS} value={task} onChange={setTask} />
      <div className="flex items-center gap-3"><label className="text-xs text-slate-400">seed <input className="inp w-20 ml-1" type="number" value={seed} onChange={e => setSeed(+e.target.value)} /></label>
        <button className="btn btn-primary" disabled={job.status === 'running'} onClick={go}>Run {TABS.find(t => t.id === task).label}</button>
        {job.status === 'running' && <Spinner text="Running (grid search can take a while)" />}</div>
      {job.error && <Notice>{job.error}</Notice>}
      {r?.data_note && <Notice kind={r.mode === 'history' ? 'observed' : 'derived'}><b>{r.mode === 'history' ? 'Real history targets.' : 'Single-snapshot mode.'}</b> {r.data_note}</Notice>}
      {r && <Body task={task} r={r} />}
      {!r && job.status !== 'running' && <div className="text-xs text-slate-500">Nothing run yet for this task.</div>}
    </div>
  )
}
