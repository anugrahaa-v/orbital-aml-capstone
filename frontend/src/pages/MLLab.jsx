import { useState } from 'react'
import { post } from '../api.js'
import { Panel, Notice, Badge, useJobRunner, Progress } from '../components/ui.jsx'
import Plot, { CLASS_COLORS } from '../components/Plot.jsx'

const TABS = [['regression', 'Regression'], ['classification', 'Classification'], ['feature_selection', 'Feature selection'], ['dimred', 'PCA / LDA / ICA'], ['kmeans', 'K-Means'], ['hierarchical', 'Hierarchical']]
const num = (x, d = 3) => (x == null ? '-' : Number(x).toFixed(d))

function Regression({ r }) {
  const ok = r.models.filter((m) => m.test)
  return <>
    <Panel title="Model comparison (held-out final 20% chronologically)" badge="ml">
      <table className="t"><thead><tr><th>Model</th><th>Family</th><th>CV MAE</th><th>MAE</th><th>RMSE</th><th>MSE</th><th>R²</th><th>Best params</th><th>s</th></tr></thead><tbody>
        {r.models.map((m) => m.test ? <tr key={m.model}><td>{m.model}</td><td>{m.family}</td><td>{num(m.cv_mae)}</td><td>{num(m.test.mae)}</td><td>{num(m.test.rmse)}</td><td>{num(m.test.mse)}</td><td>{num(m.test.r2)}</td><td className="text-slate-400">{JSON.stringify(m.best_params)}</td><td>{m.fit_seconds}</td></tr> : <tr key={m.model}><td>{m.model}</td><td colSpan="8" className="text-rose-300">{m.error}</td></tr>)}
        <tr className="text-slate-400"><td>baseline (train mean)</td><td /><td /><td>{num(r.baseline_mean.mae)}</td><td>{num(r.baseline_mean.rmse)}</td><td>{num(r.baseline_mean.mse)}</td><td>{num(r.baseline_mean.r2)}</td><td /><td /></tr></tbody></table>
      <div className="text-xs text-slate-400 mt-2">{r.split}. Train rows {r.n_train}, test rows {r.n_test}. Scalers (features and target) are fit inside each CV fold.{!r.xgboost_available && ' XGBoost not installed.'}</div>
    </Panel>
    <div className="grid xl:grid-cols-2 gap-4">
      <Panel title="RMSE by model" badge="ml"><Plot data={[{ type: 'bar', x: ok.map((m) => m.model), y: ok.map((m) => m.test.rmse), marker: { color: ok.map((m) => (m.family === 'linear' ? '#38bdf8' : '#a78bfa')) } }]} layout={{ yaxis: { title: 'RMSE' } }} /></Panel>
      <Panel title="Actual vs predicted (best model)" badge="ml">{ok[0] && <Plot data={[{ x: ok[0].actual_vs_pred.y_true, y: ok[0].actual_vs_pred.y_pred, mode: 'markers', marker: { size: 4, opacity: 0.6 } }, { x: [Math.min(...ok[0].actual_vs_pred.y_true), Math.max(...ok[0].actual_vs_pred.y_true)], y: [Math.min(...ok[0].actual_vs_pred.y_true), Math.max(...ok[0].actual_vs_pred.y_true)], mode: 'lines', line: { dash: 'dot' } }]} layout={{ title: ok[0].model, xaxis: { title: 'actual' }, yaxis: { title: 'predicted' }, showlegend: false }} />}</Panel>
      <Panel title="Residuals (best model)" badge="ml">{ok[0] && <Plot data={[{ type: 'histogram', x: ok[0].residuals }]} layout={{ xaxis: { title: 'actual − predicted' } }} />}</Panel>
    </div></>
}
function Classification({ r }) {
  return <>
    <Panel title="Classifiers: DEBRIS vs PAYLOAD (chronological test split)" badge="ml"><Notice kind="warn">{r.label_note}. Strong scores show catalogued fragmentation groups are separable by orbit and drag features, not that true object type is detected.</Notice>
      <table className="t mt-3"><thead><tr><th>Model</th><th>Acc</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th></tr></thead><tbody>{r.models.map((m) => <tr key={m.model}><td>{m.model}</td><td>{num(m.accuracy)}</td><td>{num(m.precision)}</td><td>{num(m.recall)}</td><td>{num(m.f1)}</td><td>{num(m.roc_auc)}</td></tr>)}</tbody></table></Panel>
    <div className="grid xl:grid-cols-2 gap-4">
      <Panel title="ROC curves" badge="ml"><Plot data={[...r.models.map((m) => ({ x: m.roc.fpr, y: m.roc.tpr, mode: 'lines', name: `${m.model} (${num(m.roc_auc, 2)})` })), { x: [0, 1], y: [0, 1], mode: 'lines', line: { dash: 'dot', color: '#64748b' }, showlegend: false }]} layout={{ xaxis: { title: 'FPR' }, yaxis: { title: 'TPR' } }} /></Panel>
      <Panel title={`Confusion matrix (${r.models[0].model})`} badge="ml"><Plot data={[{ type: 'heatmap', z: r.models[0].confusion_matrix, x: r.confusion_labels.map((l) => `pred ${l}`), y: r.confusion_labels.map((l) => `true ${l}`), colorscale: 'Blues', text: r.models[0].confusion_matrix, texttemplate: '%{text}' }]} layout={{ yaxis: { autorange: 'reversed' } }} /></Panel>
    </div></>
}
function FeatureSel({ r }) {
  return <div className="grid xl:grid-cols-2 gap-4">
    <Panel title={`Selection methods for target "${r.target}"`} badge="derived"><div className="scroll"><table className="t"><thead><tr><th>Feature</th><th>Votes</th><th>corr</th><th>F</th><th>MI</th><th>RFE</th><th>RF imp.</th><th>Lasso</th></tr></thead><tbody>
      {r.table.map((t) => <tr key={t.feature}><td>{t.feature}</td><td><b>{t.votes}</b>/5</td><td>{num(t.corr_with_target, 2)}</td><td>{num(t.f_score, 1)}</td><td>{num(t.mutual_info)}</td><td>{t.rfe_selected ? '✓' : t.rfe_rank}</td><td>{num(t.rf_importance)}</td><td>{num(t.lasso_coef)}</td></tr>)}</tbody></table></div>
      <div className="text-xs text-slate-400 mt-2">Fit on the chronological training segment only ({r.n_fit_rows} rows).</div></Panel>
    <Panel title="Correlation matrix" badge="derived"><Plot height={420} data={[{ type: 'heatmap', z: r.correlation_matrix.values, x: r.correlation_matrix.labels, y: r.correlation_matrix.labels, zmin: -1, zmax: 1, colorscale: 'RdBu' }]} layout={{ yaxis: { autorange: 'reversed' }, margin: { l: 110, b: 100, t: 10, r: 10 } }} /></Panel>
  </div>
}
function DimRed({ r }) {
  return <>{r.lda_note && <Notice>{r.lda_note}</Notice>}
    <div className="grid xl:grid-cols-3 gap-4">{Object.entries(r.methods).map(([k, m]) => <Panel key={k} title={k} badge="ml">
      <Plot height={280} data={r.classes.map((c) => ({ type: 'scattergl', mode: 'markers', name: c, marker: { size: 4, color: CLASS_COLORS[c] }, x: m.projection.x.filter((_, i) => m.projection.label[i] === c), y: m.projection.y.filter((_, i) => m.projection.label[i] === c) }))} />
      <div className="text-xs text-slate-300">kNN accuracy {num(m.knn_accuracy)} · macro-F1 {num(m.knn_macro_f1)}{m.explained_variance_ratio && ` · explained var ${m.explained_variance_ratio.map((v) => num(v, 2)).join(', ')}`}</div></Panel>)}</div></>
}
function KMeans({ r }) {
  const sw = r.sweep
  return <div className="grid xl:grid-cols-2 gap-4">
    <Panel title="Elbow (inertia) and silhouette" badge="ml"><Plot data={[{ x: sw.map((s) => s.k), y: sw.map((s) => s.inertia), name: 'inertia', mode: 'lines+markers' }, { x: sw.map((s) => s.k), y: sw.map((s) => s.silhouette), name: 'silhouette', yaxis: 'y2', mode: 'lines+markers' }]} layout={{ xaxis: { title: 'K' }, yaxis: { title: 'inertia' }, yaxis2: { overlaying: 'y', side: 'right', title: 'silhouette' } }} />
      <div className="text-sm">Optimal K by silhouette: <b>{r.optimal_k_silhouette}</b> · elbow K: <b>{r.elbow_k}</b> · DB {num(r.metrics_at_optimum.davies_bouldin)} · CH {num(r.metrics_at_optimum.calinski_harabasz, 0)}</div></Panel>
    <Panel title="Clusters in PCA space (3-D)" badge="ml"><Plot height={380} data={[{ type: 'scatter3d', mode: 'markers', x: r.points.x, y: r.points.y, z: r.points.z, text: r.points.name, marker: { size: 2.5, color: r.points.cluster, colorscale: 'Turbo' } }]} layout={{ margin: { l: 0, r: 0, t: 0, b: 0 } }} /></Panel>
    <Panel title="Cluster composition by class (interpretation only)" badge="derived"><pre className="text-xs text-slate-300 overflow-auto">{JSON.stringify(r.cluster_vs_class, null, 1)}</pre></Panel></div>
}
function Hier({ r }) {
  const d = r.dendrogram
  return <div className="grid xl:grid-cols-2 gap-4">
    <Panel title="Dendrogram (Ward, truncated)" badge="ml"><Plot height={360} data={d.icoord.map((x, i) => ({ x: d.dcoord[i], y: x, mode: 'lines', line: { color: '#7dd3fc', width: 1 }, hoverinfo: 'none', showlegend: false }))} layout={{ xaxis: { title: 'distance' }, yaxis: { showticklabels: false } }} /></Panel>
    <Panel title={`Linkage comparison (k = ${r.n_clusters})`} badge="ml"><table className="t"><thead><tr><th>Linkage</th><th>Silhouette</th><th>Davies-Bouldin</th><th>Calinski-Harabasz</th><th>Sizes</th></tr></thead><tbody>
      {Object.entries(r.linkages).map(([k, m]) => <tr key={k} className={k === r.best_linkage_by_silhouette ? 'text-emerald-300' : ''}><td>{k}</td><td>{num(m.silhouette)}</td><td>{num(m.davies_bouldin)}</td><td>{num(m.calinski_harabasz, 0)}</td><td>{m.cluster_sizes.join(', ')}</td></tr>)}</tbody></table></Panel></div>
}

export default function MLLab() {
  const [tab, setTab] = useState('regression'); const [state, run] = useJobRunner(); const [res, setRes] = useState({})
  const go = async () => { const r = await run(() => post('/ml/run', { task: tab })); if (r) setRes((x) => ({ ...x, [tab]: r })) }
  const r = res[tab]
  return (
    <div className="p-4 space-y-4">
      <div className="flex flex-wrap gap-2 items-center">
        {TABS.map(([k, l]) => <button key={k} onClick={() => setTab(k)} className={`px-3 py-1.5 rounded-lg text-sm ${tab === k ? 'bg-blue-600/30 text-sky-200' : 'text-slate-400 hover:bg-white/5'}`}>{l}</button>)}
        <button className="btn ml-auto" disabled={state.running} onClick={go}>{state.running ? 'Running…' : `Run ${TABS.find((t) => t[0] === tab)[1]}`}</button>
      </div>
      <Progress s={state} />
      {state.error && <Notice kind="error">{state.error}</Notice>}
      {!r && !state.running && <Panel><div className="text-sm text-slate-400">Runs on the ingested CelesTrak catalogue. Every split is chronological by element-set epoch; scalers are fit on training data only; seeds are fixed (42) and each run is logged for the Analytics page.</div></Panel>}
      {r?.error && <Notice kind="warn">{r.error}</Notice>}
      {r && !r.error && <>
        {(tab === 'regression' || tab === 'feature_selection') && <Notice kind={r.mode === 'history' ? 'info' : 'warn'}><b>{r.mode === 'history' ? 'Chronological history targets' : 'Cross-sectional targets'}.</b> {r.data_note} Target: <code>{r.target || r.targets_available?.[0]}</code> ({r.data_label}).</Notice>}
        {tab === 'regression' && <Regression r={r} />}{tab === 'classification' && <Classification r={r} />}{tab === 'feature_selection' && <FeatureSel r={r} />}
        {tab === 'dimred' && <DimRed r={r} />}{tab === 'kmeans' && <KMeans r={r} />}{tab === 'hierarchical' && <Hier r={r} />}
      </>}
    </div>
  )
}
