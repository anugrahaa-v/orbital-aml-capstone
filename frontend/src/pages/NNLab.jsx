import { useState, useMemo } from 'react'
import { post, get } from '../api.js'
import { Panel, Notice, Badge, useAsync } from '../components/ui.jsx'
import Plot from '../components/Plot.jsx'

const sig = (x) => 1 / (1 + Math.exp(-x))
export const ACT = {
  binary_step: (x) => (x >= 0 ? 1 : 0), linear: (x) => x, sigmoid: sig, bipolar_sigmoid: (x) => 2 * sig(x) - 1,
  tanh: Math.tanh, relu: (x) => Math.max(0, x), leaky_relu: (x) => (x > 0 ? x : 0.01 * x),
}
const NAMES = [...Object.keys(ACT), 'softmax']

function Neuron() {
  const [x, setX] = useState([0.5, -1.0, 2.0]); const [w, setW] = useState([0.8, 0.4, -0.3]); const [b, setB] = useState(0.2); const [act, setAct] = useState('sigmoid')
  const [chk, setChk] = useState(null)
  const terms = x.map((xi, i) => xi * w[i]); const net = terms.reduce((a, c) => a + c, 0) + b
  const out = act === 'softmax' ? 1 / (1 + Math.exp(-net)) : ACT[act](net)
  const curve = useAsync(() => get(`/nn/curve/${act}`), [act])
  const upd = (arr, set, i, v) => set(arr.map((a, j) => (j === i ? v : a)))
  return (
    <div className="grid xl:grid-cols-2 gap-4">
      <Panel title="Single neuron" badge="derived">
        <div className="space-y-2 text-[13px]">
          {x.map((xi, i) => <div key={i} className="grid grid-cols-[28px_1fr_1fr] gap-2 items-center"><span className="label">{`x${i + 1}`}</span>
            <label>x <input className="inp w-20" type="number" step="0.1" value={xi} onChange={(e) => upd(x, setX, i, +e.target.value)} /></label>
            <label>w <input className="inp w-20" type="number" step="0.1" value={w[i]} onChange={(e) => upd(w, setW, i, +e.target.value)} /></label></div>)}
          <label className="block">bias b <input className="inp w-24" type="number" step="0.1" value={b} onChange={(e) => setB(+e.target.value)} /></label>
          <label className="block">activation <select className="inp" value={act} onChange={(e) => setAct(e.target.value)}>{NAMES.map((n) => <option key={n}>{n}</option>)}</select></label>
        </div>
        <div className="mt-4 p-3 rounded-lg bg-black/30 text-[13px] font-mono leading-relaxed">
          {terms.map((t, i) => <div key={i}>w{i + 1}·x{i + 1} = {w[i]} × {x[i]} = {t.toFixed(4)}</div>)}
          <div>net = Σ + b = {net.toFixed(4)}</div><div className="text-sky-300">output = {act}(net) = {out.toFixed(5)}</div>
          {act === 'softmax' && <div className="text-slate-400">(softmax shown over [net, 0], i.e. the 2-class logistic view)</div>}
        </div>
        <div className="mt-3 flex gap-2 items-center"><button className="btn-ghost" onClick={async () => { try { setChk(await post('/nn/neuron', { inputs: x, weights: w, bias: b, activation: act === 'softmax' ? 'sigmoid' : act })) } catch (e) { setChk({ error: e.message }) } }}>Verify with Python backend</button>
          {chk && (chk.error ? <span className="text-rose-300 text-xs">{chk.error}</span> : <span className="text-xs text-emerald-300">backend output {chk.output.toFixed(5)} {Math.abs(chk.output - out) < 1e-9 ? '✓ matches' : '(softmax view differs)'}</span>)}</div>
      </Panel>
      <Panel title={`${act} activation`} badge="derived">
        {curve.data && <Plot height={320} data={[{ x: curve.data.x, y: curve.data.y, mode: 'lines', line: { color: '#38bdf8' } }, { x: [net], y: [out], mode: 'markers', marker: { size: 10, color: '#fde047' }, name: 'current' }]} layout={{ xaxis: { title: 'net input' }, yaxis: { title: 'output' } }} />}
        <div className="text-xs text-slate-400">The yellow dot is your neuron's current operating point.</div>
      </Panel>
    </div>
  )
}

function Net({ snap, sizes, acts }) {
  const W = snap.W; const xs = sizes.map((_, i) => 60 + i * (260 / (sizes.length - 1))); const H = 220
  const pos = sizes.map((n, l) => Array.from({ length: n }, (_, k) => [xs[l], (H / (n + 1)) * (k + 1)]))
  return <svg viewBox="0 0 380 240" className="w-full max-w-lg">
    {W.map((m, l) => m.flatMap((row, i) => row.map((w, j) => <line key={`${l}-${i}-${j}`} x1={pos[l][i][0]} y1={pos[l][i][1]} x2={pos[l + 1][j][0]} y2={pos[l + 1][j][1]} stroke={w >= 0 ? '#38bdf8' : '#fb7185'} strokeWidth={Math.min(4, Math.abs(w) * 0.9 + 0.2)} opacity="0.7" />))) }
    {pos.flatMap((layer, l) => layer.map(([x, y], k) => <circle key={`${l}-${k}`} cx={x} cy={y} r="9" fill="#0b1220" stroke="#93c5fd" />))}
  </svg>
}

function Trainer() {
  const [cfg, setCfg] = useState({ dataset: 'xor', hidden: 4, activation: 'tanh', lr: 0.5, epochs: 1500 })
  const [r, setR] = useState(null); const [err, setErr] = useState(null); const [k, setK] = useState(2)
  async function train() { setErr(null); try { setR(await post('/nn/train', { ...cfg, hidden: [cfg.hidden] })) } catch (e) { setErr(e.message) } }
  return <div className="grid xl:grid-cols-2 gap-4">
    <Panel title="Train a from-scratch NumPy MLP" badge="derived">
      <div className="grid grid-cols-2 gap-3 text-[13px]">
        <label><div className="label">Dataset</div><select className="inp w-full" value={cfg.dataset} onChange={(e) => setCfg({ ...cfg, dataset: e.target.value })}><option>xor</option><option>and</option><option>or</option></select></label>
        <label><div className="label">Hidden units</div><input className="inp w-full" type="number" min="1" max="12" value={cfg.hidden} onChange={(e) => setCfg({ ...cfg, hidden: +e.target.value })} /></label>
        <label><div className="label">Hidden activation</div><select className="inp w-full" value={cfg.activation} onChange={(e) => setCfg({ ...cfg, activation: e.target.value })}>{['tanh', 'relu', 'sigmoid', 'bipolar_sigmoid'].map((a) => <option key={a}>{a}</option>)}</select></label>
        <label><div className="label">Learning rate</div><input className="inp w-full" type="number" step="0.05" value={cfg.lr} onChange={(e) => setCfg({ ...cfg, lr: +e.target.value })} /></label>
      </div>
      <button className="btn mt-3" onClick={train}>Train (forward → loss → backprop → update)</button>{err && <div className="mt-2"><Notice kind="error">{err}</Notice></div>}
      {r && <div className="mt-3 text-[13px]">Accuracy <b>{(r.accuracy * 100).toFixed(0)}%</b> · final loss {r.loss[r.loss.length - 1].toFixed(4)}<div className="text-xs text-slate-400 mt-1">Outputs for inputs 00, 01, 10, 11: {r.predictions.map((p) => p.toFixed(3)).join(', ')}</div></div>}
    </Panel>
    <Panel title="Weights and loss" badge="derived">
      {r ? <><Plot height={200} data={[{ y: r.loss, mode: 'lines' }]} layout={{ yaxis: { title: 'cross-entropy' }, xaxis: { title: 'epoch' } }} />
        <label className="text-xs text-slate-400">Weights at epoch {r.snapshots[k].epoch}<input type="range" className="w-full" min="0" max={r.snapshots.length - 1} value={k} onChange={(e) => setK(+e.target.value)} /></label>
        <Net snap={r.snapshots[k]} sizes={[2, cfg.hidden, 1]} /><div className="text-xs text-slate-400">Blue = positive weight, red = negative; thickness = magnitude.</div></>
        : <div className="text-sm text-slate-400">Train to see the loss curve and how weights evolve.</div>}
    </Panel>
  </div>
}

export default function NNLab() {
  const [tab, setTab] = useState('neuron')
  return <div className="p-4 space-y-4">
    <div className="flex gap-2">{[['neuron', 'Neuron and activations'], ['train', 'Backprop trainer']].map(([k, l]) => <button key={k} onClick={() => setTab(k)} className={`px-3 py-1.5 rounded-lg text-sm ${tab === k ? 'bg-blue-600/30 text-sky-200' : 'text-slate-400 hover:bg-white/5'}`}>{l}</button>)}</div>
    {tab === 'neuron' ? <Neuron /> : <Trainer />}
  </div>
}
