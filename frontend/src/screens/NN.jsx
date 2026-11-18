import React, { useEffect, useMemo, useState } from 'react'
import { post } from '../api.js'
import { Panel, Badge, Notice, Plot, Table, Field, fmt, COLORS, Spinner } from '../ui.jsx'

const sig = x => 1 / (1 + Math.exp(-x))
const ACT = {
  binary_step: x => (x >= 0 ? 1 : 0), linear: x => x, sigmoid: sig, bipolar_sigmoid: x => 2 * sig(x) - 1,
  tanh: Math.tanh, relu: x => Math.max(0, x), leaky_relu: (x, a = 0.01) => (x > 0 ? x : a * x), softmax: x => sig(x) /* 2-class view: softmax([x,0])[0] */,
}
const ACTS = Object.keys(ACT)
const FORMULA = {
  binary_step: 'f(x) = 1 if x ≥ 0 else 0', linear: 'f(x) = x', sigmoid: 'f(x) = 1 / (1 + e⁻ˣ)', bipolar_sigmoid: 'f(x) = 2 / (1 + e⁻ˣ) − 1', tanh: 'f(x) = tanh(x)',
  relu: 'f(x) = max(0, x)', leaky_relu: 'f(x) = x if x>0 else αx', softmax: 'softmax(z)ᵢ = e^zᵢ / Σ e^zⱼ (shown: 2-class [x,0])',
}

function NeuronDiagram({ x, w, b, net, out, act }) {
  const ys = x.map((_, i) => 40 + i * 70)
  return (
    <svg viewBox="0 0 520 260" className="w-full">
      <defs><style>{`@keyframes flow{to{stroke-dashoffset:-24}} .fl{stroke-dasharray:6 6;animation:flow 1s linear infinite}`}</style></defs>
      {x.map((xi, i) => <g key={i}>
        <line x1="70" y1={ys[i]} x2="290" y2="130" className="fl" stroke={w[i] >= 0 ? '#4cd6a0' : '#ff6b8a'} strokeWidth={0.8 + Math.min(4, Math.abs(w[i]) * 2)} opacity="0.8" />
        <circle cx="50" cy={ys[i]} r="20" fill="#16213d" stroke="#5b8dff" /><text x="50" y={ys[i] + 4} textAnchor="middle" fill="#e8eefc" fontSize="12">{fmt(xi, 2)}</text>
        <text x="150" y={(ys[i] + 130) / 2 - 6} fill="#9fb4ff" fontSize="11">w={fmt(w[i], 2)}</text>
        <text x="25" y={ys[i] - 26} fill="#667" fontSize="10">x{i + 1}</text>
      </g>)}
      <circle cx="310" cy="130" r="30" fill="#1b2750" stroke="#a37bff" strokeWidth="2" /><text x="310" y="126" textAnchor="middle" fill="#fff" fontSize="11">Σ + b</text><text x="310" y="140" textAnchor="middle" fill="#c8b6ff" fontSize="10">{fmt(net, 3)}</text>
      <text x="310" y="180" textAnchor="middle" fill="#9fb4ff" fontSize="10">b = {fmt(b, 2)}</text>
      <line x1="340" y1="130" x2="400" y2="130" className="fl" stroke="#ffd45b" strokeWidth="2" />
      <rect x="400" y="105" width="64" height="50" rx="10" fill="#241b3f" stroke="#ffd45b" /><text x="432" y="126" textAnchor="middle" fill="#ffd45b" fontSize="10">{act}</text><text x="432" y="144" textAnchor="middle" fill="#fff" fontSize="12">{fmt(out, 4)}</text>
      <line x1="464" y1="130" x2="505" y2="130" stroke="#ffd45b" markerEnd="url(#a)" /><text x="500" y="122" fill="#ffd45b" fontSize="11" textAnchor="end">y</text>
    </svg>)
}

function NetDiagram({ sizes, W, hot }) {
  const cols = sizes.length, X = i => 60 + i * (400 / (cols - 1)), Y = (i, j) => 30 + (j + 0.5) * (200 / sizes[i])
  return <svg viewBox="0 0 520 260" className="w-full">
    {W.map((m, l) => m.map((row, i) => row.map((w, j) => <line key={`${l}-${i}-${j}`} x1={X(l)} y1={Y(l, i)} x2={X(l + 1)} y2={Y(l + 1, j)}
      stroke={w >= 0 ? '#4cd6a0' : '#ff6b8a'} strokeWidth={0.4 + Math.min(4, Math.abs(w) * 0.9)} opacity={0.35 + (hot === l ? 0.5 : 0)} />)))}
    {sizes.map((n, l) => Array.from({ length: n }).map((_, j) => <circle key={`${l}-${j}`} cx={X(l)} cy={Y(l, j)} r="11" fill="#16213d" stroke={hot === l - 1 || hot === l ? '#ffd45b' : '#5b8dff'} />))}
    <text x="60" y="250" fill="#778" fontSize="10" textAnchor="middle">input</text><text x="460" y="250" fill="#778" fontSize="10" textAnchor="middle">output</text>
  </svg>
}

export default function NN() {
  const [x, setX] = useState([0.5, -1.0, 2.0]); const [w, setW] = useState([0.8, 0.4, -0.3]); const [b, setB] = useState(0.1); const [act, setAct] = useState('sigmoid')
  const net = useMemo(() => x.reduce((s, v, i) => s + v * w[i], b), [x, w, b])
  const out = ACT[act](net)
  const xs = useMemo(() => Array.from({ length: 241 }, (_, i) => -6 + i * 0.05), [])
  const [srv, setSrv] = useState(null)
  const verify = async () => { try { setSrv(await post('/api/nn/neuron', { inputs: x, weights: w, bias: b, activation: act })) } catch (e) { setSrv({ error: e.message }) } }
  useEffect(() => setSrv(null), [x, w, b, act])
  const setAt = (arr, set, i, v) => set(arr.map((a, k) => k === i ? v : a))

  const [cfg, setCfg] = useState({ dataset: 'xor', hidden: 4, activation: 'tanh', lr: 0.5, epochs: 1500 })
  const [tr, setTr] = useState(null); const [busy, setBusy] = useState(false); const [snap, setSnap] = useState(2); const [terr, setTerr] = useState(null)
  const train = async () => { setBusy(true); setTerr(null); try { setTr(await post('/api/nn/train', { ...cfg, hidden: [+cfg.hidden], lr: +cfg.lr, epochs: +cfg.epochs })); setSnap(2) } catch (e) { setTerr(e.message) } setBusy(false) }
  const sizes = [2, +cfg.hidden, 1]
  const W = tr ? tr.snapshots[snap].W : null

  return (
    <div className="h-full overflow-auto p-4 grid grid-cols-12 gap-4 content-start">
      <div className="col-span-12 flex items-center gap-3"><h2 className="text-lg font-semibold">Neural Network Playground</h2><Badge kind="ml">educational · pure NumPy backend</Badge></div>
      <Panel className="col-span-7" title="Single neuron: live calculation">
        <NeuronDiagram x={x} w={w} b={b} net={net} out={out} act={act} />
        <div className="grid grid-cols-4 gap-2 text-xs">
          {x.map((_, i) => <div key={i} className="space-y-1"><Field label={`x${i + 1}`}><input type="number" step="0.1" className="inp" value={x[i]} onChange={e => setAt(x, setX, i, +e.target.value)} /></Field>
            <Field label={`w${i + 1}`}><input type="number" step="0.1" className="inp" value={w[i]} onChange={e => setAt(w, setW, i, +e.target.value)} /></Field></div>)}
          <Field label="bias b"><input type="number" step="0.1" className="inp" value={b} onChange={e => setB(+e.target.value)} /></Field>
        </div>
        <div className="mt-3 flex items-center gap-3 flex-wrap">
          <select className="inp" value={act} onChange={e => setAct(e.target.value)}>{ACTS.map(a => <option key={a}>{a}</option>)}</select>
          <button className="btn" onClick={verify}>Verify with Python backend</button>
          {srv && !srv.error && <span className="text-xs text-emerald-300">backend: net={fmt(srv.net_input, 5)}, y={fmt(srv.output, 6)} {Math.abs(srv.output - out) < 1e-9 ? '✓ matches' : '✗ differs'}</span>}
          {srv?.error && <span className="text-xs text-rose-300">{srv.error}</span>}
        </div>
        <div className="mt-3 font-mono text-xs text-slate-300 bg-black/30 rounded-lg p-3 leading-6">
          net = {x.map((v, i) => `(${fmt(w[i], 2)}×${fmt(v, 2)})`).join(' + ')} + ${fmt(b, 2)}<br />
          net = {x.map((v, i) => fmt(v * w[i], 3)).join(' + ')} + {fmt(b, 2)} = <b>{fmt(net, 4)}</b><br />
          y = {act}(net) = <b>{fmt(out, 5)}</b><br /><span className="text-slate-500">{FORMULA[act]}</span></div>
      </Panel>
      <Panel className="col-span-5" title={`Activation: ${act}`}>
        <Plot height={300} data={[{ x: xs, y: xs.map(ACT[act]), mode: 'lines', line: { color: COLORS[0], width: 2 } }, { x: [net], y: [out], mode: 'markers', marker: { size: 10, color: '#ffd45b' }, name: 'current' }]} layout={{ xaxis: { title: 'net input' }, yaxis: { title: 'f(net)' }, showlegend: false }} />
        <div className="grid grid-cols-4 gap-1 mt-1">{ACTS.map(a => <button key={a} className={`btn text-[10px] ${a === act ? 'btn-primary' : ''}`} onClick={() => setAct(a)}>{a}</button>)}</div>
      </Panel>

      <Panel className="col-span-7" title="Train a small network (forward + backprop, gradient descent)">
        <div className="flex gap-3 items-end flex-wrap mb-3">
          <Field label="dataset"><select className="inp" value={cfg.dataset} onChange={e => setCfg({ ...cfg, dataset: e.target.value })}><option>xor</option><option>and</option><option>or</option></select></Field>
          <Field label="hidden units"><input className="inp w-20" type="number" min="1" max="12" value={cfg.hidden} onChange={e => setCfg({ ...cfg, hidden: e.target.value })} /></Field>
          <Field label="activation"><select className="inp" value={cfg.activation} onChange={e => setCfg({ ...cfg, activation: e.target.value })}>{['tanh', 'relu', 'sigmoid', 'bipolar_sigmoid'].map(a => <option key={a}>{a}</option>)}</select></Field>
          <Field label="learning rate"><input className="inp w-20" value={cfg.lr} onChange={e => setCfg({ ...cfg, lr: e.target.value })} /></Field>
          <Field label="epochs"><input className="inp w-24" type="number" value={cfg.epochs} onChange={e => setCfg({ ...cfg, epochs: e.target.value })} /></Field>
          <button className="btn btn-primary" onClick={train} disabled={busy}>Train</button>{busy && <Spinner />}
        </div>
        {terr && <Notice>{terr}</Notice>}
        {W ? <NetDiagram sizes={sizes} W={W} hot={snap} /> : <div className="text-xs text-slate-500">Train to see the weights evolve (green = positive, red = negative).</div>}
        {tr && <div className="flex items-center gap-2 text-xs mt-1">weights at: {tr.snapshots.map((s, i) => <button key={i} className={`btn ${i === snap ? 'btn-primary' : ''}`} onClick={() => setSnap(i)}>epoch {s.epoch}</button>)}</div>}
      </Panel>
      <Panel className="col-span-5" title="Training result">
        {tr ? <>
          <Plot height={200} data={[{ y: tr.loss, mode: 'lines', line: { color: COLORS[1] } }]} layout={{ yaxis: { title: 'cross-entropy', type: 'log' }, xaxis: { title: 'epoch' } }} />
          <Table columns={[{ key: 'x', label: 'input' }, { key: 'y', label: 'target' }, { key: 'p', label: 'prediction' }]}
            rows={[[0, 0], [0, 1], [1, 0], [1, 1]].map((v, i) => ({ x: v.join(', '), y: tr.labels[i], p: fmt(tr.predictions[i], 4) }))} />
          <div className="text-xs text-slate-400 mt-2">accuracy {fmt(tr.accuracy * 100, 0)}% · XOR needs a hidden layer; AND/OR do not.</div></> : <div className="text-xs text-slate-500">No run yet.</div>}
      </Panel>
    </div>
  )
}
