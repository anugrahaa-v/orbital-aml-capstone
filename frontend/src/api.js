// Thin API client. Errors surface the backend's actionable message (e.g. "no catalogue ingested").
async function req(path, opts) {
  const r = await fetch(path, opts)
  const body = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail || body))
  return body
}
const qs = (o) => Object.entries(o || {}).filter(([, v]) => v !== undefined && v !== null && v !== '').map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&')
export const get = (p, params) => req(`/api${p}${params ? '?' + qs(params) : ''}`)
export const post = (p, body) => req(`/api${p}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) })

export async function runJob(start, onProgress) {
  const { job } = await start()
  for (;;) {
    const j = await get(`/jobs/${job}`)
    onProgress && onProgress(j)
    if (j.status === 'done') return j.result
    if (j.status === 'error') throw new Error(j.error)
    await new Promise((r) => setTimeout(r, 700))
  }
}

export function decodePositions(p) {
  const bin = atob(p.data_b64); const buf = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i)
  return new Float32Array(buf.buffer) // (frames, objects, 3) TEME km
}
