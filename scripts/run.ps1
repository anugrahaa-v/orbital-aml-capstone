# Downloads data on first run (cache-first, fair-use safe), then starts the app at http://localhost:8000
param([int]$Port = 8000)
$py = ".\.venv\Scripts\python.exe"
if (-not (Get-ChildItem data\cache -Filter "gp_*.json" -ErrorAction SilentlyContinue)) {
  & $py -m backend.ingest_cli --groups active cosmos-2251-debris iridium-33-debris fengyun-1c-debris cosmos-1408-debris
}
& $py -m uvicorn backend.main:app --port $Port
