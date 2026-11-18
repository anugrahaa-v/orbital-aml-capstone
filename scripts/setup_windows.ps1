# One-time setup on Windows PowerShell:  .\scripts\setup_windows.ps1   (add -WithTorch to also install PyTorch)
param([switch]$WithTorch)
$ErrorActionPreference = "Stop"
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($WithTorch) { .\.venv\Scripts\python.exe -m pip install torch xgboost } else { Write-Host "Skipping PyTorch/XGBoost (optional). Re-run with -WithTorch to install them." }
Write-Host "Done. Next: .\scripts\run.ps1"
