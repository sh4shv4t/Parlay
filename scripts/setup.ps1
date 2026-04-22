# scripts/setup.ps1
# Run from project root: .\scripts\setup.ps1
# Robust Python 3.11 detection — see Find-Python311.ps1

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $scriptDir "Find-Python311.ps1")

Write-Host "Setting up Parlay..." -ForegroundColor Cyan

$chosen = Get-Python311

if (-not $chosen) {
    Write-Host ""
    Write-Host "Python 3.11 not found. Parlay needs 3.11 (see .cursorrules)." -ForegroundColor Red
    $diag = Get-Python311Diagnostics
    if ($diag.Count -gt 0) {
        Write-Host "These interpreters were found but are not 3.11:" -ForegroundColor Yellow
        foreach ($line in $diag) { Write-Host "  - $line" -ForegroundColor Gray }
    } else {
        Write-Host "No Python was found in PATH. Add Python to PATH or install 3.11." -ForegroundColor Yellow
    }
    Write-Host "Install 3.11, then re-run this script:" -ForegroundColor Yellow
    Write-Host "  winget install Python.Python.3.11" -ForegroundColor Cyan
    exit 1
}

Write-Host "  Found Python 3.11: $($chosen.Name)" -ForegroundColor Green
Write-Host "  Version line: $($chosen.Ver)" -ForegroundColor Gray
Write-Host "Using: $($chosen.Exe) $($chosen.PreArgs -join ' ')" -ForegroundColor Cyan

Write-Host "Creating game venv..." -ForegroundColor Yellow
& $chosen.Exe @($chosen.PreArgs + @("-m", "venv", "venv"))
if (-not (Test-Path ".\venv\Scripts\python.exe")) { throw "venv was not created (python -m venv failed)" }

.\venv\Scripts\pip install --upgrade pip --quiet
.\venv\Scripts\pip install -r requirements.txt --quiet

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example" -ForegroundColor Green
}

.\venv\Scripts\python scripts\init_db.py

Write-Host "Verifying installed packages..." -ForegroundColor Yellow
try {
    $checkCmd = "import fastapi, uvicorn, pydantic, aiosqlite; from google import genai; import fastmcp; print('All game deps OK')"
    & .\venv\Scripts\python -c $checkCmd
    Write-Host "  All game deps OK" -ForegroundColor Green
} catch {
    Write-Host "  Dependency check failed - re-installing:" -ForegroundColor Red
    Write-Host "  .\venv\Scripts\pip install -r requirements.txt --force-reinstall" -ForegroundColor Yellow
    .\venv\Scripts\pip install -r requirements.txt --force-reinstall --quiet
}

Write-Host ""
Write-Host "Game venv ready." -ForegroundColor Green
Write-Host "Run the server:         .\scripts\run.ps1"
Write-Host "Run without an API key: make run  (mock mode enabled automatically)"
Write-Host "Training stack:         .\scripts\setup_train.ps1"
