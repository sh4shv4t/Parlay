# scripts/setup.ps1
# Run from project root: .\scripts\setup.ps1
# Requires Python 3.11 on PATH and PowerShell 5+

$ErrorActionPreference = "Stop"

Write-Host "Setting up Parlay..." -ForegroundColor Cyan

# Check Python 3.11
try {
    $pyver = & python --version 2>&1
    if ($pyver -notmatch "3\.11") {
        Write-Host "Python 3.11 required. Found: $pyver" -ForegroundColor Red
        Write-Host "Download from https://www.python.org/downloads/release/python-3110/"
        exit 1
    }
} catch {
    Write-Host "Python not found on PATH." -ForegroundColor Red
    exit 1
}

# Game venv
Write-Host "Creating game venv..." -ForegroundColor Yellow
python -m venv venv
.\venv\Scripts\pip install --upgrade pip --quiet
.\venv\Scripts\pip install -r requirements.txt --quiet

# .env
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example" -ForegroundColor Green
}

# DB init
.\venv\Scripts\python scripts\init_db.py

Write-Host ""
Write-Host "Game venv ready." -ForegroundColor Green
Write-Host "To start the server: .\scripts\run.ps1"
Write-Host "For training stack: .\scripts\setup_train.ps1"
