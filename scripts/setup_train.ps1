# scripts/setup_train.ps1
# Run from project root: .\scripts\setup_train.ps1
# Installs the training stack (~3 GB: PyTorch + HF TRL).

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $scriptDir "Find-Python311.ps1")

Write-Host "Setting up training venv (this installs PyTorch ~3GB)..." -ForegroundColor Cyan

$chosen = Get-Python311

if (-not $chosen) {
    Write-Host ""
    Write-Host "Python 3.11 not found. Parlay needs 3.11." -ForegroundColor Red
    $diag = Get-Python311Diagnostics
    if ($diag.Count -gt 0) {
        Write-Host "These interpreters were found but are not 3.11:" -ForegroundColor Yellow
        foreach ($line in $diag) { Write-Host "  - $line" -ForegroundColor Gray }
    } else {
        Write-Host "No Python was found in PATH." -ForegroundColor Yellow
    }
    Write-Host "  winget install Python.Python.3.11" -ForegroundColor Cyan
    exit 1
}

Write-Host "  Found Python 3.11: $($chosen.Name)" -ForegroundColor Green
Write-Host "Using: $($chosen.Exe) $($chosen.PreArgs -join ' ')" -ForegroundColor Cyan

Write-Host "Creating training venv..." -ForegroundColor Yellow
& $chosen.Exe @($chosen.PreArgs + @("-m", "venv", "venv-train"))
if (-not (Test-Path ".\venv-train\Scripts\python.exe")) { throw "venv-train was not created" }

.\venv-train\Scripts\pip install --upgrade pip --quiet
.\venv-train\Scripts\pip install -r requirements.txt --quiet
.\venv-train\Scripts\pip install -r requirements-train.txt

Write-Host ""
Write-Host "Training venv ready." -ForegroundColor Green
Write-Host "Generate data:  .\scripts\train_data.ps1"
Write-Host "SFT warmup:     .\scripts\train_sft.ps1"
Write-Host "GRPO training:  .\scripts\train_grpo.ps1"
