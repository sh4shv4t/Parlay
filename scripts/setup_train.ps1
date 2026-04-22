$ErrorActionPreference = "Stop"
Write-Host "Setting up training venv (this installs PyTorch ~3GB)..." -ForegroundColor Cyan
python -m venv venv-train
.\venv-train\Scripts\pip install --upgrade pip --quiet
.\venv-train\Scripts\pip install -r requirements.txt --quiet
.\venv-train\Scripts\pip install -r requirements-train.txt
Write-Host "Training venv ready. Run: .\scripts\train_data.ps1" -ForegroundColor Green
