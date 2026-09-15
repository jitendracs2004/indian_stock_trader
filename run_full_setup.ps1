# =====================================================================
# One-time full setup for the Indian Stock Trader project.
#
# Runs every step needed to populate the database and open the
# dashboard with live predictions:
#   1. Ingest the full Nifty 500 OHLCV history into PostgreSQL
#   2. Train weekly + monthly models and write signals
#   3. Launch the Streamlit dashboard
#
# Use this the FIRST time you set up the project. For recurring
# end-of-day updates, use run_daily_pipeline.ps1 instead (it does an
# incremental refresh rather than a full re-download).
#
# Usage:
#   .\run_full_setup.ps1                 # full Nifty 500 (5y history)
#   .\run_full_setup.ps1 -Limit 20       # quick test with 20 symbols
#   .\run_full_setup.ps1 -Period 2y      # shorter history
#   .\run_full_setup.ps1 -SkipDashboard  # run steps 1-2 only
# =====================================================================

param(
    [int]    $Limit         = 0,      # 0 = all Nifty 500 symbols
    [string] $Period        = "5y",   # 1y | 2y | 5y | max
    [switch] $SkipDashboard          # stop after predictions
)

$ErrorActionPreference = "Stop"

# Resolve project directory to this script's location so it works
# regardless of the current shell directory.
$ProjectDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDirectory

$LogDirectory = Join-Path $ProjectDirectory "logs"
if (-not (Test-Path $LogDirectory)) {
    New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
}

$Timestamp  = Get-Date -Format "yyyyMMdd_HHmmss"
$SetupLog   = Join-Path $LogDirectory "full_setup_$Timestamp.log"

function Write-Step {
    param([string] $Message)
    Write-Host ""
    Write-Host ("=" * 70)
    Write-Host $Message
    Write-Host ("=" * 70)
}

# ---------------------------------------------------------------------
# Step 1 — Ingest OHLCV data into PostgreSQL
# ---------------------------------------------------------------------
Write-Step "STEP 1/3  Ingesting Nifty 500 OHLCV into PostgreSQL"

$IngestArgs = @("data_ingestion.py", "--period", $Period)
if ($Limit -gt 0) {
    $IngestArgs += @("--limit", "$Limit")
}

python @IngestArgs 2>&1 | Tee-Object -FilePath $SetupLog
if ($LASTEXITCODE -ne 0) {
    Write-Error "Data ingestion failed (exit $LASTEXITCODE). See $SetupLog"
    exit $LASTEXITCODE
}

# ---------------------------------------------------------------------
# Step 2 — Train models and generate predictions
# ---------------------------------------------------------------------
Write-Step "STEP 2/3  Training models and generating predictions"

python predict_weekly_monthly.py 2>&1 | Tee-Object -FilePath $SetupLog -Append
if ($LASTEXITCODE -ne 0) {
    Write-Error "Prediction step failed (exit $LASTEXITCODE). See $SetupLog"
    exit $LASTEXITCODE
}

# ---------------------------------------------------------------------
# Step 3 — Launch the dashboard
# ---------------------------------------------------------------------
if ($SkipDashboard) {
    Write-Step "DONE  Data is in PostgreSQL. Dashboard launch skipped."
    Write-Host "Start it manually with:"
    Write-Host "  python -m streamlit run streamlit_dashboard.py"
    exit 0
}

Write-Step "STEP 3/3  Launching the Streamlit dashboard"
Write-Host "The dashboard opens in your browser. Press Ctrl+C to stop it."

python -m streamlit run streamlit_dashboard.py
