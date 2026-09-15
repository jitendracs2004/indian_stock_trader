# Run the Indian stock daily pipeline from Windows Task Scheduler.

$ProjectDirectory = "C:\Users\elnpqtw\OneDrive - Ericsson\Documents\Jit\Personal\Learning\indian_stock_trader"

# Replace this with the exact output of:
# python -c "import sys; print(sys.executable)"
$PythonExecutable = "C:\Users\elnpqtw\AppData\Local\Programs\Python\Python312\python.exe"

$LogDirectory = Join-Path $ProjectDirectory "logs"

if (-not (Test-Path $LogDirectory)) {
    New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$WrapperLog = Join-Path $LogDirectory "task_scheduler_$Timestamp.log"

Set-Location $ProjectDirectory

& $PythonExecutable "$ProjectDirectory\daily_pipeline.py" *>&1 |
    Tee-Object -FilePath $WrapperLog

exit $LASTEXITCODE