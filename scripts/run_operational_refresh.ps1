$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$PythonBin = if ($env:PYTHON) { $env:PYTHON } else { "python" }

$RefreshName = "simple_loss_daysweeks_v2_operational_refresh_v1"
$ReportDir = Join-Path $ProjectRoot "reports/backtests/temporal_multires/$RefreshName"
$LatestSummary = Join-Path $ReportDir "latest_case_summary.md"
$FirstRead = Join-Path $ReportDir "summary.md"

Push-Location $ProjectRoot
try {
    Write-Host "Running locked operational refresh..."
    & $PythonBin "run_temporal_operational_refresh_v1.py" "--project-root" $ProjectRoot
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Latest case summary:"
if (Test-Path $LatestSummary) {
    Get-Content $LatestSummary -TotalCount 80
}
else {
    Write-Error "Missing expected latest summary: $LatestSummary"
    exit 1
}

Write-Host ""
Write-Host "Refresh bundle: $ReportDir"
Write-Host "Read first: $FirstRead"
