$EnvPath = Join-Path $PSScriptRoot "hf\Scripts\Activate.ps1"
if (Test-Path $EnvPath) {
    Write-Host "Activating hf environment..." -ForegroundColor Green
    & $EnvPath
} else {
    Write-Error "Could not find hf environment at $EnvPath"
}
