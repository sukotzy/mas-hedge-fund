$EnvPath = Join-Path $PSScriptRoot "hf2\Scripts\Activate.ps1"
if (Test-Path $EnvPath) {
    Write-Host "Activating hf2 environment..." -ForegroundColor Green
    & $EnvPath
} else {
    Write-Error "Could not find hf2 environment at $EnvPath"
}
