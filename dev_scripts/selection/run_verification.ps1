# Run Factor DB (Two-Speed Structure Generation)
# Using 2023 start year to ensure structural recalibration happens (assuming data availability)
# Limiting to 10 days for quick verification
Write-Host "Running Factor DB Generation..."
$env:PYTHONPATH="d:\2025-2026A\Thesis-MAS Hedge Fund\projects\hf with antigravity\mas-hedge-fund"
& "d:\2025-2026A\Thesis-MAS Hedge Fund\projects\hf with antigravity\mas-hedge-fund\hf\Scripts\python.exe" src/selection/factor_db.py --start_year 2023 --limit 10

if ($LASTEXITCODE -eq 0) {
    Write-Host "Factor DB Generation Complete."
    
    # Run Selection output verification
    Write-Host "Verifying Selection Pipeline Output..."
    & "d:\2025-2026A\Thesis-MAS Hedge Fund\projects\hf with antigravity\mas-hedge-fund\hf\Scripts\python.exe" verify_selection_output.py
} else {
    Write-Host "Factor DB Generation Failed."
}
