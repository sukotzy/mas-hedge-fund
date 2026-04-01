@echo off
echo =========================================================
echo Running pure baseline executions for Loss Aversion Analysis
echo Options: No Risk Manager, 0.0 Turnover Penalty, 0.0 Margin
echo =========================================================

python run_all_backtests.py --base-dirs data/allocator_reuslts_test --out-dir data/ablation_loss_aversion --fast --disable-risk-manager --turnover-penalty 0.0 --margin-requirement 0.0 --segregate-capital 0.0

echo.
echo All 24 baseline tests completed!
