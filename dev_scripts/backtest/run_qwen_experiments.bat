@echo off
set USE_LOCAL_DATA=true

echo =========================================================================
echo Starting Qwen Experiment Period 1: 2020-02-20 to 2020-03-10
echo =========================================================================
python src/training/run_qwen_experiment.py --start-date "2020-02-20" --end-date "2020-03-10" --experiments "no_hint_standard,no_hint_wealth,with_hint_standard,with_hint_wealth" --lang "zh" --output-dir "data/results_qwen/2020_crash"

echo.
echo =========================================================================
echo Starting Qwen Experiment Period 2: 2022-01-03 to 2022-01-21
echo =========================================================================
python src/training/run_qwen_experiment.py --start-date "2022-01-03" --end-date "2022-01-21" --experiments "no_hint_standard,no_hint_wealth,with_hint_standard,with_hint_wealth" --lang "zh" --output-dir "data/results_qwen/2022_jan"

echo.
echo =========================================================================
echo Starting Qwen Experiment Period 3: 2023-03-01 to 2023-03-15
echo =========================================================================
python src/training/run_qwen_experiment.py --start-date "2023-03-01" --end-date "2023-03-15" --experiments "no_hint_standard,no_hint_wealth,with_hint_standard,with_hint_wealth" --lang "zh" --output-dir "data/results_qwen/2023_svb"

echo.
echo All Qwen experiments finished successfully!
pause
