# Automation script to run experiment evaluations

Write-Host "Starting experiment evaluations with risk manager..." -ForegroundColor Cyan

$python = "..\hf\Scripts\python.exe"
$eval_script = "..\evaluate_experiments.py"

# 1. DeepSeek - 2020 Covid Crash
Write-Host "Running: DeepSeek 2020 Covid Crash..." -ForegroundColor Yellow
& $python $eval_script `
 --inputs ..\data\backtests_with_risk_manager\allocator_reuslts_test_results_deepseek_2020_crash_no_hint_standard.jsonl ..\data\backtests_with_risk_manager\allocator_reuslts_test_results_deepseek_2020_crash_no_hint_wealth.jsonl ..\data\backtests_with_risk_manager\allocator_reuslts_test_results_deepseek_2020_crash_with_hint_standard.jsonl ..\data\backtests_with_risk_manager\allocator_reuslts_test_results_deepseek_2020_crash_with_hint_wealth.jsonl `
 --labels "Standard+No Hint+RM" "Wealth+No Hint+RM" "Standard+Hint+RM" "Wealth+Hint+RM" `
 --title "DeepSeek Hedge Fund Strategy Comparison - 2020 Covid Crash" `
 --output model_strategy_selection\results\deepseek_4_settings_comparison_rm1.png

# 2. DeepSeek - 2022 Rate Hike
Write-Host "Running: DeepSeek 2022 Rate Hike..." -ForegroundColor Yellow
& $python $eval_script `
 --inputs ..\data\backtests_with_risk_manager\allocator_reuslts_test_results_deepseek_2022_jan_no_hint_standard.jsonl ..\data\backtests_with_risk_manager\allocator_reuslts_test_results_deepseek_2022_jan_no_hint_wealth.jsonl ..\data\backtests_with_risk_manager\allocator_reuslts_test_results_deepseek_2022_jan_with_hint_standard.jsonl ..\data\backtests_with_risk_manager\allocator_reuslts_test_results_deepseek_2022_jan_with_hint_wealth.jsonl `
 --labels "Standard+No Hint+RM" "Wealth+No Hint+RM" "Standard+Hint+RM" "Wealth+Hint+RM" `
 --title "DeepSeek Hedge Fund Strategy Comparison - 2022 Rate Hike" `
 --output model_strategy_selection\results\deepseek_4_settings_comparison_rm2.png

# 3. DeepSeek - 2023 SVB Collapse
Write-Host "Running: DeepSeek 2023 SVB Collapse..." -ForegroundColor Yellow
& $python $eval_script `
 --inputs ..\data\backtests_with_risk_manager\allocator_reuslts_test_results_deepseek_2023_svb_no_hint_standard.jsonl ..\data\backtests_with_risk_manager\allocator_reuslts_test_results_deepseek_2023_svb_no_hint_wealth.jsonl ..\data\backtests_with_risk_manager\allocator_reuslts_test_results_deepseek_2023_svb_with_hint_standard.jsonl ..\data\backtests_with_risk_manager\allocator_reuslts_test_results_deepseek_2023_svb_with_hint_wealth.jsonl `
 --labels "Standard+No Hint+RM" "Wealth+No Hint+RM" "Standard+Hint+RM" "Wealth+Hint+RM" `
 --title "DeepSeek Hedge Fund Strategy Comparison - 2023 SVB Collapse" `
 --output model_strategy_selection\results\deepseek_4_settings_comparison_rm3.png

# 4. Qwen - 2020 Covid Crash
Write-Host "Running: Qwen 2020 Covid Crash..." -ForegroundColor Yellow
& $python $eval_script `
 --inputs ..\data\backtests_with_risk_manager\allocator_reuslts_test_results_qwen_2020_crash_no_hint_standard.jsonl ..\data\backtests_with_risk_manager\allocator_reuslts_test_results_qwen_2020_crash_no_hint_wealth.jsonl ..\data\backtests_with_risk_manager\allocator_reuslts_test_results_qwen_2020_crash_with_hint_standard.jsonl ..\data\backtests_with_risk_manager\allocator_reuslts_test_results_qwen_2020_crash_with_hint_wealth.jsonl `
 --labels "Standard+No Hint+RM" "Wealth+No Hint+RM" "Standard+Hint+RM" "Wealth+Hint+RM" `
 --title "Qwen Hedge Fund Strategy Comparison - 2020 Covid Crash" `
 --output model_strategy_selection\results\qwen_4_settings_comparison_rm1.png

# 5. Qwen - 2022 Rate Hike
Write-Host "Running: Qwen 2022 Rate Hike..." -ForegroundColor Yellow
& $python $eval_script `
 --inputs ..\data\backtests_with_risk_manager\allocator_reuslts_test_results_qwen_2022_jan_no_hint_standard.jsonl ..\data\backtests_with_risk_manager\allocator_reuslts_test_results_qwen_2022_jan_no_hint_wealth.jsonl ..\data\backtests_with_risk_manager\allocator_reuslts_test_results_qwen_2022_jan_with_hint_standard.jsonl ..\data\backtests_with_risk_manager\allocator_reuslts_test_results_qwen_2022_jan_with_hint_wealth.jsonl `
 --labels "Standard+No Hint+RM" "Wealth+No Hint+RM" "Standard+Hint+RM" "Wealth+Hint+RM" `
 --title "Qwen Hedge Fund Strategy Comparison - 2022 Rate Hike" `
 --output model_strategy_selection\results\qwen_4_settings_comparison_rm2.png

# 6. Qwen - 2023 SVB Collapse
Write-Host "Running: Qwen 2023 SVB Collapse..." -ForegroundColor Yellow
& $python $eval_script `
 --inputs ..\data\backtests_with_risk_manager\allocator_reuslts_test_results_qwen_2023_svb_no_hint_standard.jsonl ..\data\backtests_with_risk_manager\allocator_reuslts_test_results_qwen_2023_svb_no_hint_wealth.jsonl ..\data\backtests_with_risk_manager\allocator_reuslts_test_results_qwen_2023_svb_with_hint_standard.jsonl ..\data\backtests_with_risk_manager\allocator_reuslts_test_results_qwen_2023_svb_with_hint_wealth.jsonl `
 --labels "Standard+No Hint+RM" "Wealth+No Hint+RM" "Standard+Hint+RM" "Wealth+Hint+RM" `
 --title "Qwen Hedge Fund Strategy Comparison - 2023 SVB Collapse" `
 --output model_strategy_selection\results\qwen_4_settings_comparison_rm3.png

 # Automation script to run experiment evaluations

Write-Host "Starting experiment evaluations without risk manager..." -ForegroundColor Cyan


# 1. DeepSeek - 2020 Covid Crash
Write-Host "Running: DeepSeek 2020 Covid Crash..." -ForegroundColor Yellow
& $python $eval_script `
 --inputs ..\data\backtests_without_risk_manager\allocator_reuslts_test_results_deepseek_2020_crash_no_hint_standard.jsonl ..\data\backtests_without_risk_manager\allocator_reuslts_test_results_deepseek_2020_crash_no_hint_wealth.jsonl ..\data\backtests_without_risk_manager\allocator_reuslts_test_results_deepseek_2020_crash_with_hint_standard.jsonl ..\data\backtests_without_risk_manager\allocator_reuslts_test_results_deepseek_2020_crash_with_hint_wealth.jsonl `
 --labels "Standard+No Hint" "Wealth+No Hint" "Standard+Hint" "Wealth+Hint" `
 --title "DeepSeek Hedge Fund Strategy Comparison - 2020 Covid Crash" `
 --output model_strategy_selection\results\deepseek_4_settings_comparison_no_rm1.png

# 2. DeepSeek - 2022 Rate Hike
Write-Host "Running: DeepSeek 2022 Rate Hike..." -ForegroundColor Yellow
& $python $eval_script `
 --inputs ..\data\backtests_without_risk_manager\allocator_reuslts_test_results_deepseek_2022_jan_no_hint_standard.jsonl ..\data\backtests_without_risk_manager\allocator_reuslts_test_results_deepseek_2022_jan_no_hint_wealth.jsonl ..\data\backtests_without_risk_manager\allocator_reuslts_test_results_deepseek_2022_jan_with_hint_standard.jsonl ..\data\backtests_without_risk_manager\allocator_reuslts_test_results_deepseek_2022_jan_with_hint_wealth.jsonl `
 --labels "Standard+No Hint" "Wealth+No Hint" "Standard+Hint" "Wealth+Hint" `
 --title "DeepSeek Hedge Fund Strategy Comparison - 2022 Rate Hike" `
 --output model_strategy_selection\results\deepseek_4_settings_comparison_no_rm2.png

# 3. DeepSeek - 2023 SVB Collapse
Write-Host "Running: DeepSeek 2023 SVB Collapse..." -ForegroundColor Yellow
& $python $eval_script `
 --inputs ..\data\backtests_without_risk_manager\allocator_reuslts_test_results_deepseek_2023_svb_no_hint_standard.jsonl ..\data\backtests_without_risk_manager\allocator_reuslts_test_results_deepseek_2023_svb_no_hint_wealth.jsonl ..\data\backtests_without_risk_manager\allocator_reuslts_test_results_deepseek_2023_svb_with_hint_standard.jsonl ..\data\backtests_without_risk_manager\allocator_reuslts_test_results_deepseek_2023_svb_with_hint_wealth.jsonl `
 --labels "Standard+No Hint" "Wealth+No Hint" "Standard+Hint" "Wealth+Hint" `
 --title "DeepSeek Hedge Fund Strategy Comparison - 2023 SVB Collapse" `
 --output model_strategy_selection\results\deepseek_4_settings_comparison_no_rm3.png

# 4. Qwen - 2020 Covid Crash
Write-Host "Running: Qwen 2020 Covid Crash..." -ForegroundColor Yellow
& $python $eval_script `
 --inputs ..\data\backtests_without_risk_manager\allocator_reuslts_test_results_qwen_2020_crash_no_hint_standard.jsonl ..\data\backtests_without_risk_manager\allocator_reuslts_test_results_qwen_2020_crash_no_hint_wealth.jsonl ..\data\backtests_without_risk_manager\allocator_reuslts_test_results_qwen_2020_crash_with_hint_standard.jsonl ..\data\backtests_without_risk_manager\allocator_reuslts_test_results_qwen_2020_crash_with_hint_wealth.jsonl `
 --labels "Standard+No Hint" "Wealth+No Hint" "Standard+Hint" "Wealth+Hint" `
 --title "Qwen Hedge Fund Strategy Comparison - 2020 Covid Crash" `
 --output model_strategy_selection\results\qwen_4_settings_comparison_no_rm1.png

# 5. Qwen - 2022 Rate Hike
Write-Host "Running: Qwen 2022 Rate Hike..." -ForegroundColor Yellow
& $python $eval_script `
 --inputs ..\data\backtests_without_risk_manager\allocator_reuslts_test_results_qwen_2022_jan_no_hint_standard.jsonl ..\data\backtests_without_risk_manager\allocator_reuslts_test_results_qwen_2022_jan_no_hint_wealth.jsonl ..\data\backtests_without_risk_manager\allocator_reuslts_test_results_qwen_2022_jan_with_hint_standard.jsonl ..\data\backtests_without_risk_manager\allocator_reuslts_test_results_qwen_2022_jan_with_hint_wealth.jsonl `
 --labels "Standard+No Hint" "Wealth+No Hint" "Standard+Hint" "Wealth+Hint" `
 --title "Qwen Hedge Fund Strategy Comparison - 2022 Rate Hike" `
 --output model_strategy_selection\results\qwen_4_settings_comparison_no_rm2.png

# 6. Qwen - 2023 SVB Collapse
Write-Host "Running: Qwen 2023 SVB Collapse..." -ForegroundColor Yellow
& $python $eval_script `
 --inputs ..\data\backtests_without_risk_manager\allocator_reuslts_test_results_qwen_2023_svb_no_hint_standard.jsonl ..\data\backtests_without_risk_manager\allocator_reuslts_test_results_qwen_2023_svb_no_hint_wealth.jsonl ..\data\backtests_without_risk_manager\allocator_reuslts_test_results_qwen_2023_svb_with_hint_standard.jsonl ..\data\backtests_without_risk_manager\allocator_reuslts_test_results_qwen_2023_svb_with_hint_wealth.jsonl `
 --labels "Standard+No Hint" "Wealth+No Hint" "Standard+Hint" "Wealth+Hint" `
 --title "Qwen Hedge Fund Strategy Comparison - 2023 SVB Collapse" `
 --output model_strategy_selection\results\qwen_4_settings_comparison_no_rm3.png



Write-Host "All evaluations complete!" -ForegroundColor Green
