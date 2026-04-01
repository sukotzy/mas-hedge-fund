:: 1. Harsh Decay (不带现金避风港)
python run_all_backtests.py --base-dirs data/deepseek_standard_hint_9yr_zh --out-dir data/soft_harsh_decay --fast --turnover-penalty 0.05 --decay-mode harsh --enable-smoothing --active-agents sentiment fundamental valuation technical --out-name rate005_harsh_decay.jsonl

:: 2. Harsh Decay (带现金避风港 wcash)
python run_all_backtests.py --base-dirs data/deepseek_standard_hint_9yr_zh --out-dir data/soft_harsh_decay --fast --turnover-penalty 0.05 --decay-mode harsh --enable-smoothing --out-name rate005_harsh_decay_wcash.jsonl

:: 3. Soft Decay (不带现金避风港)
python run_all_backtests.py --base-dirs data/deepseek_standard_hint_9yr_zh --out-dir data/soft_harsh_decay --fast --turnover-penalty 0.05 --decay-mode soft --enable-smoothing --active-agents sentiment fundamental valuation technical --out-name rate005_soft_decay.jsonl

:: 4. Soft Decay (带现金避风港 wcash)
python run_all_backtests.py --base-dirs data/deepseek_standard_hint_9yr_zh --out-dir data/soft_harsh_decay --fast --turnover-penalty 0.05 --decay-mode soft --enable-smoothing --out-name rate005_soft_decay_wcash.jsonl

:: 5. No Decay (不带现金避风港)
python run_all_backtests.py --base-dirs data/deepseek_standard_hint_9yr_zh --out-dir data/soft_harsh_decay --fast --turnover-penalty 0.05 --decay-mode none --enable-smoothing --active-agents sentiment fundamental valuation technical --out-name rate005_no_decay.jsonl

:: 6. No Decay (带现金避风港)
python run_all_backtests.py --base-dirs data/deepseek_standard_hint_9yr_zh --out-dir data/soft_harsh_decay --fast --turnover-penalty 0.05 --decay-mode none --enable-smoothing --out-name rate005_no_decay_wcash.jsonl


python run_all_backtests.py --base-dirs data/deepseek_standard_hint_9yr_zh --out-dir data/soft_harsh_decay --fast --turnover-penalty 0.05 --decay-mode panic_soft_decay_harsh --enable-smoothing --active-agents sentiment fundamental valuation technical --out-name rate005_psdh.jsonl
python run_all_backtests.py --base-dirs data/deepseek_standard_hint_9yr_zh --out-dir data/soft_harsh_decay --fast --turnover-penalty 0.05 --decay-mode panic_soft_decay_harsh --enable-smoothing --out-name rate005_psdh_wcash.jsonl
python run_all_backtests.py --base-dirs data/deepseek_standard_hint_9yr_zh --out-dir data/soft_harsh_decay --fast --turnover-penalty 0.05 --decay-mode panic_harsh_decay_soft --enable-smoothing --active-agents sentiment fundamental valuation technical --out-name rate005_phds.jsonl
python run_all_backtests.py --base-dirs data/deepseek_standard_hint_9yr_zh --out-dir data/soft_harsh_decay --fast --turnover-penalty 0.05 --decay-mode panic_harsh_decay_soft --enable-smoothing --out-name rate005_phds_wcash.jsonl

:: 7. Pure 8% Panic (Fast Decay) Ablation
python run_all_backtests.py --base-dirs data/deepseek_standard_hint_9yr_zh --out-dir data/soft_harsh_decay --fast --turnover-penalty 0.05 --decay-mode panic_08_decay_harsh --enable-smoothing --active-agents sentiment fundamental valuation technical --out-name rate005_p08_dh.jsonl
python run_all_backtests.py --base-dirs data/deepseek_standard_hint_9yr_zh --out-dir data/soft_harsh_decay --fast --turnover-penalty 0.05 --decay-mode panic_08_decay_harsh --enable-smoothing --out-name rate005_p08_dh_wcash.jsonl

:: 8. Pure 8% Panic (Slow Decay) Ablation
python run_all_backtests.py --base-dirs data/deepseek_standard_hint_9yr_zh --out-dir data/soft_harsh_decay --fast --turnover-penalty 0.05 --decay-mode panic_08_decay_soft --enable-smoothing --active-agents sentiment fundamental valuation technical --out-name rate005_p08_ds.jsonl
python run_all_backtests.py --base-dirs data/deepseek_standard_hint_9yr_zh --out-dir data/soft_harsh_decay --fast --turnover-penalty 0.05 --decay-mode panic_08_decay_soft --enable-smoothing --out-name rate005_p08_ds_wcash.jsonl


:: 9. Plot The Ultimate Dual-Panel Comparison
python src/plot_alpha_comparison.py ^
  --no-benchmark ^
  --suptitle "Signal Retention and Risk Diagnostics: An Orthogonal Cross-Ablation" ^
  --files data\soft_harsh_decay\rate005_no_decay.jsonl data\soft_harsh_decay\rate005_harsh_decay.jsonl data\soft_harsh_decay\rate005_soft_decay.jsonl data\soft_harsh_decay\rate005_psdh.jsonl data\soft_harsh_decay\rate005_phds.jsonl data\soft_harsh_decay\rate005_p08_dh.jsonl data\soft_harsh_decay\rate005_p08_ds.jsonl ^
  --labels "Immediate Liquidation" "Harsh (8% + MA5, Fast Decay)" "Soft (20% Panic, Slow Decay)" "Hybrid (20% Panic, Fast Decay)" "Hybrid (8% + MA5, Slow Decay)" "Pure 8% Panic (Fast Decay)" "Pure 8% Panic (Slow Decay)" ^
  --titles "Panel A: 4-Agent Framework (Without Virtual Cash Haven)" ^
  --files data\soft_harsh_decay\rate005_no_decay_wcash.jsonl data\soft_harsh_decay\rate005_harsh_decay_wcash.jsonl data\soft_harsh_decay\rate005_soft_decay_wcash.jsonl data\soft_harsh_decay\rate005_psdh_wcash.jsonl data\soft_harsh_decay\rate005_phds_wcash.jsonl data\soft_harsh_decay\rate005_p08_dh_wcash.jsonl data\soft_harsh_decay\rate005_p08_ds_wcash.jsonl ^
  --labels "Immediate Liquidation" "Harsh (8% + MA5, Fast Decay)" "Soft (20% Panic, Slow Decay)" "Hybrid (20% Panic, Fast Decay)" "Hybrid (8% + MA5, Slow Decay)" "Pure 8% Panic (Fast Decay)" "Pure 8% Panic (Slow Decay)" ^
  --titles "Panel B: 5-Agent Framework (With Virtual Cash Haven)" ^
  --output plots/decay_dual_panel_comparison_7lines.png