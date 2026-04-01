@echo off
echo =========================================================
echo Re-Rendering Model Setting Comparisons (SegCap = kappa)
echo Output Directory: plots/ (Monolithic Dual-Panel Mode)
echo =========================================================

echo.
echo [1/3] Plotting Model Settings (TR=1x)...
python src/plot_alpha_comparison.py ^
  --benchmark "^GSPC" ^
  --suptitle "" ^
  --decouple ^
  --files data\22_24_exp\rate005_soft_decay_segcap20_smooth_dshw.jsonl data\22_24_exp\rate005_soft_decay_segcap10_smooth_dshw.jsonl data\22_24_exp\rate005_soft_decay_segcap20_smooth_dsnhw.jsonl data\22_24_exp\rate005_soft_decay_segcap10_smooth_dsnhw.jsonl data\22_24_exp\rate005_soft_decay_segcap20_smooth_qwhw.jsonl data\22_24_exp\rate005_soft_decay_segcap10_smooth_qwhw.jsonl ^
  --labels "DeepSeek (Hint, 20%% $\kappa_{segregate}$)" "DeepSeek (Hint, 10%% $\kappa_{segregate}$)" "DeepSeek (No Hint, 20%% $\kappa_{segregate}$)" "DeepSeek (No Hint, 10%% $\kappa_{segregate}$)" "Qwen (Hint, 20%% $\kappa_{segregate}$)" "Qwen (Hint, 10%% $\kappa_{segregate}$)" ^
  --titles "Panel A: 4-Agent Framework (Without Virtual Cash Haven)" ^
  --files data\22_24_exp\rate005_soft_decay_segcap20_smooth_dshw_wcash.jsonl data\22_24_exp\rate005_soft_decay_segcap10_smooth_dshw_wcash.jsonl data\22_24_exp\rate005_soft_decay_segcap20_smooth_dsnhw_wcash.jsonl data\22_24_exp\rate005_soft_decay_segcap10_smooth_dsnhw_wcash.jsonl data\22_24_exp\rate005_soft_decay_segcap20_smooth_qwhw_wcash.jsonl data\22_24_exp\rate005_soft_decay_segcap10_smooth_qwhw_wcash.jsonl ^
  --labels "DeepSeek (Hint, 20%% $\kappa_{segregate}$)" "DeepSeek (Hint, 10%% $\kappa_{segregate}$)" "DeepSeek (No Hint, 20%% $\kappa_{segregate}$)" "DeepSeek (No Hint, 10%% $\kappa_{segregate}$)" "Qwen (Hint, 20%% $\kappa_{segregate}$)" "Qwen (Hint, 10%% $\kappa_{segregate}$)" ^
  --titles "Panel B: 5-Agent Framework (With Virtual Cash Haven)" ^
  --output plots/model_setting_segcap_comparison_22_24.png

echo.
echo [2/3] Plotting Model Settings (TR=25x)...
python src/plot_alpha_comparison.py ^
  --benchmark "^GSPC" ^
  --suptitle "" ^
  --decouple ^
  --files data\22_24_exp\rate005_soft_decay_segcap20_smooth_dshw_tr25.jsonl data\22_24_exp\rate005_soft_decay_segcap10_smooth_dshw_tr25.jsonl data\22_24_exp\rate005_soft_decay_segcap20_smooth_dsnhw_tr25.jsonl data\22_24_exp\rate005_soft_decay_segcap10_smooth_dsnhw_tr25.jsonl data\22_24_exp\rate005_soft_decay_segcap20_smooth_qwhw_tr25.jsonl data\22_24_exp\rate005_soft_decay_segcap10_smooth_qwhw_tr25.jsonl ^
  --labels "DeepSeek (Hint, 20%% $\kappa_{segregate}$)" "DeepSeek (Hint, 10%% $\kappa_{segregate}$)" "DeepSeek (No Hint, 20%% $\kappa_{segregate}$)" "DeepSeek (No Hint, 10%% $\kappa_{segregate}$)" "Qwen (Hint, 20%% $\kappa_{segregate}$)" "Qwen (Hint, 10%% $\kappa_{segregate}$)" ^
  --titles "Panel A: 4-Agent Framework (Without Virtual Cash Haven)" ^
  --files data\22_24_exp\rate005_soft_decay_segcap20_smooth_dshw_wcash_tr25.jsonl data\22_24_exp\rate005_soft_decay_segcap10_smooth_dshw_wcash_tr25.jsonl data\22_24_exp\rate005_soft_decay_segcap20_smooth_dsnhw_wcash_tr25.jsonl data\22_24_exp\rate005_soft_decay_segcap10_smooth_dsnhw_wcash_tr25.jsonl data\22_24_exp\rate005_soft_decay_segcap20_smooth_qwhw_wcash_tr25.jsonl data\22_24_exp\rate005_soft_decay_segcap10_smooth_qwhw_wcash_tr25.jsonl ^
  --labels "DeepSeek (Hint, 20%% $\kappa_{segregate}$)" "DeepSeek (Hint, 10%% $\kappa_{segregate}$)" "DeepSeek (No Hint, 20%% $\kappa_{segregate}$)" "DeepSeek (No Hint, 10%% $\kappa_{segregate}$)" "Qwen (Hint, 20%% $\kappa_{segregate}$)" "Qwen (Hint, 10%% $\kappa_{segregate}$)" ^
  --titles "Panel B: 5-Agent Framework (With Virtual Cash Haven)" ^
  --output plots/model_setting_segcap_comparison_22_24_tr25.png

echo.
echo [3/3] Plotting Model Settings (TR=50x)...
python src/plot_alpha_comparison.py ^
  --benchmark "^GSPC" ^
  --suptitle "" ^
  --decouple ^
  --files data\22_24_exp\rate005_soft_decay_segcap20_smooth_dshw_tr50.jsonl data\22_24_exp\rate005_soft_decay_segcap10_smooth_dshw_tr50.jsonl data\22_24_exp\rate005_soft_decay_segcap20_smooth_dsnhw_tr50.jsonl data\22_24_exp\rate005_soft_decay_segcap10_smooth_dsnhw_tr50.jsonl data\22_24_exp\rate005_soft_decay_segcap20_smooth_qwhw_tr50.jsonl data\22_24_exp\rate005_soft_decay_segcap10_smooth_qwhw_tr50.jsonl ^
  --labels "DeepSeek (Hint, 20%% $\kappa_{segregate}$)" "DeepSeek (Hint, 10%% $\kappa_{segregate}$)" "DeepSeek (No Hint, 20%% $\kappa_{segregate}$)" "DeepSeek (No Hint, 10%% $\kappa_{segregate}$)" "Qwen (Hint, 20%% $\kappa_{segregate}$)" "Qwen (Hint, 10%% $\kappa_{segregate}$)" ^
  --titles "Panel A: 4-Agent Framework (Without Virtual Cash Haven)" ^
  --files data\22_24_exp\rate005_soft_decay_segcap20_smooth_dshw_wcash_tr50.jsonl data\22_24_exp\rate005_soft_decay_segcap10_smooth_dshw_wcash_tr50.jsonl data\22_24_exp\rate005_soft_decay_segcap20_smooth_dsnhw_wcash_tr50.jsonl data\22_24_exp\rate005_soft_decay_segcap10_smooth_dsnhw_wcash_tr50.jsonl data\22_24_exp\rate005_soft_decay_segcap20_smooth_qwhw_wcash_tr50.jsonl data\22_24_exp\rate005_soft_decay_segcap10_smooth_qwhw_wcash_tr50.jsonl ^
  --labels "DeepSeek (Hint, 20%% $\kappa_{segregate}$)" "DeepSeek (Hint, 10%% $\kappa_{segregate}$)" "DeepSeek (No Hint, 20%% $\kappa_{segregate}$)" "DeepSeek (No Hint, 10%% $\kappa_{segregate}$)" "Qwen (Hint, 20%% $\kappa_{segregate}$)" "Qwen (Hint, 10%% $\kappa_{segregate}$)" ^
  --titles "Panel B: 5-Agent Framework (With Virtual Cash Haven)" ^
  --output plots/model_setting_segcap_comparison_22_24_tr50.png

echo =========================================================
echo Done! Re-rendered grids successfully!
echo =========================================================
pause
