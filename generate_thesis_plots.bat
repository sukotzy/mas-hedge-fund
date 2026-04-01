@echo off
echo =========================================================
echo Generating ALL Decoupled Thesis Plots (Panel A ^& B)
echo Output Directory: thesis_plots/
echo =========================================================

echo.
echo [1/7] Generating Figure 1: The Macro View...
python src/plot_figure1_macro.py

echo.
echo [2/7] Generating Figure 3: LTS vs RDM Dynamics (Panel A ^& B)...
python src/plot_figure3_dynamics.py

echo.
echo [3/7] Generating Figure 4: Baseline Ensembles (Panel A ^& B)...
python src/plot_figure4_ensembles.py

echo.
echo [4/7] Plotting Transfer Rate (TR) Ablation Matrix - SN50...
python src/plot_alpha_comparison.py ^
  --no-benchmark ^
  --decouple ^
  --suptitle "Darwinian Zero-Sum Intensity: Transfer Rate (TR) Multiplier Ablation" ^
  --files data\experiments\segcap10\rate005_soft_decay_segcap10_smooth.jsonl data\experiments\segcap10\rate005_soft_decay_segcap10_smooth_tr5.jsonl data\experiments\segcap10\rate005_soft_decay_segcap10_smooth_tr10.jsonl data\experiments\segcap10\rate005_soft_decay_segcap10_smooth_tr20.jsonl data\experiments\segcap10\rate005_soft_decay_segcap10_smooth_tr30.jsonl data\experiments\segcap10\rate005_soft_decay_segcap10_smooth_tr40.jsonl data\experiments\segcap10\rate005_soft_decay_segcap10_smooth_tr50.jsonl ^
  --labels "TR=1x" "TR=5x" "TR=10x" "TR=20x" "TR=30x (50%% $\gamma_{loss}$)" "TR=40x (50%% $\gamma_{loss}$)" "TR=50x (50%% $\gamma_{loss}$)" ^
  --titles "Panel A: 4-Agents without Cash (Fixed at 10%% $\kappa_{segregate}$)" ^
  --files data\experiments\segcap20\rate005_soft_decay_segcap20_smooth_wcash.jsonl data\experiments\segcap20\rate005_soft_decay_segcap20_smooth_tr5_wcash.jsonl data\experiments\segcap20\rate005_soft_decay_segcap20_smooth_tr10_wcash.jsonl data\experiments\segcap20\rate005_soft_decay_segcap20_smooth_tr20_wcash.jsonl data\experiments\segcap20\rate005_soft_decay_segcap20_smooth_tr30_wcash.jsonl data\experiments\segcap20\rate005_soft_decay_segcap20_smooth_tr40_wcash.jsonl data\experiments\segcap20\rate005_soft_decay_segcap20_smooth_tr50_wcash.jsonl ^
  --labels "TR=1x" "TR=5x" "TR=10x" "TR=20x" "TR=30x (50%% $\gamma_{loss}$)" "TR=40x (50%% $\gamma_{loss}$)" "TR=50x (50%% $\gamma_{loss}$)" ^
  --titles "Panel B: 5-Agents with Cash Haven (Fixed at 20%% $\kappa_{segregate}$)" ^
  --output plots/tr_ablation_comparison_sn50.png

echo.
echo [5/7] Plotting Transfer Rate (TR) Ablation Matrix - SN80...
python src/plot_alpha_comparison.py ^
  --no-benchmark ^
  --decouple ^
  --suptitle "Darwinian Zero-Sum Intensity: Transfer Rate (TR) Multiplier Ablation" ^
  --files data\experiments\segcap10\rate005_soft_decay_segcap10_smooth.jsonl data\experiments\segcap10\rate005_soft_decay_segcap10_smooth_tr5.jsonl data\experiments\segcap10\rate005_soft_decay_segcap10_smooth_tr10.jsonl data\experiments\segcap10\rate005_soft_decay_segcap10_smooth_tr20.jsonl data\experiments\segcap10\rate005_soft_decay_segcap10_smooth_tr302.jsonl data\experiments\segcap10\rate005_soft_decay_segcap10_smooth_tr402.jsonl data\experiments\segcap10\rate005_soft_decay_segcap10_smooth_tr502.jsonl ^
  --labels "TR=1x" "TR=5x" "TR=10x" "TR=20x" "TR=30x (80%% $\gamma_{loss}$)" "TR=40x (80%% $\gamma_{loss}$)" "TR=50x (80%% $\gamma_{loss}$)" ^
  --titles "Panel A: 4-Agents without Cash (Fixed at 10%% $\kappa_{segregate}$)" ^
  --files data\experiments\segcap20\rate005_soft_decay_segcap20_smooth_wcash.jsonl data\experiments\segcap20\rate005_soft_decay_segcap20_smooth_tr5_wcash.jsonl data\experiments\segcap20\rate005_soft_decay_segcap20_smooth_tr10_wcash.jsonl data\experiments\segcap20\rate005_soft_decay_segcap20_smooth_tr20_wcash.jsonl data\experiments\segcap20\rate005_soft_decay_segcap20_smooth_tr30_wcash2.jsonl data\experiments\segcap20\rate005_soft_decay_segcap20_smooth_tr40_wcash2.jsonl data\experiments\segcap20\rate005_soft_decay_segcap20_smooth_tr50_wcash2.jsonl ^
  --labels "TR=1x" "TR=5x" "TR=10x" "TR=20x" "TR=30x (80%% $\gamma_{loss}$)" "TR=40x (80%% $\gamma_{loss}$)" "TR=50x (80%% $\gamma_{loss}$)" ^
  --titles "Panel B: 5-Agents with Cash Haven (Fixed at 20%% $\kappa_{segregate}$)" ^
  --output plots/tr_ablation_comparison_sn80.png

echo.
echo [6/7] Plotting SegCap Control Variable Matrix (10%% to 50%%)...
python src/plot_alpha_comparison.py ^
  --no-benchmark ^
  --decouple ^
  --suptitle "Capital Reinvestment Friction: Segregated Capital ($\kappa_{segregate}$) Ablation" ^
  --files data\experiments\segcap10\rate005_soft_decay_segcap10_smooth.jsonl data\experiments\segcap20\rate005_soft_decay_segcap20_smooth.jsonl data\experiments\rate005_soft_decay_segcap30_smooth.jsonl data\experiments\rate005_soft_decay_segcap40_smooth.jsonl data\experiments\rate005_soft_decay_segcap50_smooth.jsonl ^
  --labels "10%% $\kappa_{segregate}$" "20%% $\kappa_{segregate}$" "30%% $\kappa_{segregate}$" "40%% $\kappa_{segregate}$" "50%% $\kappa_{segregate}$" ^
  --titles "Panel A: 4-Agent Framework (Without Virtual Cash Haven)" ^
  --files data\experiments\segcap10\rate005_soft_decay_segcap10_smooth_wcash.jsonl data\experiments\segcap20\rate005_soft_decay_segcap20_smooth_wcash.jsonl data\experiments\rate005_soft_decay_segcap30_smooth_wcash.jsonl data\experiments\rate005_soft_decay_segcap40_smooth_wcash.jsonl data\experiments\rate005_soft_decay_segcap50_smooth_wcash.jsonl ^
  --labels "10%% $\kappa_{segregate}$" "20%% $\kappa_{segregate}$" "30%% $\kappa_{segregate}$" "40%% $\kappa_{segregate}$" "50%% $\kappa_{segregate}$" ^
  --titles "Panel B: 5-Agent Framework (With Virtual Cash Haven)" ^
  --output plots/segcap_ablation_comparison.png

echo.
echo [7/7] Plotting The Ultimate Dual-Panel Decay Comparison...
python src/plot_alpha_comparison.py ^
  --no-benchmark ^
  --decouple ^
  --suptitle "Signal Retention and Risk Diagnostics: An Orthogonal Cross-Ablation" ^
  --files data\soft_harsh_decay\rate005_no_decay.jsonl data\soft_harsh_decay\rate005_harsh_decay.jsonl data\soft_harsh_decay\rate005_soft_decay.jsonl data\soft_harsh_decay\rate005_psdh.jsonl data\soft_harsh_decay\rate005_phds.jsonl data\soft_harsh_decay\rate005_p08_dh.jsonl data\soft_harsh_decay\rate005_p08_ds.jsonl ^
  --labels "Immediate Liquidation" "Harsh (8%% + MA5, Fast Decay)" "Soft (20%% Panic, Slow Decay)" "Hybrid (20%% Panic, Fast Decay)" "Hybrid (8%% + MA5, Slow Decay)" "Pure 8%% Panic (Fast Decay)" "Pure 8%% Panic (Slow Decay)" ^
  --titles "Panel A: 4-Agent Framework (Without Virtual Cash Haven)" ^
  --files data\soft_harsh_decay\rate005_no_decay_wcash.jsonl data\soft_harsh_decay\rate005_harsh_decay_wcash.jsonl data\soft_harsh_decay\rate005_soft_decay_wcash.jsonl data\soft_harsh_decay\rate005_psdh_wcash.jsonl data\soft_harsh_decay\rate005_phds_wcash.jsonl data\soft_harsh_decay\rate005_p08_dh_wcash.jsonl data\soft_harsh_decay\rate005_p08_ds_wcash.jsonl ^
  --labels "Immediate Liquidation" "Harsh (8%% + MA5, Fast Decay)" "Soft (20%% Panic, Slow Decay)" "Hybrid (20%% Panic, Fast Decay)" "Hybrid (8%% + MA5, Slow Decay)" "Pure 8%% Panic (Fast Decay)" "Pure 8%% Panic (Slow Decay)" ^
  --titles "Panel B: 5-Agent Framework (With Virtual Cash Haven)" ^
  --output plots/decay_dual_panel_comparison_7lines.png

echo.
echo =========================================================
echo All Splitted Panel Figures Generated Successfully! 
echo Please check the 'thesis_plots/' directory!
echo =========================================================
pause
