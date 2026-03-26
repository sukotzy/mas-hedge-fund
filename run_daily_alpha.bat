python run_all_backtests.py --base-dirs data/deepseek_standard_hint_9yr_zh --out-dir data/daily_alpha_raw --fast --disable-risk-manager --turnover-penalty 0.0 --agent sentiment --decay-mode none

python run_all_backtests.py --base-dirs data/deepseek_standard_hint_9yr_zh --out-dir data/daily_alpha_raw --fast --disable-risk-manager --turnover-penalty 0.0 --agent fundamental --decay-mode none

python run_all_backtests.py --base-dirs data/deepseek_standard_hint_9yr_zh --out-dir data/daily_alpha_raw --fast --disable-risk-manager --turnover-penalty 0.0 --agent technical --decay-mode none

python run_all_backtests.py --base-dirs data/deepseek_standard_hint_9yr_zh --out-dir data/daily_alpha_raw --fast --disable-risk-manager --turnover-penalty 0.0 --agent valuation --decay-mode none
