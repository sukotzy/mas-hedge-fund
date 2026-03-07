#!/bin/bash

export USE_LOCAL_DATA=true
export PYTHONPATH="."

echo "========================================================================="
echo "Starting DeepSeek Experiment Period 1: 2020-02-20 to 2020-03-10"
echo "========================================================================="
python src/training/run_deepseek_experiment.py --start-date "2020-02-20" --end-date "2020-03-10" --experiments "no_hint_standard,no_hint_wealth,with_hint_standard,with_hint_wealth" --lang "zh" --output-dir "data/results_deepseek/2020_crash"

echo ""
echo "========================================================================="
echo "Starting DeepSeek Experiment Period 2: 2022-01-03 to 2022-01-21"
echo "========================================================================="
python src/training/run_deepseek_experiment.py --start-date "2022-01-03" --end-date "2022-01-21" --experiments "no_hint_standard,no_hint_wealth,with_hint_standard,with_hint_wealth" --lang "zh" --output-dir "data/results_deepseek/2022_jan"

echo ""
echo "========================================================================="
echo "Starting DeepSeek Experiment Period 3: 2023-03-01 to 2023-03-15"
echo "========================================================================="
python src/training/run_deepseek_experiment.py --start-date "2023-03-01" --end-date "2023-03-15" --experiments "no_hint_standard,no_hint_wealth,with_hint_standard,with_hint_wealth" --lang "zh" --output-dir "data/results_deepseek/2023_svb"

echo ""
echo "All DeepSeek experiments finished successfully!"
