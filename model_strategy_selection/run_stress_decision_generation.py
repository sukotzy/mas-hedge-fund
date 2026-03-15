import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from pathlib import Path

# 获取项目根目录 (假设当前脚本之后会被移动到 model_strategy_selection/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

STRESS_TESTS = {
    "covid_crash": {
        "start": "2020-02-20",
        "end": "2020-03-10"
    },
    "rate_hike": {
        "start": "2022-01-03",
        "end": "2022-01-21"
    },
    "svb_collapse": {
        "start": "2023-03-01",
        "end": "2023-03-15"
    }
}

SCRIPTS = [
    ("qwen", str(PROJECT_ROOT / "src/training/run_qwen_experiment.py")),
    ("deepseek", str(PROJECT_ROOT / "src/training/run_deepseek_experiment.py"))
]

EXPERIMENTS_TO_RUN = "no_hint_wealth,with_hint_wealth"
PYTHON_BIN = str(PROJECT_ROOT / "hf/Scripts/python.exe")

def main():
    logger.info("Starting Stress Tests Suite...")
    
    for test_name, dates in STRESS_TESTS.items():
        logger.info(f"=== Running Stress Test: {test_name.upper()} ({dates['start']} to {dates['end']}) ===")
        
        for model_name, script_path in SCRIPTS:
            for lang in ["en", "zh"]:
                output_dir = f"data/training_output_{model_name}_{lang}_{test_name}"
                logger.info(f"-> Starting {model_name.upper()} ({lang.upper()}) execution. Output: {output_dir}")
                
                cmd = [
                    PYTHON_BIN, script_path,
                    "--start-date", dates["start"],
                    "--end-date", dates["end"],
                    "--output-dir", output_dir,
                    "--experiments", EXPERIMENTS_TO_RUN,
                    "--lang", lang
                ]
                
                logger.info(f"Running command: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
                logger.info(f"-> {model_name.upper()} execution for {test_name} completed successfully.")
            except subprocess.CalledProcessError as e:
                logger.error(f"-> execution failed for {model_name} on {test_name}: {e}")
                
    logger.info("All stress tests have been executed.")

if __name__ == "__main__":
    main()
