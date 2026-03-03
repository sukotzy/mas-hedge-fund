import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    ("qwen", "src/training/run_qwen_experiment.py"),
    ("deepseek", "src/training/run_deepseek_experiment.py")
]

EXPERIMENTS_TO_RUN = "no_hint_wealth,with_hint_wealth"
PYTHON_BIN = r".\hf\Scripts\python.exe"

def main():
    logger.info("Starting Stress Tests Suite...")
    
    for test_name, dates in STRESS_TESTS.items():
        logger.info(f"=== Running Stress Test: {test_name.upper()} ({dates['start']} to {dates['end']}) ===")
        
        for model_name, script_path in SCRIPTS:
            output_dir = f"data/training_output_{model_name}_{test_name}"
            logger.info(f"-> Starting {model_name.upper()} execution. Output: {output_dir}")
            
            cmd = [
                PYTHON_BIN, script_path,
                "--start-date", dates["start"],
                "--end-date", dates["end"],
                "--output-dir", output_dir,
                "--experiments", EXPERIMENTS_TO_RUN
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
