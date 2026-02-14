import pandas as pd
import json
import logging
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv

# Load Env Vars
load_dotenv()

# --- FORCE LOCAL DATA ---
os.environ["USE_LOCAL_DATA"] = "true"

# Import Allocators
from src.agents.allocators.fundamental_allocator import fundamental_allocator
from src.agents.allocators.technical_allocator import technical_allocator
from src.agents.allocators.valuation_allocator import valuation_allocator
from src.agents.allocators.news_sentiment_allocator import news_sentiment_allocator

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION (QWEN SPECIFIC) ---
DATA_DIR = Path("data/processed")
OUTPUT_BASE_DIR = Path("data/training_output_qwen_2020h1")

# Qwen Config
MODEL_NAME = "qwen-max"
MODEL_PROVIDER = "Dashscope"

# Date Range: 2020-01-01 to 2020-06-30
START_DATE = "2020-01-01"
END_DATE = "2020-06-30"

EXPERIMENTS = {
    "no_hint_standard": {
        "input_file": "daily_candidates_no_hint.parquet",
        "prompt_version": "standard"
    },
    "no_hint_wealth": {
        "input_file": "daily_candidates_no_hint.parquet",
        "prompt_version": "wealth"
    },
    "with_hint_standard": {
        "input_file": "daily_candidates_with_hint.parquet",
        "prompt_version": "standard"
    },
    "with_hint_wealth": {
        "input_file": "daily_candidates_with_hint.parquet",
        "prompt_version": "wealth"
    }
}

ALLOCATORS = {
    "fundamental": fundamental_allocator,
    "technical": technical_allocator,
    "valuation": valuation_allocator,
    "sentiment": news_sentiment_allocator
}

def load_candidates(filename):
    path = DATA_DIR / filename
    if not path.exists():
        logger.error(f"Input file not found: {path}")
        return None
    return pd.read_parquet(path)

def run_allocator(allocator_func, agent_id, date_str, tickers, tasks, prompt_version):
    """
    Mock State Wrapper that forces QWEN model usage.
    """
    # Construct State
    state = {
        "data": {
            "tickers": tickers,
            "end_date": date_str,
            "start_date": date_str, 
            "tasks": tasks
        },
        "metadata": {
            "prompt_version": prompt_version,
            "allocator_mode": "global_batch",
            "show_reasoning": False,
            
            # --- FORCE QWEN ---
            "model_name": MODEL_NAME,
            "model_provider": MODEL_PROVIDER
        }
    }
    
    # Adjust Date Window for Technicals (1 Year Lookback)
    dt_end = pd.Timestamp(date_str)
    dt_start = dt_end - pd.Timedelta(days=365)
    state["data"]["start_date"] = dt_start.strftime("%Y-%m-%d")

    try:
        result = allocator_func(state, agent_id=agent_id)
        if "allocator_decisions" in result["data"]:
            return result["data"]["allocator_decisions"][agent_id]
        else:
            logger.warning(f"No decision returned for {agent_id}")
            return None
    except Exception as e:
        logger.error(f"Error running {agent_id} on {date_str}: {e}")
        return None

def main():
    logger.info(f"Starting Qwen-Max Experiment ({START_DATE} to {END_DATE})...")
    
    # Load inputs
    inputs = {}
    for key, config in EXPERIMENTS.items():
        fname = config["input_file"]
        if fname not in inputs:
            logger.info(f"Loading {fname}...")
            inputs[fname] = load_candidates(fname)

    # Filter Dates
    all_dates = sorted(list(inputs[list(inputs.keys())[0]].index))
    target_dates = [d for d in all_dates if START_DATE <= pd.Timestamp(d).strftime("%Y-%m-%d") <= END_DATE]
    
    # --- TEST LIMIT: 3 DAYS ---
    target_dates = target_dates[:3]
    
    logger.info(f"Processing {len(target_dates)} days (2020 H1 Test Run)...")
    
    for date in tqdm(target_dates):
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        
        for exp_name, config in EXPERIMENTS.items():
            df_candidates = inputs[config["input_file"]]
            
            if date not in df_candidates.index:
                continue
                
            tasks_json = df_candidates.loc[date, "tasks"]
            tasks = json.loads(tasks_json)
            tickers = [t['ticker'] for t in tasks]
            
            # Prepare Output Directory
            month_str = pd.Timestamp(date).strftime("%Y_%m")
            exp_dir = OUTPUT_BASE_DIR / exp_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            output_file = exp_dir / f"{month_str}.jsonl"
            
            day_results = {
                "date": date_str,
                "tickers": tickers,
                "tasks": tasks,
                "model": MODEL_NAME
            }
            
            for alloc_name, alloc_func in ALLOCATORS.items():
                decision = run_allocator(
                    alloc_func, 
                    f"{alloc_name}_allocator", 
                    date_str, 
                    tickers, 
                    tasks, 
                    config["prompt_version"]
                )
                if decision:
                    day_results[alloc_name] = decision

            with open(output_file, "a") as f:
                f.write(json.dumps(day_results) + "\n")

    logger.info("Qwen-Max Experiment Complete.")

if __name__ == "__main__":
    main()
