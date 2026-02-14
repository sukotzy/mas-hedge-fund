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

# --- CONFIGURATION ---
DATA_DIR = Path("data/processed")
OUTPUT_BASE_DIR = Path("data/training_output")

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
    Mock State Wrapper to call Allocators.
    """
    # Construct State
    state = {
        "data": {
            "tickers": tickers,
            "end_date": date_str,
            "start_date": date_str, # Tech allocator needs start/end, usually expects longer window but fetches via get_prices which handles it? 
            # Tech Allocator: get_prices(ticker, start_date, end_date)
            # We should probably set start_date back 200 days to ensure indicators work?
            # Actually technical_allocator relies on get_prices. If start_date == end_date, fetch might fail or return 1 day.
            # Let's check tech allocator: Line 37: get_prices(ticker, start_date, end_date)
            # If we want 50 days (Line 43), we need window.
            # Fix: Set start_date to 365 days ago.
            "tasks": tasks
        },
        "metadata": {
            "prompt_version": prompt_version,
            "allocator_mode": "global_batch", # For sentiment
            "show_reasoning": False
        }
    }
    
    # Adjust Date Window for Technicals
    dt_end = pd.Timestamp(date_str)
    dt_start = dt_end - pd.Timedelta(days=365)
    state["data"]["start_date"] = dt_start.strftime("%Y-%m-%d")

    try:
        result = allocator_func(state, agent_id=agent_id)
        # Extract decision
        if "allocator_decisions" in result["data"]:
            return result["data"]["allocator_decisions"][agent_id]
        else:
            logger.warning(f"No decision returned for {agent_id}")
            return None
    except Exception as e:
        logger.error(f"Error running {agent_id} on {date_str}: {e}")
        return None

def main():
    logger.info("Starting Training Data Generation...")
    
    # Load all inputs first to avoid repeated I/O
    inputs = {}
    for key, config in EXPERIMENTS.items():
        fname = config["input_file"]
        if fname not in inputs:
            logger.info(f"Loading {fname}...")
            inputs[fname] = load_candidates(fname)

    # Union of all Dates (should be same, but safe check)
    all_dates = sorted(list(inputs[list(inputs.keys())[0]].index))
    # Filter for testing?
    # all_dates = all_dates[-5:] # Uncomment to test last 5 days
    
    logger.info(f"Processing {len(all_dates)} days across {len(EXPERIMENTS)} experiments...")
    
    for date in tqdm(all_dates):
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        
        for exp_name, config in EXPERIMENTS.items():
            df_candidates = inputs[config["input_file"]]
            
            # Get Tasks for this day
            if date not in df_candidates.index:
                continue
                
            tasks_json = df_candidates.loc[date, "tasks"]
            tasks = json.loads(tasks_json)
            tickers = [t['ticker'] for t in tasks]
            
            # Prepare Output Directory for this Experiment
            # We save by Month to avoid 3000 files, but also avoid 1 huge file.
            # e.g. data/training_output/exp_name/2016_01.jsonl
            month_str = pd.Timestamp(date).strftime("%Y_%m")
            exp_dir = OUTPUT_BASE_DIR / exp_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            output_file = exp_dir / f"{month_str}.jsonl"
            
            day_results = {
                "date": date_str,
                "tickers": tickers,
                "tasks": tasks
            }
            
            # Run All Allocators
            for alloc_name, alloc_func in ALLOCATORS.items():
                # Check if result already exists? (Resume capability)
                # Implementing resume at line-level is expensive. 
                # Optimization: Load 'done' set if needed. For now simple append.
                
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

            # Append to JSONL
            with open(output_file, "a") as f:
                f.write(json.dumps(day_results) + "\n")

    logger.info("Training Data Generation Complete.")

if __name__ == "__main__":
    main()
