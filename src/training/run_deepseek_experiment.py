import pandas as pd
import json
import logging
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import concurrent.futures
from dotenv import load_dotenv

# Load Env Vars
load_dotenv()

# --- FORCE LOCAL DATA ---
os.environ["USE_LOCAL_DATA"] = "true"

from src.agents.allocators.fundamental_allocator import fundamental_allocator
from src.agents.allocators.technical_allocator import technical_allocator
from src.agents.allocators.valuation_allocator import valuation_allocator
from src.agents.allocators.news_sentiment_allocator import news_sentiment_allocator

from src.agents.allocators.fundamental_zh_allocator import fundamental_zh_allocator
from src.agents.allocators.technical_zh_allocator import technical_zh_allocator
from src.agents.allocators.valuation_zh_allocator import valuation_zh_allocator
from src.agents.allocators.news_sentiment_zh_allocator import news_sentiment_zh_allocator

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import argparse

# --- CONFIGURATION (DEEPSEEK SPECIFIC) ---
DATA_DIR = Path("data/processed")

# DeepSeek Config
MODEL_NAME = "deepseek-chat"
MODEL_PROVIDER = "DeepSeek"

ALL_EXPERIMENTS = {
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

parser = argparse.ArgumentParser(description="Run DeepSeek Experiment over a date range.")
parser.add_argument("--start-date", type=str, default="2020-01-01", help="Start Date (YYYY-MM-DD)")
parser.add_argument("--end-date", type=str, default="2020-06-30", help="End Date (YYYY-MM-DD)")
parser.add_argument("--output-dir", type=str, default="data/training_output_deepseek_2020h1", help="Output directory")
parser.add_argument("--experiments", type=str, default="no_hint_wealth,with_hint_wealth", help="Comma-separated list of experiments to run")
parser.add_argument("--lang", type=str, default="en", choices=["en", "zh"], help="Language of allocators to use (en or zh)")
args = parser.parse_args()

START_DATE = args.start_date
END_DATE = args.end_date
OUTPUT_BASE_DIR = Path(args.output_dir)

# Filter experiments
selected_exps = [e.strip() for e in args.experiments.split(",")]
EXPERIMENTS = {k: v for k, v in ALL_EXPERIMENTS.items() if k in selected_exps}

if args.lang == "zh":
    ALLOCATORS = {
        "fundamental": fundamental_zh_allocator,
        "technical": technical_zh_allocator,
        "valuation": valuation_zh_allocator,
        "sentiment": news_sentiment_zh_allocator
    }
else:
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

def run_allocator(allocator_func, agent_id, date_str, tickers, tasks, prompt_version, current_capital=100.0, current_rf_rate=0.0):
    """
    Mock State Wrapper that forces DEEPSEEK model usage.
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
            
            # --- FORCE DEEPSEEK ---
            "model_name": MODEL_NAME,
            "model_provider": MODEL_PROVIDER
        }
    }
    
    # Adjust Date Window for Technicals (1 Year Lookback)
    dt_end = pd.Timestamp(date_str)
    dt_start = dt_end - pd.Timedelta(days=365)
    state["data"]["start_date"] = dt_start.strftime("%Y-%m-%d")
    
    # Inject Dynamic Risk Free Rate
    state["data"]["risk_free_rate"] = current_rf_rate
    
    # We can inject current_capital if the prompt wants to know its total size, 
    # but the prompt already defaults to "$100". Thus, we let the allocator use $100 
    # as a "percentage" base, and we scale it down the line. Alternatively, we just log it.
    
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
    logger.info(f"Starting DeepSeek Experiment ({START_DATE} to {END_DATE})...")
    
    # Load inputs
    inputs = {}
    for key, config in EXPERIMENTS.items():
        fname = config["input_file"]
        if fname not in inputs:
            logger.info(f"Loading {fname}...")
            inputs[fname] = load_candidates(fname)

    # Load RF Data
    rf_file = DATA_DIR / "daily_risk_free_rates.parquet"
    if rf_file.exists():
        rf_df = pd.read_parquet(rf_file)
    else:
        logger.warning(f"RF data not found at {rf_file}, using fallback.")
        rf_df = pd.DataFrame()
        
    # Filter Dates
    all_dates = sorted(list(inputs[list(inputs.keys())[0]].index))
    target_dates = [d for d in all_dates if START_DATE <= pd.Timestamp(d).strftime("%Y-%m-%d") <= END_DATE]
    
    # --- TEST LIMIT FOR 1 DAY RUN ---
    # target_dates = target_dates[:1]
    
    logger.info(f"Processing {len(target_dates)} days for full experiment...")
    
    for exp_name, config in EXPERIMENTS.items():
        logger.info(f"Starting experiment: {exp_name}")
        
        # Initialize capital tracking for this experiment
        allocator_capital = {
            "fundamental": 100.0,
            "technical": 100.0,
            "valuation": 100.0,
            "sentiment": 100.0
        }
        current_month = None
        
        for date in tqdm(target_dates):
            date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
            
            df_candidates = inputs[config["input_file"]]
            
            if date not in df_candidates.index:
                continue
                
            # Get RF Rate
            current_rf_rate = 0.05 / 252 # Default fallback
            ts = pd.Timestamp(date)
            if not rf_df.empty:
                if ts in rf_df.index:
                    current_rf_rate = float(rf_df.loc[ts, 'risk_free_rate'])
                else:
                    past_rates = rf_df[rf_df.index <= ts]
                    if not past_rates.empty:
                        current_rf_rate = float(past_rates.iloc[-1]['risk_free_rate'])
                        
            tasks_json = df_candidates.loc[date, "tasks"]
            tasks = json.loads(tasks_json)
            tickers = [t['ticker'] for t in tasks]
            
            # Prepare Output Directory
            month_str = pd.Timestamp(date).strftime("%Y_%m")
            
            # --- MONTHLY CAPITAL RESET LOGIC ---
            # If the month changes, reset the capital
            if month_str != current_month:
                if current_month is not None:
                    logger.info(f"Month changed from {current_month} to {month_str}. Applying 10/90 Capital Reset.")
                    # Distribute new base capital (90%) and retain fraction of old (10%)
                    # Default base is $100 per allocator
                    base_capital = 100.0
                    for alloc_name in allocator_capital:
                        old_cap = allocator_capital[alloc_name]
                        new_cap = (0.10 * old_cap) + (0.90 * base_capital)
                        allocator_capital[alloc_name] = new_cap
                current_month = month_str

            exp_dir = OUTPUT_BASE_DIR / exp_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            output_file = exp_dir / f"{month_str}.jsonl"
            
            day_results = {
                "date": date_str,
                "tickers": tickers,
                "tasks": tasks,
                "model": MODEL_NAME
            }
            # Parallel execution of allocators
            import concurrent.futures
            
            def process_allocator(alloc_name, alloc_func):
                current_cap = allocator_capital.get(alloc_name, 100.0)
                decision = run_allocator(
                    alloc_func, 
                    f"{alloc_name}_allocator", 
                    date_str, 
                    tickers, 
                    tasks, 
                    config["prompt_version"],
                    current_capital=current_cap,
                    current_rf_rate=current_rf_rate
                )
                return alloc_name, decision, current_cap

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(ALLOCATORS)) as executor:
                future_to_alloc = {
                    executor.submit(process_allocator, name, func): name 
                    for name, func in ALLOCATORS.items()
                }
                
                for future in concurrent.futures.as_completed(future_to_alloc):
                    alloc_name = future_to_alloc[future]
                    try:
                        name, decision, current_cap = future.result()
                        if decision:
                            day_results[name] = decision
                            day_results[name]["starting_capital"] = current_cap
                    except Exception as exc:
                        logger.error(f"{alloc_name} generated an exception: {exc}")

            with open(output_file, "a") as f:
                f.write(json.dumps(day_results) + "\n")

    logger.info("DeepSeek Experiment Complete.")

if __name__ == "__main__":
    main()
