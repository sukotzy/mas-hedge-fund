import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from tqdm import tqdm
from src.selection.pipeline import run_batch_pipeline
from src.selection.data import SelectionDataLoader

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def batch_selection(
    processed_dir: str = "data/processed",
    output_dir: str = "data/processed",
    start_year: int = 2016
):
    """
    Generate Top Candidates for every day in history using the batch pipeline.
    Generates TWO datasets:
    1. daily_candidates_with_hint.parquet (Action: Long/Short)
    2. daily_candidates_no_hint.parquet (Action: Analyze)
    """
    p_dir = Path(processed_dir)
    factors_path = p_dir / "stock_factors.parquet"
    
    if not factors_path.exists():
        logger.error("Pre-computed factors not found. Run factor_db.py first.")
        return

    logger.info("Loading Available Dates from Factors...")
    factors_df = pd.read_parquet(factors_path, columns=['date'])
    
    # Get unique dates from factors
    valid_dates = factors_df['date'].unique()
    valid_dates = np.sort(valid_dates)
    
    # Filter by start year
    # Ensure end date is capped at 2024 if needed, but data limits usually handle it. 
    valid_dates = [d for d in valid_dates if pd.Timestamp(d).year >= start_year]
    
    logger.info("Pre-loading entire Factor & Regime DB into memory for fast slicing...")
    regime_path = p_dir / "market_regime.parquet"
    regime_full = pd.read_parquet(regime_path)
    
    factors_full = pd.read_parquet(factors_path)
    factors_full['date'] = pd.to_datetime(factors_full['date'])
    factors_full.set_index('date', inplace=True)
    
    logger.info("Pre-loading OHLCV matrices (this takes ~5 seconds once)...")
    loader = SelectionDataLoader()
    loader.fetch_all_history()
    
    preloaded = {
        'loader': loader,
        'regime_full': regime_full,
        'factors_full': factors_full
    }
    
    logger.info(f"Processing {len(valid_dates)} days through Batch Pipeline (In-Memory)...")
    
    results_hint = []
    results_no_hint = []
    
    for date in tqdm(valid_dates):
        ts = pd.Timestamp(date)
        date_str = ts.strftime('%Y-%m-%d')
        
        try:
            # Run Once (With Hint)
            output_hint = run_batch_pipeline(end_date=date_str, lookback_days=252, include_hint=True, preloaded_data=preloaded)
            tasks_hint = output_hint.get('tasks', [])
            
            if tasks_hint:
                results_hint.append({'date': ts, 'tasks': json.dumps(tasks_hint)})
                
                # Derive No Hint Version
                tasks_no_hint = []
                for t in tasks_hint:
                    t_no_hint = t.copy()
                    t_no_hint['action'] = "analyze"
                    # reason starts with "Cluster X: ...", so we can extract "Cluster X"
                    cluster_str = t_no_hint['reason'].split(':')[0]
                    t_no_hint['reason'] = f"{cluster_str} Representative (Hidden)"
                    tasks_no_hint.append(t_no_hint)
                    
                results_no_hint.append({'date': ts, 'tasks': json.dumps(tasks_no_hint)})
                
        except Exception as e:
            logger.error(f"Error processing {date_str}: {e}")
            continue
        
    # Save Hint Version
    if results_hint:
        df_hint = pd.DataFrame(results_hint).set_index('date')
        path_hint = Path(output_dir) / "daily_candidates_with_hint.parquet"
        path_hint.parent.mkdir(parents=True, exist_ok=True)
        df_hint.to_parquet(path_hint)
        logger.info(f"Saved: {path_hint} ({len(df_hint)} days)")
        
    # Save No Hint Version
    if results_no_hint:
        df_no_hint = pd.DataFrame(results_no_hint).set_index('date')
        path_no_hint = Path(output_dir) / "daily_candidates_no_hint.parquet"
        path_no_hint.parent.mkdir(parents=True, exist_ok=True)
        df_no_hint.to_parquet(path_no_hint)
        logger.info(f"Saved: {path_no_hint} ({len(df_no_hint)} days)")

if __name__ == "__main__":
    batch_selection()
