import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from tqdm import tqdm
from src.selection.batch.pipeline import run_batch_pipeline

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
    
    logger.info(f"Processing {len(valid_dates)} days through Batch Pipeline...")
    
    results_hint = []
    results_no_hint = []
    
    for date in tqdm(valid_dates):
        ts = pd.Timestamp(date)
        date_str = ts.strftime('%Y-%m-%d')
        
        try:
            # 1. With Hint
            output_hint = run_batch_pipeline(end_date=date_str, lookback_days=252, include_hint=True)
            tasks_hint = output_hint.get('tasks', [])
            if tasks_hint:
                results_hint.append({'date': ts, 'tasks': json.dumps(tasks_hint)})

            # 2. No Hint
            output_no_hint = run_batch_pipeline(end_date=date_str, lookback_days=252, include_hint=False)
            tasks_no_hint = output_no_hint.get('tasks', [])
            if tasks_no_hint:
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
