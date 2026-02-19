import pandas as pd
from src.selection.pipeline import run_batch_pipeline
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_output():
    # 1. Check available dates in factor DB
    try:
        regime_df = pd.read_parquet("data/processed/market_regime.parquet")
        available_dates = regime_df.index.sort_values()
        
        if available_dates.empty:
            logger.error("No data in market_regime.parquet. Did factor_db.py run successfully?")
            return

        # Pick a date from the end of the available range
        target_date = available_dates[-1].strftime('%Y-%m-%d')
        logger.info(f"Testing Selection Pipeline for date: {target_date}")
        
        # 2. Run Pipeline
        start_time = time.time()
        result = run_batch_pipeline(end_date=target_date)
        end_time = time.time()
        
        # 3. Report
        logger.info(f"Execution Time: {end_time - start_time:.4f} seconds")
        logger.info(f"Market State: {result.get('market_state')}")
        logger.info(f"NTL: {result.get('ntl')}")
        
        tasks = result.get('tasks', [])
        logger.info(f"Generated {len(tasks)} candidates.")
        
        for i, task in enumerate(tasks):
            logger.info(f"Candidate {i+1}: {task['ticker']} (Action: {task['action']}, Confidence: {task['confidence']})")
            
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_output()
