import pandas as pd
import numpy as np
import random
import logging
from src.selection.pipeline import run_selection_pipeline
from pathlib import Path

# Setup simple logging to console
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    print("Loading dates from data...")
    try:
        # Load dates to pick valid ones
        # We can just check the raw parquet quickly or use the loader if initialized
        df = pd.read_parquet("data/raw/sp500_ohlcv.parquet", columns=['date'])
        unique_dates = df['date'].unique()
        unique_dates = pd.to_datetime(unique_dates).sort_values()
        
        # Valid range: Need at least 252 days history for lookback
        start_idx = 260 
        valid_dates = unique_dates[start_idx:]
        
        if len(valid_dates) < 3:
            print("Not enough data history to pick 3 random days.")
            return

        # Pick 3 random dates
        random_dates = sorted(random.sample(list(valid_dates), 3))
        
        print(f"Selected Test Dates: {[d.strftime('%Y-%m-%d') for d in random_dates]}")
        
        for date_val in random_dates:
            date_str = date_val.strftime('%Y-%m-%d')
            print(f"\n--- Running Pipeline for {date_str} ---")
            
            # Run with Hint
            print("  [Mode: With Hint]")
            try:
                result = run_selection_pipeline(date_str, lookback_days=252, include_hint=True)
                print(f"  Market Regime: {result['market_state']}")
                print(f"  Selected Candidates (All):")
                for task in result['tasks']:
                    print(f"    - {task['ticker']}: {task['action'].upper()} | {task['reason']}")
            except Exception as e:
                print(f"  Failed (Hint=True): {e}")

            # Run without Hint
            print("  [Mode: No Hint]")
            try:
                result = run_selection_pipeline(date_str, lookback_days=252, include_hint=False)
                print(f"  Selected Candidates (All):")
                for task in result['tasks']:
                    print(f"    - {task['ticker']}: {task['action'].upper()} | {task['reason']}")
            except Exception as e:
                print(f"  Failed (Hint=False): {e}")
                
    except FileNotFoundError:
        print("Error: data/raw/sp500_ohlcv.parquet not found. Please run downloader first.")

if __name__ == "__main__":
    main()
