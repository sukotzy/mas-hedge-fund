import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import random

# Disable info logging from pipeline to keep output clean
logging.getLogger("src.selection.pipeline").setLevel(logging.WARNING)

from src.selection.pipeline import run_selection_pipeline
from src.selection.data import LocalDataLoader

def calculate_avg_pairwise_correlation(returns_df: pd.DataFrame) -> float:
    """Calculate the average upper-triangle pairwise correlation."""
    if returns_df.empty or len(returns_df.columns) < 2:
        return 0.0
    
    corr_matrix = returns_df.corr().values
    
    # Extract the upper triangle, excluding the diagonal (k=1)
    upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
    upper_tri_values = corr_matrix[upper_tri_indices]
    
    # Handle NaNs
    upper_tri_values = upper_tri_values[~np.isnan(upper_tri_values)]
    
    if len(upper_tri_values) == 0:
        return 0.0
        
    return np.mean(upper_tri_values)

def main():
    parser = argparse.ArgumentParser(description="Calculate Correlation of Layer 2 Selected Stocks")
    parser.add_argument("--dates", type=str, nargs="+", default=["2020-03-23", "2022-09-30", "2023-01-04", "2024-01-03"],
                        help="List of YYYY-MM-DD dates to test")
    parser.add_argument("--lookback", type=int, default=252, help="Lookback days for correlation (default: 252)")
    args = parser.parse_args()
    
    print("=================================================================================")
    print("      Testing Selection Layer Diversity (Average Pairwise Correlation)     ")
    print("=================================================================================")
    print(f"{'Date':<12} | {'Regime':<20} | {'Layer 2 Avg Corr':<16} | {'Random Avg Corr':<15}")
    print("-" * 81)
    
    for test_date in args.dates:
        try:
            # 1. Run Pipeline to get 5 stocks
            result = run_selection_pipeline(test_date, lookback_days=args.lookback, include_hint=False)
            selected_tickers = [task["ticker"] for task in result["tasks"]]
            regime = result["market_state"]
            
            # 2. Get data for these specific stocks to calculate their correlation
            # We use the data loader
            loader = LocalDataLoader()
            prices_df, _ = loader.load_data(test_date, lookback_days=args.lookback)
            
            if prices_df.empty:
                print(f"{test_date:<12} | No data found.")
                continue
                
            returns_df = np.log(prices_df / prices_df.shift(1)).dropna()
            
            # Filter to selected stocks only (handle if one was dropped somehow)
            valid_selected = [t for t in selected_tickers if t in returns_df.columns]
            selected_returns = returns_df[valid_selected]
            
            avg_corr_selected = calculate_avg_pairwise_correlation(selected_returns)
            
            # 3. Baseline: Random 5 stocks from the same universe
            universe = list(returns_df.columns)
            random_tickers = random.sample(universe, min(5, len(universe)))
            random_returns = returns_df[random_tickers]
            
            avg_corr_random = calculate_avg_pairwise_correlation(random_returns)
            
            print(f"{test_date:<12} | {regime:<20} | {avg_corr_selected:>15.4f} | {avg_corr_random:>14.4f}")
            
            print(f"  -> Selected: {', '.join(valid_selected)}")
        except Exception as e:
            print(f"{test_date:<12} | Error: {e}")
            
    print("=================================================================================")
    print("Lower correlation indicates better portfolio diversification (orthogonal risks).")

if __name__ == "__main__":
    main()
