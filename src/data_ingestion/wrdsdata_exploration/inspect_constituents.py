'''Verified the S&P 500 list format.'''

import pandas as pd
import os

def main():
    path = 'data/raw/sp500_constituents.parquet'
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    df = pd.read_parquet(path)
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df[['permno', 'comnam', 'ticker', 'start', 'ending']].head())
    
    # Check if there are duplicate tickers or permnos
    print(f"\nUnique PERMNOs: {df['permno'].nunique()}")
    print(f"Unique Tickers: {df['ticker'].nunique()}")
    
    # Find a case where 1 PERMNO has multiple Tickers (common reason for mapping)
    counts = df.groupby('permno')['ticker'].nunique()
    multi_ticker = counts[counts > 1]
    if not multi_ticker.empty:
        p = multi_ticker.index[0]
        print(f"\nExample of PERMNO {p} changing tickers:")
        print(df[df['permno'] == p][['start', 'ending', 'ticker', 'comnam']])

if __name__ == "__main__":
    main()
