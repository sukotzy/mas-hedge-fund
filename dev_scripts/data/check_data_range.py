import pandas as pd
from pathlib import Path

def check_range():
    p = Path('data/raw/sp500_ohlcv.parquet')
    if not p.exists():
        print("File not found")
        return
    df = pd.read_parquet(p)
    print(f"Date Range: {df['date'].min()} to {df['date'].max()}")

if __name__ == "__main__":
    check_range()
