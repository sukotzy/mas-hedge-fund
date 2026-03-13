import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.loader import LocalDataLoader

def check_event_counts():
    loader = LocalDataLoader(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw"))
    
    ticker = "AAPL"
    print(f"Checking event counts for {ticker}...")
    
    # Load all news/keydev
    # We can peek at the raw parquet or just use loader if getting all history is supported?
    # Loader relies on 'end_date' filtering.
    # Let's read the parquet directly for speed.
    
    keydev_path = os.path.join(loader.data_dir, "sp500_keydev.parquet")
    if not os.path.exists(keydev_path):
        print("Keydev parquet not found.")
        return

    df = pd.read_parquet(keydev_path)
    # Join with constituents to get GVKEY -> Ticker mapping?
    # OR just assume we have the mapping.
    # Actually, let's use the loader's map.
    loader._load_constituents()
    gvkey = loader.ticker_to_gvkey.get(ticker)
    
    if not gvkey:
        print(f"GVKEY not found for {ticker}")
        return

    # Filter for this stock
    stock_events = df[df['gvkey'] == gvkey].copy()
    stock_events['date'] = pd.to_datetime(stock_events['announcedate'])
    
    # Group by Month-Year
    monthly_counts = stock_events.groupby(stock_events['date'].dt.to_period('M')).size()
    
    print("\n--- Event Counts per Month (Top 10) ---")
    print(monthly_counts.sort_values(ascending=False).head(10))
    
    # Find a specific date with > 3 events?
    # Group by Date
    daily_counts = stock_events.groupby('date').size()
    print("\n--- Top Busy Days (>2 events) ---")
    busy_days = daily_counts[daily_counts >= 3].sort_index(ascending=False)
    print(busy_days.head(10))

if __name__ == "__main__":
    check_event_counts()
