import pandas as pd
from pathlib import Path

# Path to the parquet file
parquet_path = Path("data/raw/sp500_ohlcv.parquet")

if not parquet_path.exists():
    print(f"Error: File not found at {parquet_path.absolute()}")
else:
    try:
        df = pd.read_parquet(parquet_path)
        print("Columns in sp500_ohlcv.parquet:")
        print(df.columns.tolist())
        
        if 'vol' in df.columns:
            print("\n'vol' column found!")
            print("Sample volume data:")
            print(df['vol'].head())
            print(f"Missing values: {df['vol'].isna().sum()}")
        else:
            print("\n'vol' column NOT found!")
            
    except Exception as e:
        print(f"Error reading parquet: {e}")
