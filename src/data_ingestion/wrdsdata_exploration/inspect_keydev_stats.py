'''Analyzed the volume/types of news events before we built the agent.'''

import pandas as pd
from pathlib import Path

def main():
    path = Path("data/raw/sp500_keydev.parquet")
    if not path.exists():
        print("File not found.")
        return

    df = pd.read_parquet(path)
    print(f"File Path: {path}")
    print(f"Size: {path.stat().st_size / (1024*1024):.2f} MB")
    print(f"Rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    if 'announcedate' in df.columns:
        print(f"Date Range: {df['announcedate'].min()} to {df['announcedate'].max()}")
        print("\nEvents per Year:")
        print(df['announcedate'].dt.year.value_counts().sort_index())
    
    print("\nSample Headlines:")
    print(df[['announcedate', 'headline', 'keydeveventtypeid']].head(5))

if __name__ == "__main__":
    main()
