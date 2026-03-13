import pandas as pd
from pathlib import Path

def verify_data():
    data_dir = Path("data/processed")
    regime_path = data_dir / "market_regime.parquet"
    factors_path = data_dir / "stock_factors.parquet"
    
    print("-" * 50)
    print("VERIFYING BATCH DATA")
    print("-" * 50)
    
    # 1. Market Regime
    if regime_path.exists():
        df_regime = pd.read_parquet(regime_path)
        print(f"\n[Market Regime] {regime_path}")
        print(f"Rows: {len(df_regime)}")
        print(f"Date Range: {df_regime.index.min()} to {df_regime.index.max()}")
        print(f"Columns: {list(df_regime.columns)}")
        print("Regime Counts:")
        print(df_regime['regime'].value_counts())
        print("Sample:")
        print(df_regime.head(3))
    else:
        print(f"[ERROR] {regime_path} NOT FOUND")

    # 2. Stock Factors
    if factors_path.exists():
        df_factors = pd.read_parquet(factors_path)
        print(f"\n[Stock Factors] {factors_path}")
        print(f"Rows: {len(df_factors)}")
        print(f"Date Range: {df_factors['date'].min()} to {df_factors['date'].max()}")
        print(f"Columns: {list(df_factors.columns)}")
        print("Sample:")
        print(df_factors.head(3))
        
        # Check Stats
        print("\nStatistics:")
        print(df_factors[['anomaly_score', 'degree', 'panic_score']].describe())
    else:
        print(f"[ERROR] {factors_path} NOT FOUND")
        
if __name__ == "__main__":
    verify_data()
