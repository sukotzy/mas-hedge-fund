import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure local imports work
sys.path.append(str(Path(__file__).resolve().parent))
from src.data.loader import get_local_loader

def update_factors_fast():
    print("Loading existing stock factors...")
    factors_path = Path("data/processed/stock_factors.parquet")
    if not factors_path.exists():
        print("Error: stock_factors.parquet not found!")
        return
        
    factors_df = pd.read_parquet(factors_path)
    print(f"Loaded {len(factors_df)} rows from existing factor database.")
    
    print("Loading OHLCV data to compute new metrics...")
    loader = get_local_loader("data")
    ohlcv = loader.ohlcv.copy()
    
    if ohlcv.empty:
        print("Error: No OHLCV data found in raw parquets.")
        return
        
    print("Mapping Permno to Tickers...")
    constituents = loader.constituents
    permno_map = constituents[['permno', 'ticker']].drop_duplicates(subset='permno', keep='last')
    
    ohlcv = ohlcv.merge(permno_map, on='permno', how='left')
    ohlcv = ohlcv.dropna(subset=['ticker'])
    ohlcv = ohlcv.sort_values('vol', ascending=False).drop_duplicates(subset=['date', 'ticker'])
    
    print("Pivoting matrices for fast rolling calculation...")
    prices = ohlcv.pivot(index='date', columns='ticker', values='prc').ffill(limit=3)
    volume = ohlcv.pivot(index='date', columns='ticker', values='vol').fillna(0)
    
    print("Computing Returns, Volatility (20d), and Volume Ratio (20d)...")
    returns = np.log(prices / prices.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan)
    
    volatility = returns.rolling(window=20).std()
    volume_avg = volume.rolling(window=20).mean().replace(0, np.nan)
    volume_ratio = volume / volume_avg
    
    print("Melting matrices to align with factor database...")
    # Convert wide to long
    vol_long = volatility.reset_index().melt(id_vars='date', var_name='ticker', value_name='volatility_20d')
    vr_long = volume_ratio.reset_index().melt(id_vars='date', var_name='ticker', value_name='volume_ratio')
    
    # Merge them
    new_metrics = pd.merge(vol_long, vr_long, on=['date', 'ticker'])
    # Drop rows with all NaN in metrics to save memory
    new_metrics = new_metrics.dropna(subset=['volatility_20d', 'volume_ratio'], how='all')
    
    print("Merging new metrics into existing stock_factors.parquet...")
    # Ensure types match
    factors_df['date'] = pd.to_datetime(factors_df['date'])
    new_metrics['date'] = pd.to_datetime(new_metrics['date'])
    
    # Clean up existing columns if they were partially added
    if 'volatility_20d' in factors_df.columns:
        factors_df = factors_df.drop(columns=['volatility_20d'])
    if 'volume_ratio' in factors_df.columns:
        factors_df = factors_df.drop(columns=['volume_ratio'])
        
    # Join!
    updated_factors = pd.merge(factors_df, new_metrics, on=['date', 'ticker'], how='left')
    
    # Fill NAs for safety
    updated_factors['volatility_20d'] = updated_factors['volatility_20d'].fillna(0)
    updated_factors['volume_ratio'] = updated_factors['volume_ratio'].fillna(1.0)
    
    print("Saving updated factor database...")
    updated_factors.to_parquet(factors_path, index=False)
    print(f"Success! {factors_path} has been successfully updated with the new columns. No MST recalculation needed!")

if __name__ == "__main__":
    update_factors_fast()
