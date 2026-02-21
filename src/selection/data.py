import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional
from src.data.loader import get_local_loader

class SelectionDataLoader:
    """
    Handles data fetching for the Selection Layer with strict Point-in-Time constraints.
    Ensures no future data (Look-ahead bias) is included.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.loader = get_local_loader(data_dir)
        
    def fetch_universe_data(self, end_date: str, lookback_days: int = 252) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fetch Price (Adjusted Close), Volume, and Constituents universe for the lookback window.
        
        Args:
            end_date: The analysis date (inclusive). We only know data UP TO this day.
            lookback_days: Number of trading days approx (or calendar days) to look back.
            
        Returns:
            prices_df: DataFrame (Date x Ticker) of Close prices.
            volume_df: DataFrame (Date x Ticker) of Volume.
            constituents: List of tickers strictly valid in index on end_date.
        """
        
        target_date = pd.to_datetime(end_date)
        start_date = target_date - timedelta(days=int(lookback_days * 1.5)) # Buffer for non-trading days
        
        # 1. Get Universe (S&P 500 Constituents AT end_date)
        # Avoid survivorship bias by ensuring we only pick what was actually in the index then.
        if self.loader.constituents.empty:
            print("Warning: Constituents data empty. Using all available tickers.")
            available_tickers = self.loader.ohlcv['ticker'].unique()
            valid_tickers = available_tickers
        else:
            # Filter constituents strictly
            valid_constituents = self.loader.constituents[
                (self.loader.constituents['start'] <= target_date) & 
                (self.loader.constituents['ending'] >= target_date)
            ]
            valid_tickers = valid_constituents['ticker'].unique()
            
        if len(valid_tickers) == 0:
            print(f"Warning: No valid constituents found for {end_date}. Check constituent dates.")
            return pd.DataFrame(), pd.DataFrame(), []

        # Build Permno -> Ticker map for the valid set
        # We need this because OHLCV usually only has Permno
        permno_map = valid_constituents[['permno', 'ticker']].drop_duplicates(subset='permno')
        valid_permnos = permno_map['permno'].unique()

        # 2. Fetch OHLCV Data
        # We need efficient bulk loading.
        df = self.loader.ohlcv.copy()
        
        # Filter Date Range (Strictly <= end_date)
        date_mask = (df['date'] >= start_date) & (df['date'] <= target_date)
        df = df[date_mask]
        
        if df.empty:
             print(f"Warning: No price data found between {start_date} and {end_date}.")
             return pd.DataFrame(), pd.DataFrame(), []
             
        # Filter for valid permnos
        if 'permno' not in df.columns:
            print("Error: 'permno' column missing in ohlcv data.")
            return pd.DataFrame(), pd.DataFrame(), []
            
        df = df[df['permno'].isin(valid_permnos)]
        
        # Merge Ticker info back onto OHLCV
        # This allows us to pivot by Ticker
        df = df.merge(permno_map, on='permno', how='left')
        
        # Drop rows where ticker is NaN (if permno map incomplete)
        df = df.dropna(subset=['ticker', 'date'])

        # Handle duplicates: If multiple Permnos map to same Ticker (e.g. Class A/B), 
        # or data issues. We keep the first one or one with highest volume?
        # Simple fix: Drop duplicates on Date/Ticker
        df = df.sort_values('vol', ascending=False).drop_duplicates(subset=['date', 'ticker'])
        
        # 3. Pivot to Wide Format (Date x Ticker)
        # Choosing 'prc' (Close Price) - Use Adjusted if available, but 'prc' is usually raw close in WRDS.
        # Ideally we want Adjusted Close for returns. 
        # For this prototype, we'll use 'prc' (Close).
        # Note: If 'ret' (Total Return) column exists in WRDS Compustat/CRSP, it's better for calculating correlations on returns directly.
        # But let's stick to prices -> log returns.
        
        prices_df = df.pivot(index='date', columns='ticker', values='prc')
        
        # Ensure 'vol' column exists before pivoting
        if 'vol' not in df.columns:
            # Fallback if 'vol' is missing from raw fetch for some reason, though it should be there.
            # We strictly need volume.
            # If completely missing, we have a problem.
            # Try to see if there's 'volume' or other alias.
            # For now panic if not found or fill 0.
            # Assuming 'vol' is standard CRSP.
            print("Warning: 'vol' column missing in DataFrame. creating empty volume df.")
            volume_df = pd.DataFrame(0, index=prices_df.index, columns=prices_df.columns)
        else:
            volume_df = df.pivot(index='date', columns='ticker', values='vol')
        
        # Forward fill missing prices for short gaps (limit 3 days), then drop remaining NaNs (inactive stocks)
        prices_df = prices_df.ffill(limit=3).dropna(axis=1, how='any')
        
        # Align volume with valid prices
        # Fill missing volume with 0 (CRSP sometimes has NaN for 0 volume)
        volume_df = volume_df[prices_df.columns].loc[prices_df.index].fillna(0)
        
        valid_tickers_final = prices_df.columns.tolist()
        
        return prices_df, volume_df, valid_tickers_final

    def fetch_fundamentals_snapshot(self, end_date: str, tickers: List[str]) -> pd.DataFrame:
        """
        Fetch latest fundamental snapshot for given tickers AS OF end_date.
        Uses 'rdq' (Release Date Quarterly) to ensure Point-in-Time correctness.
        """
        target_date = pd.to_datetime(end_date)
        fund_df = self.loader.fundamentals.copy()
        
        if fund_df.empty:
            return pd.DataFrame()
            
        # Filter by RDQ <= End Date (We must have SEEN the report)
        mask = (fund_df['rdq'] <= target_date)
        valid_funds = fund_df[mask]
        
        # We need to map Tickers -> GVKEYs to link with fundamentals
        # Ideally Loader handles this, but for bulk:
        # We can try to map back or iterate.
        # Since we need this for 50 tickers max later, maybe acceptable to be slower?
        # Or simplistic:
        # We need a Ticker->GVKEY map valid at end_date.
        
        # For now, let's assume we can rely on the loader's helper to get gvkey for each ticker
        # Optimization: Build a bulk map
        
        records = []
        for ticker in tickers:
            permno = self.loader.get_permno(ticker, end_date)
            if not permno: continue
            gvkey = self.loader.get_gvkey(permno)
            if not gvkey: continue
            
            # Get latest record for this gvkey
            company_rows = valid_funds[valid_funds['gvkey'] == gvkey]
            if company_rows.empty: continue
            
            # Sort by RDQ desc, take top 1
            latest = company_rows.sort_values('rdq', ascending=False).iloc[0]
            
            # Extract minimal fields needed for factors (Book Value, etc.)
            # P/B Ratio = Price / (Total Equity / Shares)
            # We need Price at end_date.
            
            records.append({
                'ticker': ticker,
                'gvkey': gvkey,
                'rdq': latest['rdq'],
                'total_equity': latest.get('seqq', np.nan), # Shareholders Equity
                'net_income': latest.get('niq', np.nan)
            })
            
        return pd.DataFrame(records)

    def fetch_sectors(self, tickers: List[str], target_date: str = None) -> pd.Series:
        """Fetch static GICS Sector numeric code (gsector) for a list of tickers utilizing GVKEY mapping. 
        Unmapped become 0 or NaN."""
        if not hasattr(self.loader, 'company_info') or self.loader.company_info.empty:
            return pd.Series(np.nan, index=tickers)
            
        result = pd.Series(index=tickers, dtype=float)
        
        # We need a date to resolve permno if available, else just use today
        lookup_date = target_date if target_date else pd.Timestamp.today().strftime('%Y-%m-%d')
        
        # company_info now only has gvkey, gsector, etc. (no ticker)
        # We must map Ticker -> Permno -> GVKEY -> Gsector
        for t in tickers:
            permno = self.loader.get_permno(t, lookup_date)
            if not permno:
                result[t] = np.nan
                continue
                
            gvkey = self.loader.get_gvkey(permno)
            if not gvkey:
                result[t] = np.nan
                continue
                
            # Find in company info
            matches = self.loader.company_info[self.loader.company_info['gvkey'] == gvkey]
            if not matches.empty:
                try:
                    gsector_val = matches.iloc[-1]['gsector']
                    result[t] = float(gsector_val) if pd.notna(gsector_val) else np.nan
                except (ValueError, KeyError):
                     result[t] = np.nan
            else:
                 result[t] = np.nan
                 
        return result

