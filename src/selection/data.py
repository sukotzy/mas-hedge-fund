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
        self._preloaded_prices = None
        self._preloaded_volume = None
        
    def fetch_all_history(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Pre-loads and pivots the entire OHLCV dataset into memory ONCE.
        This eliminates the O(N) looping cost of pivoting millions of rows daily.
        Returns:
            prices_df: Full Date x Ticker DataFrame (forward filled).
            volume_df: Full Date x Ticker DataFrame.
        """
        import time
        start_t = time.time()
        
        if self._preloaded_prices is not None:
            return self._preloaded_prices, self._preloaded_volume
            
        print("SelectionDataLoader: Pre-pivoting 10-year OHLCV data into memory matrices...")
        df = self.loader.ohlcv.copy()
        valid_constituents = self.loader.constituents
        
        # Build Permno -> Ticker map for the valid set (use newest mapping if multiple)
        permno_map = valid_constituents[['permno', 'ticker']].drop_duplicates(subset='permno', keep='last')
        
        # Merge Ticker info back onto OHLCV
        df = df.merge(permno_map, on='permno', how='left')
        df = df.dropna(subset=['ticker', 'date'])
        df = df.sort_values('vol', ascending=False).drop_duplicates(subset=['date', 'ticker'])
        
        # Pivot
        prices_df = df.pivot(index='date', columns='ticker', values='prc')
        
        if 'vol' not in df.columns:
            volume_df = pd.DataFrame(0, index=prices_df.index, columns=prices_df.columns)
        else:
            volume_df = df.pivot(index='date', columns='ticker', values='vol')
            
        # Forward fill up to 3 days. We don't dropna entirely here because 
        # a stock might be valid in 2018 but dead in 2020. We keep all NaNs.
        prices_df = prices_df.ffill(limit=3)
        volume_df = volume_df.fillna(0)
        
        self._preloaded_prices = prices_df
        self._preloaded_volume = volume_df
        
        print(f"SelectionDataLoader: Matrices ready in {time.time() - start_t:.2f}s. Shape: {prices_df.shape}")
        return prices_df, volume_df

    def fetch_universe_data(self, end_date: str, lookback_days: int = 252) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Fetch Price, Volume, and Constituents universe for the lookback window.
        Optimized version: purely slices the preloaded memory matrices in O(1).
        """
        target_date = pd.to_datetime(end_date)
        start_date = target_date - timedelta(days=int(lookback_days * 1.5))
        
        # 1. Get Universe
        if self.loader.constituents.empty:
            return pd.DataFrame(), pd.DataFrame(), []
            
        valid_constituents = self.loader.constituents[
            (self.loader.constituents['start'] <= target_date) & 
            (self.loader.constituents['ending'] >= target_date)
        ]
        valid_tickers = valid_constituents['ticker'].unique().tolist()
        
        if not valid_tickers:
            return pd.DataFrame(), pd.DataFrame(), []
            
        # 2. Slice preloaded matrices
        prices_full, volume_full = self.fetch_all_history()
        
        # Slicing by date range
        mask = (prices_full.index >= start_date) & (prices_full.index <= target_date)
        
        # Slicing by tickers (only valid ones that exist in the columns)
        valid_cols = [t for t in valid_tickers if t in prices_full.columns]
        
        p_slice = prices_full.loc[mask, valid_cols]
        v_slice = volume_full.loc[mask, valid_cols]
        
        # Drop columns that are completely NaN in this window
        p_slice = p_slice.dropna(axis=1, how='all')
        
        # Drop stocks that still have NaNs in the lookback window (need full history for MST)
        p_slice = p_slice.dropna(axis=1, how='any')
        
        # Align volume
        v_slice = v_slice[p_slice.columns].loc[p_slice.index].fillna(0)
        
        return p_slice, v_slice, p_slice.columns.tolist()

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
        Uses an O(1) dictionary cache to avoid slow DataFrame scans."""
        if not hasattr(self.loader, 'company_info') or self.loader.company_info.empty:
            return pd.Series(np.nan, index=tickers)
            
        # 1. Build Cache once (O(1) lookups instead of O(N) DataFrame scans)
        if not hasattr(self, '_sector_cache'):
            # Build GVKEY -> Gsector map
            # Assuming company_info has unique gvkeys or taking the last one
            self._gvkey_to_sector = dict(zip(self.loader.company_info['gvkey'].astype(str), self.loader.company_info['gsector']))
            
        result = pd.Series(index=tickers, dtype=float)
        lookup_date = target_date if target_date else pd.Timestamp.today().strftime('%Y-%m-%d')
        
        # 2. Iterate and map
        for t in tickers:
            # get_permno is still a bit slow (dataframe scan), but acceptable if cached in loader.
            # Ideally get_permno is also fully O(1) in the loader itself.
            permno = self.loader.get_permno(t, lookup_date)
            if not permno:
                result[t] = np.nan
                continue
                
            gvkey = self.loader.get_gvkey(permno)
            if not gvkey:
                result[t] = np.nan
                continue
                
            # O(1) Dict Lookup
            gsector_val = self._gvkey_to_sector.get(str(gvkey).zfill(6))
            if gsector_val is not None and pd.notna(gsector_val):
                try:
                    result[t] = float(gsector_val)
                except ValueError:
                    result[t] = np.nan
            else:
                result[t] = np.nan
                 
        return result

