"""
Pre-loaded Price Matrix for fast O(1) price lookups.
Eliminates per-ticker per-day API calls by loading the entire OHLCV 
dataset into a pivoted (date × permno) matrix in memory.
"""
import logging
import pandas as pd
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class PriceMatrix:
    """
    Pre-loads the ENTIRE OHLCV dataset into memory as a (date x ticker) matrix.
    Provides O(1) price lookups and fast DataFrame slicing for any ticker/date range.
    """
    def __init__(self, data_dir: str = "data"):
        from src.data.loader import get_local_loader
        
        logger.info("Pre-loading OHLCV price matrix into memory...")
        loader = get_local_loader(data_dir)
        loader._build_caches()
        
        # Build ticker -> permno mapping (use most recent permno for each ticker)
        self._ticker_to_permno = {}
        for ticker, records in loader._permno_cache.items():
            if records:
                self._ticker_to_permno[ticker] = records[0][2]  # Most recent permno
        
        # Build pivoted close price matrix: index=date, columns=permno
        # Reset index first since loader sets permno as both index and column
        ohlcv = loader.ohlcv.reset_index(drop=True).copy()
        ohlcv = ohlcv.sort_values('date')
        
        # Pivot to get close prices (prc column) by permno and date
        self._close_matrix = ohlcv.pivot_table(
            index='date', columns='permno', values='prc', aggfunc='last'
        )
        
        # Date index as sorted array for fast bisect
        self._dates = self._close_matrix.index.values
        
        logger.info(f"Price matrix loaded: {self._close_matrix.shape[0]} dates x {self._close_matrix.shape[1]} permnos")
    
    def get_permno(self, ticker: str) -> Optional[int]:
        return self._ticker_to_permno.get(ticker)
    
    def get_close(self, ticker: str, date_str: str) -> float:
        """Get closing price for a ticker on a specific date."""
        permno = self.get_permno(ticker)
        if permno is None or permno not in self._close_matrix.columns:
            return 0.0
        ts = pd.Timestamp(date_str)
        if ts in self._close_matrix.index:
            val = self._close_matrix.at[ts, permno]
            return float(val) if pd.notna(val) else 0.0
        # Fallback: find closest date <= ts using fast O(log N) numpy search
        idx = np.searchsorted(self._dates, np.datetime64(ts), side='right')
        # If idx == 0, it means all dates are > ts, so no previous date exists
        if idx > 0:
            last_date = self._dates[idx - 1]
            val = self._close_matrix.at[last_date, permno]
            return float(val) if pd.notna(val) else 0.0
        return 0.0
    
    def get_price_history_df(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get price history as a DataFrame with 'close' column. Fast slice from matrix using searchsorted."""
        permno = self.get_permno(ticker)
        if permno is None or permno not in self._close_matrix.columns:
            return pd.DataFrame()
        
        ts_start = pd.Timestamp(start_date)
        ts_end = pd.Timestamp(end_date)
        
        # O(log N) fast integer positional lookup instead of O(N) boolean mask
        idx_start = np.searchsorted(self._dates, np.datetime64(ts_start), side='left')
        idx_end = np.searchsorted(self._dates, np.datetime64(ts_end), side='right')
        
        if idx_start >= idx_end:
            return pd.DataFrame()
            
        # Fast positional slice using .iloc
        closes = self._close_matrix.iloc[idx_start:idx_end, self._close_matrix.columns.get_loc(permno)].dropna()
        
        if closes.empty:
            return pd.DataFrame()
        
        return pd.DataFrame({'close': closes})
    
    def get_returns_series(self, ticker: str, start_date: str, end_date: str) -> pd.Series:
        """Get daily returns for a ticker in a date range using fast searchsorted slice."""
        permno = self.get_permno(ticker)
        if permno is None or permno not in self._close_matrix.columns:
            return pd.Series(dtype=float)
        
        ts_start = pd.Timestamp(start_date)
        ts_end = pd.Timestamp(end_date)
        
        # O(log N) fast integer positional lookup
        idx_start = np.searchsorted(self._dates, np.datetime64(ts_start), side='left')
        idx_end = np.searchsorted(self._dates, np.datetime64(ts_end), side='right')
        
        if idx_start >= idx_end:
            return pd.Series(dtype=float)
            
        closes = self._close_matrix.iloc[idx_start:idx_end, self._close_matrix.columns.get_loc(permno)].dropna()
        
        if len(closes) < 2:
            return pd.Series(dtype=float)
        
        return closes.pct_change().dropna()


def calculate_risk_limits_fast(
    date_str: str,
    tickers: list[str],
    price_matrix: PriceMatrix,
    portfolio,
    current_portfolio_value: float,
    disable_risk_manager: bool = False
) -> tuple[dict, dict]:
    """
    Fast, inlined version of risk_management_agent.
    Uses PriceMatrix for O(1) lookups instead of per-ticker API calls.
    Returns (risk_limits, current_prices).
    """
    if disable_risk_manager:
        prices = {}
        for t in tickers:
            prices[t] = price_matrix.get_close(t, date_str)
        limits = {t: current_portfolio_value for t in tickers}
        return limits, prices
    
    dt_end = pd.Timestamp(date_str)
    dt_start = dt_end - pd.Timedelta(days=90)
    start_str = dt_start.strftime("%Y-%m-%d")
    
    current_prices = {}
    volatility_data = {}
    returns_by_ticker = {}
    
    for ticker in tickers:
        price = price_matrix.get_close(ticker, date_str)
        current_prices[ticker] = price
        
        if price <= 0:
            volatility_data[ticker] = {"annualized_volatility": 0.05 * np.sqrt(252)}
            continue
        
        returns = price_matrix.get_returns_series(ticker, start_str, date_str)
        
        if len(returns) >= 2:
            daily_vol = returns.std()
            ann_vol = daily_vol * np.sqrt(252)
            volatility_data[ticker] = {
                "annualized_volatility": float(ann_vol) if not np.isnan(ann_vol) else 0.25
            }
            if len(returns) > 0:
                returns_by_ticker[ticker] = returns
        else:
            volatility_data[ticker] = {"annualized_volatility": 0.25}
    
    # Correlation matrix
    correlation_matrix = None
    if len(returns_by_ticker) >= 2:
        try:
            returns_df = pd.DataFrame(returns_by_ticker).dropna(how="any")
            if returns_df.shape[1] >= 2 and returns_df.shape[0] >= 5:
                correlation_matrix = returns_df.corr()
        except Exception:
            pass
    
    # Active positions
    active_positions = {
        t for t, pos in portfolio.get_positions().items()
        if abs(pos.get("long", 0) - pos.get("short", 0)) > 0
    }
    
    # Calculate limits
    limits = {}
    for ticker in tickers:
        ann_vol = volatility_data.get(ticker, {}).get("annualized_volatility", 0.25)
        vol_limit_pct = _volatility_adjusted_limit(ann_vol)
        
        corr_multiplier = 1.0
        if correlation_matrix is not None and ticker in correlation_matrix.columns:
            comparable = [t for t in active_positions if t in correlation_matrix.columns and t != ticker]
            if not comparable:
                comparable = [t for t in correlation_matrix.columns if t != ticker]
            if comparable:
                series = correlation_matrix.loc[ticker, comparable].dropna()
                if len(series) > 0:
                    avg_corr = float(series.mean())
                    corr_multiplier = _correlation_multiplier(avg_corr)
        
        combined_pct = vol_limit_pct * corr_multiplier
        limits[ticker] = current_portfolio_value * combined_pct
    
    return limits, current_prices


def _volatility_adjusted_limit(annualized_volatility: float) -> float:
    """Same logic as risk_manager.py's calculate_volatility_adjusted_limit."""
    base_limit = 0.40
    if annualized_volatility < 0.15:
        vol_multiplier = 1.25
    elif annualized_volatility < 0.30:
        vol_multiplier = 1.0 - (annualized_volatility - 0.15) * 0.5
    elif annualized_volatility < 0.50:
        vol_multiplier = 0.75 - (annualized_volatility - 0.30) * 0.5
    else:
        vol_multiplier = 0.50
    vol_multiplier = max(0.25, min(1.25, vol_multiplier))
    return base_limit * vol_multiplier


def _correlation_multiplier(avg_correlation: float) -> float:
    """Same logic as risk_manager.py's calculate_correlation_multiplier."""
    if avg_correlation >= 0.80:
        return 0.70
    if avg_correlation >= 0.60:
        return 0.85
    if avg_correlation >= 0.40:
        return 1.00
    if avg_correlation >= 0.20:
        return 1.05
    return 1.10
