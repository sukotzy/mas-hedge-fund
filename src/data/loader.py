"""Local Data Loader for Parquet-based WRDS Data."""
import os
import pandas as pd
from pathlib import Path
from typing import List, Optional
from src.data.models import (
    Price,
    FinancialMetrics,
    CompanyNews,
    InsiderTrade,
    LineItem,
)
import numpy as np

class LocalDataLoader:
    """Load financial data from local Parquet files (WRDS Dump)."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        
        # Pre-load data tables
        self._load_tables()
        
    def _load_tables(self):
        """Load Parquet files into memory."""
        try:
            # 1. Constituents (Ticker <-> Permno)
            self.constituents = pd.read_parquet(self.raw_dir / "sp500_constituents.parquet")
            # Ensure dates are datetime
            self.constituents['start'] = pd.to_datetime(self.constituents['start'])
            self.constituents['ending'] = pd.to_datetime(self.constituents['ending'])
            
            # 2. OHLCV (Prices)
            self.ohlcv = pd.read_parquet(self.raw_dir / "sp500_ohlcv.parquet")
            self.ohlcv['date'] = pd.to_datetime(self.ohlcv['date'])
            
            # 3. CCM Links (Permno <-> GVKEY)
            self.ccm_links = pd.read_parquet(self.raw_dir / "ccm_links.parquet")
            # Links usually imply valid range, but for now we assume 1:1 dominant link for the period
            
            # 4. Fundamentals (comp.fundq)
            self.fundamentals = pd.read_parquet(self.raw_dir / "comp_fundq.parquet")
            if 'rdq' in self.fundamentals.columns:
                self.fundamentals['rdq'] = pd.to_datetime(self.fundamentals['rdq'])
            
            # 5. Ratios (firm_ratio)
            self.ratios = pd.read_parquet(self.raw_dir / "sp500_ratios_firm_ratio.parquet")
            self.ratios['date'] = pd.to_datetime(self.ratios['date']) # This is public_date
            
            # 6. Key Developments (News)
            try:
                self.keydev = pd.read_parquet(self.raw_dir / "sp500_keydev.parquet")
                self.keydev['announcedate'] = pd.to_datetime(self.keydev['announcedate'])
            except FileNotFoundError:
                print("LocalDataLoader Warning: sp500_keydev.parquet not found. News features will be disabled.")
                self.keydev = pd.DataFrame()

            # 7. Company Info (GICS Sectors)
            try:
                self.company_info = pd.read_parquet(self.raw_dir / "company_info.parquet")
                if 'gvkey' in self.company_info.columns:
                    self.company_info['gvkey'] = self.company_info['gvkey'].astype(str).str.zfill(6)
            except FileNotFoundError:
                print("LocalDataLoader Warning: company_info.parquet not found. Sector mapping disabled.")
                self.company_info = pd.DataFrame()
                
            print("LocalDataLoader: Successfully loaded WRDS Parquet files.")
            
        except FileNotFoundError as e:
            print(f"LocalDataLoader Warning: Missing Parquet file - {e}")
            # Initialize empty DFs to prevent crashes
            self.constituents = pd.DataFrame()
            self.ohlcv = pd.DataFrame()
            self.ccm_links = pd.DataFrame()
            self.fundamentals = pd.DataFrame()
            self.ratios = pd.DataFrame()
            self.company_info = pd.DataFrame()

    def get_permno(self, ticker: str, date: str) -> Optional[int]:
        """Resolve Ticker to Permno for a specific date."""
        if self.constituents.empty:
            return None
        
        target_date = pd.to_datetime(date)
        
        # Filter by ticker match
        # Note: Ticker in WRDS might be just symbol, or have suffix. We assume exact match for now.
        matches = self.constituents[self.constituents['ticker'] == ticker]
        
        # Filter by date range validity
        valid = matches[
            (matches['start'] <= target_date) & 
            (matches['ending'] >= target_date)
        ]
        
        if not valid.empty:
            return int(valid.iloc[0]['permno'])
        
        # Fallback: Check if it exists at all (maybe date mismatch), return most recent
        if not matches.empty:
            # Sort by ending desc
            matches = matches.sort_values('ending', ascending=False)
            return int(matches.iloc[0]['permno'])
            
        return None

    def get_gvkey(self, permno: int) -> Optional[str]:
        """Resolve Permno to GVKEY."""
        if self.ccm_links.empty:
            return None
        
        match = self.ccm_links[self.ccm_links['permno'] == permno]
        if not match.empty:
            return str(match.iloc[0]['gvkey']).zfill(6) # GVKEYs are 6 char strings
        return None

    def get_prices(self, ticker: str, start_date: str, end_date: str) -> List[Price]:
        """Get OHLCV data from Parquet."""
        permno = self.get_permno(ticker, end_date)
        if not permno or self.ohlcv.empty:
            return []
            
        # Filter for permno
        df = self.ohlcv[self.ohlcv['permno'] == permno].copy()
        
        # Filter Date Range
        mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
        df = df[mask].sort_values('date')
        
        prices = []
        for _, row in df.iterrows():
            prices.append(Price(
                ticker=ticker,
                time=row['date'].isoformat(),
                open=float(row['openprc']) if pd.notna(row['openprc']) else 0.0,
                close=float(row['prc']) if pd.notna(row['prc']) else 0.0, # prc is closing price
                high=float(row['askhi']) if pd.notna(row['askhi']) else 0.0,
                low=float(row['bidlo']) if pd.notna(row['bidlo']) else 0.0,
                volume=int(row['vol']) if pd.notna(row['vol']) else 0
            ))
        return prices

    def get_financial_metrics(self, ticker: str, end_date: str, period: str = "ttm", limit: int = 10) -> List[FinancialMetrics]:
        """Get fundamentals from Parquet (comp.fundq + firm_ratio)."""
        permno = self.get_permno(ticker, end_date)
        if not permno:
            return []
        
        gvkey = self.get_gvkey(permno)
        target_date = pd.to_datetime(end_date)
        
        # 1. Get Ratios (Monthly)
        ratios_df = pd.DataFrame()
        if not self.ratios.empty:
            ratios_df = self.ratios[
                (self.ratios['permno'] == permno) & 
                (self.ratios['date'] <= target_date)
            ].sort_values('date', ascending=False).head(limit)
        
        # 2. Get Fundamentals (Quarterly)
        fund_df = pd.DataFrame()
        if not self.fundamentals.empty and gvkey:
            # CRITICAL: Use rdq (release date) to filter
            fund_df = self.fundamentals[
                (self.fundamentals['gvkey'] == gvkey) & 
                (self.fundamentals['rdq'] <= target_date)
            ].sort_values('rdq', ascending=False).head(limit)
            
        if fund_df.empty:
            return []

        # Convert to list of objects
        # Strategy: Iterate through fundamentals (quarterly) and attach the closest available ratio
        metrics_list = []
        
        for _, fund_row in fund_df.iterrows():
            # Find closest ratio date that is <= fund release date (or just most recent)
            # Actually, we should probably return the most recent data points as snapshots.
            # firm_ratio is monthly, fundq is quarterly.
            
            # Let's use the fundq rows as the base "Report Period"
            report_date = fund_row['rdq']
            
            # Find matching ratio (closest before or equal to report_date)
            matched_ratio = pd.Series(dtype=float)
            if not ratios_df.empty:
                candidates = ratios_df[ratios_df['date'] <= report_date]
                if not candidates.empty:
                    matched_ratio = candidates.iloc[0] # Most recent relative to this report
            
            # Calculate metrics
            m = FinancialMetrics(
                ticker=ticker,
                report_period=report_date.strftime('%Y-%m-%d'),
                period=period,
                currency="USD",
                
                # Ratios (from firm_ratio)
                price_to_earnings_ratio=float(matched_ratio.get('pe_exi', 0)) if pd.notna(matched_ratio.get('pe_exi')) else None,
                price_to_book_ratio=float(matched_ratio.get('ptb', 0)) if pd.notna(matched_ratio.get('ptb')) else None,
                
                # Deep Fundamentals (from comp.fundq)
                return_on_equity=float(fund_row.get('return_on_equity', 0)) if pd.notna(fund_row.get('return_on_equity')) else None,
                net_margin=float(fund_row.get('net_margin', 0)) if pd.notna(fund_row.get('net_margin')) else None,
                operating_margin=float(fund_row.get('operating_margin', 0)) if pd.notna(fund_row.get('operating_margin')) else None,
                current_ratio=float(fund_row.get('current_ratio', 0)) if pd.notna(fund_row.get('current_ratio')) else None,
                debt_to_equity=float(fund_row.get('debt_to_equity', 0)) if pd.notna(fund_row.get('debt_to_equity')) else None,
                
                # Required but missing fields (Set to None)
                market_cap=None,
                enterprise_value=None,
                price_to_sales_ratio=None,
                enterprise_value_to_ebitda_ratio=None,
                enterprise_value_to_revenue_ratio=None,
                free_cash_flow_yield=None,
                peg_ratio=None,
                gross_margin=None,
                return_on_assets=None,
                return_on_invested_capital=None,
                asset_turnover=None,
                inventory_turnover=None,
                receivables_turnover=None,
                days_sales_outstanding=None,
                operating_cycle=None,
                working_capital_turnover=None,
                quick_ratio=None,
                cash_ratio=None,
                operating_cash_flow_ratio=None,
                debt_to_assets=None,
                interest_coverage=None,
                revenue_growth=None,
                earnings_growth=None,
                book_value_growth=None,
                earnings_per_share_growth=None,
                free_cash_flow_growth=None,
                operating_income_growth=None,
                ebitda_growth=None,
                payout_ratio=None,
                earnings_per_share=None,
                book_value_per_share=None,
                free_cash_flow_per_share=None,
            )
            metrics_list.append(m)
            
        return metrics_list

    # --- Legacy/Pass-through methods for non-WRDS data (News/Insider) ---
    # These still point to CSVs or need a new strategy in Phase 2. 
    # For now, return empty or implement basic CSV fallback if files exist.
    
    def get_company_news(self, ticker: str, end_date: str, start_date: Optional[str] = None, limit: int = 1000) -> List[CompanyNews]:
        """
        Get Key Developments (News) from WRDS Parquet.
        Maps KeyDev fields to CompanyNews model.
        """
        permno = self.get_permno(ticker, end_date)
        gvkey = self.get_gvkey(permno) if permno else None
        
        if not gvkey or self.keydev.empty:
            return []
            
        target_end_date = pd.to_datetime(end_date)
        start_filter = pd.Timestamp.min
        if start_date:
            start_filter = pd.to_datetime(start_date)
            
        # Filter KeyDev by gvkey and date range
        mask = (
            (self.keydev['gvkey'] == gvkey) &
            (self.keydev['announcedate'] <= target_end_date) &
            (self.keydev['announcedate'] >= start_filter)
        )
        
        # Sort desc by date, take limit
        df = self.keydev[mask].sort_values('announcedate', ascending=False).head(limit)
        
        news_list = []
        for _, row in df.iterrows():
            # Map KeyDev specific columns to our Schema
            # headline -> title
            # situation -> content (not in schema yet, but good to know)
            # keydeveventtypeid -> can be mapped to source or sentiment context
            
            news = CompanyNews(
                ticker=ticker,
                title=row['headline'],
                author="Capital IQ KeyDev", # Static author
                source=f"KeyDev Event {row['keydeveventtypeid']}", # Source = Event Type ID
                date=row['announcedate'].strftime('%Y-%m-%d'),
                url="", # No URL in this dataset
                sentiment=None # To be filled by Agent LLM
            )
            news_list.append(news)
            
        return news_list

    def get_insider_trades(self, ticker: str, end_date: str, start_date: Optional[str] = None, limit: int = 1000) -> List[InsiderTrade]:
        return []

    def search_line_items(self, ticker: str, line_items: List[str], end_date: str, period: str = "ttm", limit: int = 10) -> List[LineItem]:
        # Fundamental/Buffett agent uses this. We can map `comp.fundq` columns to line items!
        # Mapping: 'net_income' -> 'niq', 'revenue' -> 'revtq', 'total_assets' -> 'atq'
        permno = self.get_permno(ticker, end_date)
        gvkey = self.get_gvkey(permno) if permno else None
        
        if not gvkey or self.fundamentals.empty:
            return []
            
        target_date = pd.to_datetime(end_date)
        df = self.fundamentals[
            (self.fundamentals['gvkey'] == gvkey) & 
            (self.fundamentals['rdq'] <= target_date)
        ].sort_values('rdq', ascending=False).head(limit)
        
        column_map = {
            'net_income': 'niq',
            'revenue': 'revtq', 
            'total_assets': 'atq',
            'total_liabilities': 'ltq',
            'total_equity': 'seqq'
        }
        
        results = []
        for item_name in line_items:
            wrds_col = column_map.get(item_name)
            if wrds_col and wrds_col in df.columns:
                for _, row in df.iterrows():
                    val = row[wrds_col]
                    if pd.notna(val):
                        results.append(LineItem(
                            ticker=ticker,
                            report_period=row['rdq'].strftime('%Y-%m-%d'),
                            period=period,
                            currency="USD",
                            line_item=item_name,
                            value=float(val)
                        ))
        return results

    def get_market_cap(self, ticker: str, end_date: str) -> Optional[float]:
        # Calc from cshoq * price? or just return None for now
        return None

# Global instance pattern
_local_loader: Optional[LocalDataLoader] = None

def get_local_loader(data_dir: str = "data") -> LocalDataLoader:
    global _local_loader
    if _local_loader is None:
        _local_loader = LocalDataLoader(data_dir)
    return _local_loader
