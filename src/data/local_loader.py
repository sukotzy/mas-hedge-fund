"""Local CSV data loader for financial data."""
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


class LocalDataLoader:
    """Load financial data from local CSV files."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the local data loader.
        
        Args:
            data_dir: Base directory for data files (default: 'data')
        """
        self.data_dir = Path(data_dir)
        self.prices_dir = self.data_dir / "prices"
        self.financial_metrics_dir = self.data_dir / "financial_metrics"
        self.news_dir = self.data_dir / "news"
        self.insider_trades_dir = self.data_dir / "insider_trades"
        
    def get_prices(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str
    ) -> List[Price]:
        """
        Load price data from local CSV file.
        
        CSV format:
        time,ticker,open,close,high,low,volume
        2024-10-01T00:00:00Z,AAPL,150.25,152.30,153.10,149.80,50000000
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            List of Price objects
        """
        csv_path = self.prices_dir / f"{ticker}.csv"
        
        if not csv_path.exists():
            print(f"Warning: Price data file not found: {csv_path}")
            return []
        
        try:
            df = pd.read_csv(csv_path)
            df['time'] = pd.to_datetime(df['time'])
            
            # Filter by date range
            mask = (df['time'] >= start_date) & (df['time'] <= end_date)
            df = df[mask]
            
            # Convert to Price objects
            prices = []
            for _, row in df.iterrows():
                prices.append(Price(
                    ticker=row['ticker'],
                    time=row['time'].isoformat(),
                    open=float(row['open']),
                    close=float(row['close']),
                    high=float(row['high']),
                    low=float(row['low']),
                    volume=int(row['volume'])
                ))
            
            return prices
            
        except Exception as e:
            print(f"Error loading prices for {ticker}: {e}")
            return []
    
    def get_financial_metrics(
        self,
        ticker: str,
        end_date: str,
        period: str = "ttm",
        limit: int = 10
    ) -> List[FinancialMetrics]:
        """
        Load financial metrics from local CSV file.
        
        CSV format:
        ticker,report_period,market_cap,pe_ratio,price_to_book_ratio,...
        AAPL,2024-09-30,2500000000000,28.5,45.2,...
        
        Args:
            ticker: Stock ticker symbol
            end_date: End date (YYYY-MM-DD)
            period: Period type (ttm, quarterly, annual)
            limit: Maximum number of records
            
        Returns:
            List of FinancialMetrics objects
        """
        csv_path = self.financial_metrics_dir / f"{ticker}.csv"
        
        if not csv_path.exists():
            print(f"Warning: Financial metrics file not found: {csv_path}")
            return []
        
        try:
            df = pd.read_csv(csv_path)
            df['report_period'] = pd.to_datetime(df['report_period'])
            
            # Filter by end date and limit
            df = df[df['report_period'] <= end_date]
            df = df.sort_values('report_period', ascending=False).head(limit)
            
            # Convert to FinancialMetrics objects
            metrics = []
            for _, row in df.iterrows():
                metrics.append(FinancialMetrics(
                    ticker=row['ticker'],
                    report_period=row['report_period'].strftime('%Y-%m-%d'),
                    period=period,
                    currency=row.get('currency', 'USD'),
                    market_cap=float(row.get('market_cap', 0)) if pd.notna(row.get('market_cap')) else None,
                    enterprise_value=float(row.get('enterprise_value', 0)) if pd.notna(row.get('enterprise_value')) else None,
                    price_to_earnings_ratio=float(row.get('pe_ratio', 0)) if pd.notna(row.get('pe_ratio')) else None,
                    price_to_book_ratio=float(row.get('price_to_book_ratio', 0)) if pd.notna(row.get('price_to_book_ratio')) else None,
                    price_to_sales_ratio=float(row.get('price_to_sales_ratio', 0)) if pd.notna(row.get('price_to_sales_ratio')) else None,
                    enterprise_value_to_ebitda_ratio=float(row.get('ev_to_ebitda', 0)) if pd.notna(row.get('ev_to_ebitda')) else None,
                    enterprise_value_to_revenue_ratio=None,
                    free_cash_flow_yield=None,
                    peg_ratio=None,
                    gross_margin=None,
                    operating_margin=None,
                    net_margin=None,
                    return_on_equity=float(row.get('return_on_equity', 0)) if pd.notna(row.get('return_on_equity')) else None,
                    return_on_assets=float(row.get('return_on_assets', 0)) if pd.notna(row.get('return_on_assets')) else None,
                    return_on_invested_capital=None,
                    asset_turnover=None,
                    inventory_turnover=None,
                    receivables_turnover=None,
                    days_sales_outstanding=None,
                    operating_cycle=None,
                    working_capital_turnover=None,
                    current_ratio=float(row.get('current_ratio', 0)) if pd.notna(row.get('current_ratio')) else None,
                    quick_ratio=float(row.get('quick_ratio', 0)) if pd.notna(row.get('quick_ratio')) else None,
                    cash_ratio=None,
                    operating_cash_flow_ratio=None,
                    debt_to_equity=float(row.get('debt_to_equity_ratio', 0)) if pd.notna(row.get('debt_to_equity_ratio')) else None,
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
                ))
            
            return metrics
            
        except Exception as e:
            print(f"Error loading financial metrics for {ticker}: {e}")
            return []
    
    def get_company_news(
        self,
        ticker: str,
        end_date: str,
        start_date: Optional[str] = None,
        limit: int = 1000
    ) -> List[CompanyNews]:
        """
        Load company news from local CSV file.
        
        CSV format:
        ticker,date,title,text,source,url
        AAPL,2024-10-01T10:00:00Z,Apple releases...,Apple announced...,Reuters,https://...
        
        Args:
            ticker: Stock ticker symbol
            end_date: End date
            start_date: Start date (optional)
            limit: Maximum number of records
            
        Returns:
            List of CompanyNews objects
        """
        csv_path = self.news_dir / f"{ticker}.csv"
        
        if not csv_path.exists():
            print(f"Warning: News file not found: {csv_path}")
            return []
        
        try:
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter by date range
            df = df[df['date'] <= end_date]
            if start_date:
                df = df[df['date'] >= start_date]
            
            df = df.sort_values('date', ascending=False).head(limit)
            
            # Convert to CompanyNews objects
            news = []
            for _, row in df.iterrows():
                news.append(CompanyNews(
                    ticker=row['ticker'],
                    date=row['date'].isoformat(),
                    title=row['title'],
                    text=row['text'],
                    source=row.get('source', ''),
                    url=row.get('url', '')
                ))
            
            return news
            
        except Exception as e:
            print(f"Error loading news for {ticker}: {e}")
            return []
    
    def get_insider_trades(
        self,
        ticker: str,
        end_date: str,
        start_date: Optional[str] = None,
        limit: int = 1000
    ) -> List[InsiderTrade]:
        """
        Load insider trades from local CSV file.
        
        CSV format:
        ticker,filing_date,transaction_date,owner_name,is_director,is_officer,shares,transaction_type
        AAPL,2024-09-30,2024-09-28,John Doe,True,False,10000,Purchase
        
        Args:
            ticker: Stock ticker symbol
            end_date: End date
            start_date: Start date (optional)
            limit: Maximum number of records
            
        Returns:
            List of InsiderTrade objects
        """
        csv_path = self.insider_trades_dir / f"{ticker}.csv"
        
        if not csv_path.exists():
            print(f"Warning: Insider trades file not found: {csv_path}")
            return []
        
        try:
            df = pd.read_csv(csv_path)
            df['filing_date'] = pd.to_datetime(df['filing_date'])
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
            
            # Filter by date range
            df = df[df['filing_date'] <= end_date]
            if start_date:
                df = df[df['filing_date'] >= start_date]
            
            df = df.sort_values('filing_date', ascending=False).head(limit)
            
            # Convert to InsiderTrade objects
            trades = []
            for _, row in df.iterrows():
                trades.append(InsiderTrade(
                    ticker=row['ticker'],
                    filing_date=row['filing_date'].isoformat(),
                    transaction_date=row['transaction_date'].isoformat(),
                    owner_name=row['owner_name'],
                    is_director=bool(row.get('is_director', False)),
                    is_officer=bool(row.get('is_officer', False)),
                    officer_title=row.get('officer_title', ''),
                    shares=int(row['shares']),
                    transaction_type=row['transaction_type']
                ))
            
            return trades
            
        except Exception as e:
            print(f"Error loading insider trades for {ticker}: {e}")
            return []
    
    def search_line_items(
        self,
        ticker: str,
        line_items: List[str],
        end_date: str,
        period: str = "ttm",
        limit: int = 10
    ) -> List[LineItem]:
        """
        Load line items from local CSV file.
        
        CSV format:
        ticker,report_period,line_item,value
        AAPL,2024-09-30,revenue,394000000000
        AAPL,2024-09-30,net_income,97000000000
        
        Args:
            ticker: Stock ticker symbol
            line_items: List of line item names to search for
            end_date: End date
            period: Period type
            limit: Maximum number of periods
            
        Returns:
            List of LineItem objects
        """
        csv_path = self.financial_metrics_dir / f"{ticker}_line_items.csv"
        
        if not csv_path.exists():
            print(f"Warning: Line items file not found: {csv_path}")
            return []
        
        try:
            df = pd.read_csv(csv_path)
            df['report_period'] = pd.to_datetime(df['report_period'])
            
            # Filter by line items and date
            df = df[df['line_item'].isin(line_items)]
            df = df[df['report_period'] <= end_date]
            df = df.sort_values('report_period', ascending=False)
            
            # Get unique periods, limited by limit
            unique_periods = df['report_period'].unique()[:limit]
            df = df[df['report_period'].isin(unique_periods)]
            
            # Convert to LineItem objects
            items = []
            for _, row in df.iterrows():
                items.append(LineItem(
                    ticker=row['ticker'],
                    report_period=row['report_period'].strftime('%Y-%m-%d'),
                    period=period,
                    line_item=row['line_item'],
                    value=float(row['value']) if pd.notna(row['value']) else None
                ))
            
            return items
            
        except Exception as e:
            print(f"Error loading line items for {ticker}: {e}")
            return []
    
    def get_market_cap(
        self,
        ticker: str,
        end_date: str
    ) -> float:
        """
        Get market cap from financial metrics CSV file.
        
        Args:
            ticker: Stock ticker symbol
            end_date: End date
            
        Returns:
            Market cap value or None if not found
        """
        # Get the most recent financial metrics
        metrics = self.get_financial_metrics(ticker, end_date, limit=1)
        
        if metrics and len(metrics) > 0:
            return metrics[0].market_cap
        
        return None


# Global instance
_local_loader: Optional[LocalDataLoader] = None


def get_local_loader(data_dir: str = "data") -> LocalDataLoader:
    """Get or create the global local data loader instance."""
    global _local_loader
    if _local_loader is None:
        _local_loader = LocalDataLoader(data_dir)
    return _local_loader
