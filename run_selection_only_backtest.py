"""
Selection-Only Benchmark (Ablation Experiment) — Optimized
===========================================================
Bypasses the Allocator (LLM) layer entirely.
Uses Selection Layer confidence scores directly as consensus values,
normalizes them into weights, and feeds into the QP optimizer.

OPTIMIZATION: Pre-loads the entire OHLCV matrix into memory once,
builds a ticker→close price lookup, and inlines risk calculations.
No per-ticker API calls. Runs ~2264 days in minutes, not hours.

Usage:
  python run_selection_only_backtest.py --start-date 2016-01-04 --end-date 2024-12-31
"""
import json
import logging
import os
import sys
from pathlib import Path

# Fix python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

# Load Env Vars
load_dotenv()
os.environ["USE_LOCAL_DATA"] = "true"

from src.utils.optimizer_utils import calculate_optimal_portfolio
from src.backtesting.portfolio import Portfolio
from src.backtesting.trader import TradeExecutor
from src.backtesting.valuation import calculate_portfolio_value

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Pre-loaded Price Matrix (replaces per-ticker get_price_data calls)
# =============================================================================
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
        
        # Also keep volume for potential use
        self._volume_matrix = ohlcv.pivot_table(
            index='date', columns='permno', values='vol', aggfunc='last'
        )
        
        # Date index as sorted array for fast bisect
        self._dates = self._close_matrix.index.values
        
        logger.info(f"Price matrix loaded: {self._close_matrix.shape[0]} dates x {self._close_matrix.shape[1]} permnos")
    
    def get_permno(self, ticker: str) -> int | None:
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
        # Fallback: find closest date <= ts
        mask = self._close_matrix.index <= ts
        if mask.any():
            last_date = self._close_matrix.index[mask][-1]
            val = self._close_matrix.at[last_date, permno]
            return float(val) if pd.notna(val) else 0.0
        return 0.0
    
    def get_price_history_df(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get price history as a DataFrame with 'close' column. Fast slice from matrix."""
        permno = self.get_permno(ticker)
        if permno is None or permno not in self._close_matrix.columns:
            return pd.DataFrame()
        
        ts_start = pd.Timestamp(start_date)
        ts_end = pd.Timestamp(end_date)
        
        mask = (self._close_matrix.index >= ts_start) & (self._close_matrix.index <= ts_end)
        closes = self._close_matrix.loc[mask, permno].dropna()
        
        if closes.empty:
            return pd.DataFrame()
        
        return pd.DataFrame({'close': closes})
    
    def get_returns_series(self, ticker: str, start_date: str, end_date: str) -> pd.Series:
        """Get daily returns for a ticker in a date range."""
        permno = self.get_permno(ticker)
        if permno is None or permno not in self._close_matrix.columns:
            return pd.Series(dtype=float)
        
        ts_start = pd.Timestamp(start_date)
        ts_end = pd.Timestamp(end_date)
        
        mask = (self._close_matrix.index >= ts_start) & (self._close_matrix.index <= ts_end)
        closes = self._close_matrix.loc[mask, permno].dropna()
        
        if len(closes) < 2:
            return pd.Series(dtype=float)
        
        return closes.pct_change().dropna()


# =============================================================================
# Inlined Risk Manager (replaces the agent-based risk_management_agent)
# =============================================================================
def calculate_risk_limits_fast(
    date_str: str,
    tickers: list[str],
    price_matrix: PriceMatrix,
    portfolio: Portfolio,
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
            volatility_data[ticker] = {
                "annualized_volatility": 0.05 * np.sqrt(252)
            }
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
            volatility_data[ticker] = {
                "annualized_volatility": 0.25
            }
    
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
        
        # Volatility-adjusted limit
        vol_limit_pct = _volatility_adjusted_limit(ann_vol)
        
        # Correlation adjustment
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


# =============================================================================
# Core Logic
# =============================================================================
def get_dynamic_rf_rate(date_str: str, rf_df: pd.DataFrame) -> float:
    """Fetch the risk free rate for the given date, or ffill if missing."""
    if not rf_df.empty:
        ts = pd.Timestamp(date_str)
        if ts in rf_df.index:
            return float(rf_df.loc[ts, 'risk_free_rate'])
        else:
            past_rates = rf_df[rf_df.index <= ts]
            if not past_rates.empty:
                return float(past_rates.iloc[-1]['risk_free_rate'])
    return 0.05 / 252


def selection_tasks_to_consensus(tasks: list[dict]) -> dict[str, float]:
    """
    Convert Selection Layer tasks into consensus values.
    Consensus = signed confidence: positive for 'long', negative for 'short'.
    """
    consensus = {}
    for task in tasks:
        ticker = task.get("ticker")
        action = task.get("action", "long")
        confidence = task.get("confidence", 0.0)
        
        if ticker is None or ticker == "CASH":
            continue
        
        if action == "short":
            consensus[ticker] = -abs(confidence)
        else:
            consensus[ticker] = abs(confidence)
    
    return consensus


def process_day_selection_only(
    date_str: str,
    tasks: list[dict],
    rf_rate: float,
    portfolio: Portfolio,
    executor: TradeExecutor,
    previous_consensus: dict,
    previous_prices: dict,
    price_matrix: PriceMatrix,
    disable_risk_manager: bool = False
):
    """
    Process a single day using selection-only consensus.
    Uses PriceMatrix for all price lookups (no API calls).
    """
    # 1. Derive consensus from selection tasks
    consensus_values = selection_tasks_to_consensus(tasks)
    
    tickers_from_tasks = [t for t in consensus_values.keys() if t != "CASH"]
    
    # Include tickers with existing positions
    active_tickers = set(tickers_from_tasks)
    for t, pos in portfolio.get_positions().items():
        if pos["long"] > 0 or pos["short"] > 0:
            active_tickers.add(t)
    
    # 2. Get prices from pre-loaded matrix (O(1) per ticker)
    current_prices = {}
    for t in active_tickers:
        price = price_matrix.get_close(t, date_str)
        if price > 0:
            current_prices[t] = price
        else:
            current_prices[t] = previous_prices.get(t, 0.0)
    
    # 3. Get price history for kinematic decay (fast slice from matrix)
    dt_end = pd.Timestamp(date_str)
    dt_start = dt_end - pd.Timedelta(days=15)
    
    prices_history = {}
    for t in active_tickers:
        prices_history[t] = price_matrix.get_price_history_df(
            t, dt_start.strftime("%Y-%m-%d"), date_str
        )

    # 4. Calculate portfolio value BEFORE optimization
    current_portfolio_value = calculate_portfolio_value(portfolio, current_prices)
    
    # 5. Get Risk Limits (fast inlined version)
    risk_limits, rm_prices = calculate_risk_limits_fast(
        date_str, list(active_tickers), price_matrix,
        portfolio, current_portfolio_value, disable_risk_manager
    )
    
    # Use risk manager prices if available
    for t in active_tickers:
        if t in rm_prices and rm_prices[t] > 0:
            current_prices[t] = rm_prices[t]
    
    # 6. Extract previous holdings
    previous_holdings = {}
    positions_state = portfolio.get_positions()
    for t in active_tickers:
        if t in positions_state:
            pos = positions_state[t]
            net_holding = pos.get("long", 0) - pos.get("short", 0)
            previous_holdings[t] = net_holding
        else:
            previous_holdings[t] = 0

    # 7. Optimization (reuses the same kinematic decay + QP pipeline)
    optimal_shares, adjusted_consensus = calculate_optimal_portfolio(
        today_consensus=consensus_values,
        previous_consensus=previous_consensus,
        previous_holdings=previous_holdings,
        prices_history=prices_history,
        risk_limits=risk_limits,
        initial_capital=current_portfolio_value,
        risk_free_rate=rf_rate,
        use_risk_manager=not disable_risk_manager
    )
    
    # 8. Execute Delta Trades
    executed_trades = []
    for t in active_tickers:
        target_net_shares = optimal_shares.get(t, 0.0)
        net_holding = previous_holdings.get(t, 0)
        delta = target_net_shares - net_holding
        
        delta = int(round(delta))
        price = current_prices.get(t, 0.0)
        
        action = None
        quantity = 0
        
        if net_holding >= 0:
            if delta > 0:
                action = "buy"
                quantity = delta
            elif delta < 0:
                action = "sell"
                quantity = abs(delta)
        else:
            if delta < 0:
                action = "short"
                quantity = abs(delta)
            elif delta > 0:
                action = "cover"
                quantity = delta
                
        if action and quantity > 0 and price > 0:
            executed_qty = executor.execute_trade(t, action, quantity, price, portfolio)
            if executed_qty > 0:
                executed_trades.append({
                    "ticker": t,
                    "action": action,
                    "quantity": executed_qty,
                    "price": price
                })
                
    # 9. Calculate NEW portfolio value AFTER execution
    updated_portfolio_value = calculate_portfolio_value(portfolio, current_prices)
    
    updated_positions = {t: {"long": pos["long"], "short": pos["short"]} 
                         for t, pos in portfolio.get_positions().items() 
                         if pos["long"] > 0 or pos["short"] > 0}
                         
    # 10. Update price memory
    previous_prices.update(current_prices)
    
    return {
        "date": date_str,
        "portfolio_value": updated_portfolio_value,
        "executed_trades": executed_trades,
        "updated_holdings": updated_positions,
        "consensus": consensus_values,
        "adjusted_consensus": adjusted_consensus,
        "prices": current_prices,
        "risk_limits": risk_limits,
        "optimal_shares": optimal_shares,
        "objective_cash_constant": current_portfolio_value * rf_rate
    }


def main():
    parser = argparse.ArgumentParser(description="Selection-Only Benchmark Backtest (Ablation)")
    parser.add_argument("--start-date", type=str, default="2016-01-04", help="Start Date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2024-12-31", help="End Date (YYYY-MM-DD)")
    parser.add_argument("--input-file", type=str, default="data/processed/daily_candidates_with_hint.parquet",
                        help="Path to daily candidates parquet")
    parser.add_argument("--output-file", type=str, default="data/backtests_with_risk_manager/selection_only_benchmark.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--initial-cash", type=float, default=100000.0, help="Initial portfolio cash")
    parser.add_argument("--margin-requirement", type=float, default=0.5, help="Margin requirement")
    parser.add_argument("--disable-risk-manager", action="store_true", help="Disable Risk Manager")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}. Please run run_selection.py first.")
        return

    # 1. Load candidates
    logger.info(f"Loading candidates from {input_path}...")
    df_candidates = pd.read_parquet(input_path)
    
    # Filter by date range
    all_dates = sorted(df_candidates.index)
    target_dates = [d for d in all_dates 
                    if args.start_date <= pd.Timestamp(d).strftime("%Y-%m-%d") <= args.end_date]
    
    logger.info(f"Processing {len(target_dates)} trading days ({args.start_date} to {args.end_date})...")

    # 2. Collect all unique tickers
    all_tickers = set()
    for date in target_dates:
        tasks = json.loads(df_candidates.loc[date, "tasks"])
        for t in tasks:
            ticker = t.get("ticker")
            if ticker and ticker != "CASH":
                all_tickers.add(ticker)
    
    logger.info(f"Universe size: {len(all_tickers)} unique tickers across all days.")
    
    # 3. Pre-load price matrix (ONE TIME, ~10 seconds)
    price_matrix = PriceMatrix()
    
    # 4. Initialize portfolio
    portfolio = Portfolio(
        tickers=list(all_tickers),
        initial_cash=args.initial_cash,
        margin_requirement=args.margin_requirement
    )
    executor = TradeExecutor()
    
    # 5. Load RF Data
    rf_file = Path("data/processed/daily_risk_free_rates.parquet")
    if rf_file.exists():
        rf_df = pd.read_parquet(rf_file)
    else:
        logger.warning(f"RF data not found at {rf_file}, using fallback 5% annual.")
        rf_df = pd.DataFrame()
    
    # Memory state
    previous_consensus = {}
    previous_prices = {}
    processed_count = 0
    
    print("=========================================================================")
    print("🚀 Selection-Only Benchmark Backtest (Ablation Experiment) 🚀")
    print("=========================================================================")
    
    # Open output file for immediate writing (OOM prevention)
    with open(output_path, "w") as out_f:
        for date in tqdm(target_dates, desc="Selection-Only Backtest"):
            date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
            
            try:
                tasks_json = df_candidates.loc[date, "tasks"]
                tasks = json.loads(tasks_json)
                
                rf_rate = get_dynamic_rf_rate(date_str, rf_df)
                
                res = process_day_selection_only(
                    date_str=date_str,
                    tasks=tasks,
                    rf_rate=rf_rate,
                    portfolio=portfolio,
                    executor=executor,
                    previous_consensus=previous_consensus,
                    previous_prices=previous_prices,
                    price_matrix=price_matrix,
                    disable_risk_manager=args.disable_risk_manager
                )
                
                # Write IMMEDIATELY to prevents results list from growing too large
                out_f.write(json.dumps(res) + "\n")
                # REMOVED: out_f.flush() - This was causing a massive I/O bottleneck
                processed_count += 1
                
                # Update memory for next day
                previous_consensus = res["adjusted_consensus"].copy()
                
                # Apply daily interest to cash balance
                interest_added = portfolio.add_cash_interest(rf_rate)
                if interest_added > 0:
                    # Only log occasionally in long runs to avoid console bloat
                    if processed_count % 20 == 0:
                        logger.info(f"[{date_str}] Interest added to cash: ${interest_added:,.2f}")
                
            except Exception as e:
                logger.error(f"Error processing {date_str}: {e}")
                import traceback
                traceback.print_exc()
    
    final_value = calculate_portfolio_value(portfolio, previous_prices)
    
    print("=========================================================================")
    print(f"🏆 Selection-Only Benchmark Complete!")
    print(f"   Days processed: {processed_count}")
    print(f"   Initial Value:  ${args.initial_cash:,.2f}")
    print(f"   Final Value:    ${final_value:,.2f}")
    print(f"   Total Return:   {((final_value / args.initial_cash) - 1) * 100:.2f}%")
    print(f"   Output:         {output_path}")
    print("=========================================================================")


if __name__ == "__main__":
    main()
