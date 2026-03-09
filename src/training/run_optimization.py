import json
import logging
import os
import sys
from pathlib import Path

# Fix python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
from scipy.optimize import linprog
import numpy as np
import datetime
from tqdm import tqdm
from dotenv import load_dotenv

# Load Env Vars
load_dotenv()
os.environ["USE_LOCAL_DATA"] = "true"

from src.market.betting_market import BettingMarket
from src.schemas import Bet, MarketSignal
from src.agents.risk_manager import risk_management_agent
from src.tools.api import get_prices, prices_to_df, get_price_data
from src.utils.optimizer_utils import calculate_optimal_portfolio

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Run LP is handled by solve_optimization_lp in optimizer_utils now

def get_risk_limits(date_str: str, tickers: list[str]):
    """
    Mock the state and run the risk manager to get dynamic limits.
    """
    # Create empty portfolio with initial capital
    portfolio = {
        "cash": 100000.0,
        "margin_requirement": 0.5,
        "margin_used": 0.0,
        "positions": {},
        "realized_gains": {}
    }
    
    state = {
        "messages": [],
        "data": {
            "tickers": tickers,
            "start_date": date_str,
            "end_date": date_str,
            "portfolio": portfolio,
            "analyst_signals": {}
        },
        "metadata": {"show_reasoning": False}
    }
    
    # End date is date, Start date is lookback for volatility (usually 60 days)
    dt_end = pd.Timestamp(date_str)
    dt_start = dt_end - pd.Timedelta(days=90)
    state["data"]["start_date"] = dt_start.strftime("%Y-%m-%d")
    
    result_state = risk_management_agent(state)
    risk_analysis = result_state["data"]["analyst_signals"]["risk_management_agent"]
    
    limits = {}
    prices = {}
    for t, data in risk_analysis.items():
        limits[t] = data["remaining_position_limit"]
        prices[t] = data["current_price"]
        
    return limits, prices
    

def get_dynamic_rf_rate(date_str: str, rf_df: pd.DataFrame) -> float:
    """Fetch the risk free rate for the given date, or ffill if missing."""
    if not rf_df.empty:
        # Convert date_str to timestamp to query
        ts = pd.Timestamp(date_str)
        if ts in rf_df.index:
            return float(rf_df.loc[ts, 'risk_free_rate'])
        else:
            # Get latest available
            past_rates = rf_df[rf_df.index <= ts]
            if not past_rates.empty:
                return float(past_rates.iloc[-1]['risk_free_rate'])
    return 0.05 / 252 # Fallback


def process_day(day_data: dict, rf_rate: float, previous_holdings: dict, previous_consensus: dict):
    date_str = day_data["date"]
    tickers = day_data["tickers"]
    
    # 1. Reconstruct Betting Market
    market = BettingMarket()
    agent_names = ["fundamental", "technical", "valuation", "sentiment"]
    
    for agent in agent_names:
        if agent not in day_data:
            continue
        decision = day_data[agent]
        # Instead of generic 100.0, this will naturally read the updated agent_capital 
        # that the new simulator pipeline writes into "starting_capital".
        allocator_capital = decision.get("starting_capital", 100.0) 
        
        allocations = decision.get("allocations", [])
        for alloc in allocations:
            ticker = alloc.get("ticker")
            direction = alloc.get("direction")
            amount = alloc.get("amount", 0.0)
            
            # Map string to Enum
            if direction == "long":
                sig = MarketSignal.LONG
            elif direction == "short":
                sig = MarketSignal.SHORT
            else:
                continue # Skip neutral/cash for now if it doesn't map directly
                
            # Create Bet
            bet_amt = (amount / 100.0) * allocator_capital
            
            b = Bet(
                ticker=ticker,
                direction=sig,
                amount=bet_amt,
                conviction=1.0,
                reasoning="Extracted from JSONL"
            )
            market.place_bet(b)
            
    consensus_values = market.calculate_consensus()
    
    # 2. Get Risk Limits & Prices
    rm_tickers = [t for t in tickers if t != "CASH"]
    risk_limits, prices = get_risk_limits(date_str, rm_tickers)
    
    # 3. Fetch 15-day trailing prices for decay logic
    prices_history = {}
    dt_end = pd.Timestamp(date_str)
    dt_start = dt_end - pd.Timedelta(days=15)
    
    for t in rm_tickers:
        try:
            df = get_price_data(t, dt_start.strftime("%Y-%m-%d"), date_str)
            prices_history[t] = df
        except Exception as e:
            logger.warning(f"Failed to fetch price data for {t} from {dt_start.strftime('%Y-%m-%d')} to {date_str}: {e}")
            prices_history[t] = pd.DataFrame()
    
    # 4. Optimization
    # The new pipeline will pass the actual calculated fund_wealth. If missing, defaults to 100k.
    fund_wealth = day_data.get("fund_wealth", 100000.0) 
    
    optimal_shares, adjusted_consensus = calculate_optimal_portfolio(
        today_consensus=consensus_values,
        previous_consensus=previous_consensus,
        previous_holdings=previous_holdings,
        prices_history=prices_history,
        risk_limits=risk_limits,
        initial_capital=fund_wealth,
        risk_free_rate=rf_rate
    )
    
    return {
        "date": date_str,
        "consensus": consensus_values,
        "adjusted_consensus": adjusted_consensus,
        "prices": prices,
        "risk_limits": risk_limits,
        "optimal_shares": optimal_shares,
        "fund_wealth": fund_wealth,
        "objective_cash_constant": fund_wealth * rf_rate
    }


def main():
    # Will read from the enriched output of the upstream wealth simulator
    input_file = Path("data/enriched_decisions.jsonl")
    output_file = Path("data/optimization_results_final.jsonl")
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}. Please run simulate_wealth_trajectory.py first.")
        return
        
    logger.info(f"Processing optimization on {input_file}")
    
    with open(input_file, "r") as f:
        lines = f.readlines()
        
    results = []
    
    # Load RF Data
    rf_file = Path("data/processed/daily_risk_free_rates.parquet")
    if rf_file.exists():
        rf_df = pd.read_parquet(rf_file)
    else:
        logger.warning(f"RF data not found at {rf_file}, using fallback 5% annual.")
        rf_df = pd.DataFrame()
    
    # Memory state for the optimizer
    previous_holdings = {}
    previous_consensus = {}
    
    for line in tqdm(lines):
        day_data = json.loads(line)
        try:
            date_str = day_data.get('date')
            rf_rate = get_dynamic_rf_rate(date_str, rf_df)
            
            # Pure functional optimization pass over state
            res = process_day(day_data, rf_rate=rf_rate, 
                              previous_holdings=previous_holdings,
                              previous_consensus=previous_consensus) 
            results.append(res)
            
            # Update memory state after each day
            previous_holdings = res["optimal_shares"].copy()
            previous_consensus = res["adjusted_consensus"].copy()
            
        except Exception as e:
            logger.error(f"Error processing {day_data.get('date')}: {e}")
            
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    logger.info("Optimization complete.")

if __name__ == "__main__":
    main()
