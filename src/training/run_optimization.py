import json
import logging
import os
from pathlib import Path
import pandas as pd
from scipy.optimize import linprog
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# Load Env Vars
load_dotenv()
os.environ["USE_LOCAL_DATA"] = "true"

from src.market.betting_market import BettingMarket
from src.schemas import Bet, MarketSignal
from src.agents.risk_manager import risk_management_agent
from src.tools.api import get_prices, prices_to_df

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def solve_optimization(consensus_values: dict[str, float], 
                       current_prices: dict[str, float], 
                       risk_limits: dict[str, float], 
                       initial_capital: float, 
                       risk_free_rate: float):
    """
    Runs LP Optimization.
    Objective: Maximize Sum(Value_i * Action_i) + (Initial_Capital * RiskFreeRate)
    Constraints:
    1. |Action_i * Price_i| <= Risk_Limit_i
    2. Sum(Action_i * Price_i) = 0 (Self-financing via collateral)
    """
    tickers = list(current_prices.keys())
    # Include CASH in tickers if it isn't already to ensure it gets an LP variable
    if "CASH" not in tickers:
        tickers.append("CASH")
        
    # Let x = [Action_T1, Action_T2, ..., Action_CASH] 
    # For stocks, x is number of shares. For CASH, x is dollar amount.
    # Objective: maximize c^T x + (Initial_Capital * Rf)
    # Objective: maximize c^T x + (Initial_Capital * Rf)
    # Notice that (Initial_Capital * Rf) is a constant, so we maximize c^T x.
    # We must minimize -c^T x for linprog.
    
    c = []
    prices = []
    bounds = []
    
    for t in tickers:
        value = consensus_values.get(t, 0.0)
        c.append(-value) # Minimize negative value
        
        if t == "CASH":
            price = 1.0 # Cash is always $1
            prices.append(price)
            # Cash can be anywhere from -Initial_Capital to +Initial_Capital 
            # (or unbound if we just want the equality constraint to handle it)
            # But let's bound it to available collateral for safety
            bounds.append((-initial_capital, initial_capital))
        else:
            price = current_prices[t]
            prices.append(price)
            
            limit_usd = risk_limits.get(t, 0.0)
            
            # Action is in shares. Bounds in shares: (-limit / price, +limit / price)
            if price > 0:
                max_shares = limit_usd / price
                bounds.append((-max_shares, max_shares))
            else:
                bounds.append((0, 0))
            
    if not c or all(val == 0.0 for val in c):
        logger.warning("All consensus values are zero (no bets). Returning zero allocations.")
        return {t: 0.0 for t in tickers}
        
    # Equality Constraint: Sum(Action_i * Price_i) = 0
    A_eq = [prices]
    b_eq = [0.0]
    
    # Run LP
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    results = {}
    if res.success:
        for i, t in enumerate(tickers):
            results[t] = res.x[i]
    else:
        logger.warning(f"Optimization failed: {res.message}")
        for t in tickers:
            results[t] = 0.0
                
    return results

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


def process_day(day_data: dict, rf_rate: float):
    date_str = day_data["date"]
    tickers = day_data["tickers"]
    
    # 1. Reconstruct Betting Market
    market = BettingMarket()
    agent_names = ["fundamental", "technical", "valuation", "sentiment"]
    
    for agent in agent_names:
        if agent not in day_data:
            continue
        decision = day_data[agent]
        allocator_capital = decision.get("starting_capital", 100.0) # Used as weight
        
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
            # Use raw amount (0-100) or scale it by capital. 
            # The prompt output amount is % of capital usually.
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
    # Filter out CASH from tickers sent to Risk Manager since RM fetches API prices
    rm_tickers = [t for t in tickers if t != "CASH"]
    risk_limits, prices = get_risk_limits(date_str, rm_tickers)
    
    # 3. Optimization
    optimal_shares = solve_optimization(
        consensus_values=consensus_values,
        current_prices=prices,
        risk_limits=risk_limits,
        initial_capital=100000.0,
        risk_free_rate=rf_rate
    )
    
    return {
        "date": date_str,
        "consensus": consensus_values,
        "prices": prices,
        "risk_limits": risk_limits,
        "optimal_shares": optimal_shares,
        "objective_cash_constant": 100000.0 * rf_rate
    }


def main():
    input_file = Path("data/training_output_deepseek_2020h1/no_hint_wealth/2020_01.jsonl")
    output_file = Path("data/optimization_results_2020_01.jsonl")
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
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
    
    for line in tqdm(lines):
        day_data = json.loads(line)
        try:
            date_str = day_data.get('date')
            rf_rate = get_dynamic_rf_rate(date_str, rf_df)
            
            res = process_day(day_data, rf_rate=rf_rate) 
            results.append(res)
        except Exception as e:
            logger.error(f"Error processing {day_data.get('date')}: {e}")
            
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    logger.info("Optimization complete.")

if __name__ == "__main__":
    main()
