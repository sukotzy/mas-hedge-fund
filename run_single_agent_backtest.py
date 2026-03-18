import json
import logging
import os
import sys
from pathlib import Path

# Fix python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import datetime
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

# Load Env Vars
load_dotenv()
os.environ["USE_LOCAL_DATA"] = "true"

from src.schemas import MarketSignal
from src.tools.api import get_prices, prices_to_df, get_price_data
from src.utils.optimizer_utils import calculate_optimal_portfolio
from src.backtesting.portfolio import Portfolio
from src.backtesting.trader import TradeExecutor
from src.backtesting.valuation import calculate_portfolio_value

# Optional fast mode
try:
    from src.utils.price_matrix import PriceMatrix, calculate_risk_limits_fast
except ImportError:
    PriceMatrix = None
    calculate_risk_limits_fast = None

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_risk_limits(date_str: str, tickers: list[str], portfolio: Portfolio, current_portfolio_value: float, disable_risk_manager: bool = False):
    if disable_risk_manager:
        limits = {t: current_portfolio_value for t in tickers}
        return limits, {}
    
    try:
        from src.agents.risk_manager import risk_management_agent
        rm_state = {"portfolio": portfolio, "current_portfolio_value": current_portfolio_value, "date": date_str, "tickers": tickers}
        rm_result = risk_management_agent(rm_state)
        return rm_result.get("position_limit", {}), rm_result.get("prices", {})
    except Exception as e:
        logger.error(f"Risk manager run failed: {e}")
        return {t: current_portfolio_value for t in tickers}, {}

def get_dynamic_rf_rate(date_str: str, rf_df: pd.DataFrame) -> float:
    if not rf_df.empty:
        ts = pd.Timestamp(date_str)
        if ts in rf_df.index:
            return float(rf_df.loc[ts, 'risk_free_rate'])
        else:
            past_rates = rf_df[rf_df.index <= ts]
            if not past_rates.empty:
                return float(past_rates.iloc[-1]['risk_free_rate'])
    return 0.05 / 252

def process_day(day_data: dict, rf_rate: float, portfolio: Portfolio, executor: TradeExecutor, previous_consensus: dict, previous_prices: dict, agent_name: str, disable_risk_manager: bool = False, turnover_penalty: float = 0.05, decay_mode: str = "harsh", segregate_capital: float = 0.0, transfer_rate: float = 1.0, enable_smoothing: bool = False, enable_safety_net: bool = False, max_daily_loss_pct: float = 0.25, price_matrix=None):
    date_str = day_data["date"]
    tickers = day_data["tickers"]
    rm_tickers = [t for t in tickers if t != "CASH"]
    
    active_tickers = set(rm_tickers)
    for t, pos in portfolio.get_positions().items():
        if pos["long"] > 0 or pos["short"] > 0:
            active_tickers.add(t)
            
    # Include yesterday's bet logic strictly for the target agent
    if agent_name in day_data:
        for alloc in day_data[agent_name].get("allocations", []):
            bet_ticker = alloc.get("ticker")
            if bet_ticker and bet_ticker != "CASH":
                active_tickers.add(bet_ticker)
    
    prices_history = {}
    dt_end = pd.Timestamp(date_str)
    dt_start = dt_end - pd.Timedelta(days=15)
    
    current_prices = {}
    if price_matrix is not None:
        for t in active_tickers:
            price = price_matrix.get_close(t, date_str)
            current_prices[t] = price if price > 0 else previous_prices.get(t, 0.0)
            prices_history[t] = price_matrix.get_price_history_df(
                t, dt_start.strftime("%Y-%m-%d"), date_str
            )
    else:
        for t in active_tickers:
            try:
                df = get_price_data(t, dt_start.strftime("%Y-%m-%d"), date_str)
                prices_history[t] = df
                if not df.empty and "close" in df.columns:
                    valid_closes = df["close"].dropna()
                    if not valid_closes.empty:
                        current_prices[t] = float(valid_closes.iloc[-1])
                    else:
                        current_prices[t] = previous_prices.get(t, 0.0)
                else:
                    current_prices[t] = previous_prices.get(t, 0.0)
            except Exception as e:
                logger.warning(f"Failed to load price for {t}: {e}")
                prices_history[t] = pd.DataFrame()
                current_prices[t] = previous_prices.get(t, 0.0)
                
    # Direct consensus construction from target agent
    consensus_values = {t: 0.0 for t in active_tickers}
    
    logger.info(f"\n[{date_str}] --- DAY START ---")
    current_portfolio_value = calculate_portfolio_value(portfolio, current_prices)
    logger.info(f"[{date_str}] Standalone Agent '{agent_name.upper()}' Total Capital: ${current_portfolio_value:,.2f}")
    
    if agent_name in day_data:
        decision = day_data[agent_name]
        allocations = decision.get("allocations", [])
        for alloc in allocations:
            ticker = alloc.get("ticker")
            direction = alloc.get("direction")
            amount = alloc.get("amount", 0.0) # 0 to 100
            
            if ticker in active_tickers:
                # Convert 0-100% amount to -1.0 to 1.0 signal
                signal = (amount / 100.0)
                if direction == "short":
                    signal = -signal
                elif direction == "long":
                    signal = signal
                else:
                    signal = 0.0
                
                consensus_values[ticker] = signal
                logger.info(f"    -> Signal: {direction.upper()} {ticker} | Weight: {amount}% | Translated Consensus Score: {signal:.4f}")

    if price_matrix is not None and calculate_risk_limits_fast is not None:
        risk_limits, rm_prices = calculate_risk_limits_fast(
            date_str, list(active_tickers), price_matrix,
            portfolio, current_portfolio_value, disable_risk_manager
        )
    else:
        risk_limits, rm_prices = get_risk_limits(date_str, list(active_tickers), portfolio, current_portfolio_value, disable_risk_manager)
    
    for t in active_tickers:
        if t in rm_prices and rm_prices[t] > 0:
            current_prices[t] = rm_prices[t]
            
    previous_holdings = {}
    positions_state = portfolio.get_positions()
    for t in active_tickers:
        if t in positions_state:
            pos = positions_state[t]
            previous_holdings[t] = pos.get("long", 0) - pos.get("short", 0)
        else:
            previous_holdings[t] = 0

    optimal_shares, adjusted_consensus = calculate_optimal_portfolio(
        today_consensus=consensus_values,
        previous_consensus=previous_consensus,
        previous_holdings=previous_holdings,
        prices_history=prices_history,
        risk_limits=risk_limits,
        initial_capital=current_portfolio_value,
        risk_free_rate=rf_rate,
        use_risk_manager=not disable_risk_manager,
        turnover_penalty=turnover_penalty,
        decay_mode=decay_mode,
        segregate_capital=segregate_capital
    )
    
    logger.info(f"[{date_str}] Target Shares:")
    non_zero_targets = {t: shares for t, shares in optimal_shares.items() if shares != 0}
    for t, shares in list(non_zero_targets.items())[:5]:
        logger.info(f"    -> {t}: target={shares} shares")
        
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
                
    updated_portfolio_value = calculate_portfolio_value(portfolio, current_prices)
    updated_positions = {t: {"long": pos["long"], "short": pos["short"]} 
                         for t, pos in portfolio.get_positions().items() 
                         if pos["long"] > 0 or pos["short"] > 0}
                         
    previous_prices.clear()
    for t in active_tickers:
        previous_prices[t] = current_prices.get(t, 0.0)
        
    pruned_adjusted_consensus = {t: v for t, v in adjusted_consensus.items() if t in active_tickers}
    
    return {
        "date": date_str,
        "portfolio_value": updated_portfolio_value,
        "executed_trades": executed_trades,
        "updated_holdings": updated_positions,
        "consensus": consensus_values,
        "adjusted_consensus": pruned_adjusted_consensus,
        "prices": rm_prices,
        "risk_limits": risk_limits,
        "optimal_shares": optimal_shares,
        "objective_cash_constant": current_portfolio_value * rf_rate
    }


def main():
    parser = argparse.ArgumentParser(description="Run Single-Agent Backtest (Ablation Study)")
    parser.add_argument("--agent", type=str, required=True, choices=["fundamental", "technical", "valuation", "sentiment"], help="Specific agent to evaluate")
    parser.add_argument("--input-file", type=str, default="data/enriched_decisions.jsonl")
    parser.add_argument("--output-file", type=str, help="Will default to data/optimization_results_{agent}.jsonl")
    parser.add_argument("--initial-cash", type=float, default=100000.0)
    parser.add_argument("--margin-requirement", type=float, default=0.5)
    parser.add_argument("--disable-risk-manager", action="store_true", help="Disable Risk Manager")
    parser.add_argument("--fast", action="store_true", help="Use pre-loaded PriceMatrix")
    parser.add_argument("--turnover-penalty", type=float, default=0.05, help="L1 penalty for turnover in QP optimizer")
    parser.add_argument("--decay-mode", type=str, choices=["none", "soft", "harsh"], default="harsh", help="Configure the kinematic decay speed for legacy positions")
    parser.add_argument("--segregate-capital", type=float, default=0.0, help="Ratio (0.0 to 1.0) of capital to allocate to fresh signals vs old decayed holdings. 0.0 disables segregation.")
    parser.add_argument("--transfer-rate", type=float, default=1.0, help="Multiplier for the zero-sum capital transfer penalty/reward.")
    parser.add_argument("--enable-smoothing", action="store_true", help="Enable EMA smoothing for alpha and capital floor protection in the betting market.")
    parser.add_argument("--enable-safety-net", action="store_true", help="Enable maximum daily loss caps and absolute bankruptcy floors in the betting market.")
    parser.add_argument("--max-daily-loss-pct", type=float, default=0.25, help="Maximum percentage of current capital an agent can lose in a single day during zero-sum settlement.")
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_file_path = args.output_file if args.output_file else f"data/optimization_results_{args.agent}.jsonl"
    output_file = Path(output_file_path)
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
        
    logger.info(f"Processing SINGLE AGENT ({args.agent.upper()}) optimization on {input_file}")
    
    price_matrix = None
    if args.fast:
        if PriceMatrix is not None:
            price_matrix = PriceMatrix()
        else:
            logger.warning("PriceMatrix not available. Falling back to slow mode.")
    
    all_tickers = set()
    with open(input_file, "r") as f:
        for line in f:
            try:
                day_data = json.loads(line)
                tickers = day_data.get("tickers", [])
                all_tickers.update([t for t in tickers if t != "CASH"])
            except:
                continue
        
    portfolio = Portfolio(
        tickers=list(all_tickers),
        initial_cash=args.initial_cash,
        margin_requirement=args.margin_requirement
    )
    executor = TradeExecutor()
        
    rf_file = Path("data/processed/daily_risk_free_rates.parquet")
    if rf_file.exists():
        rf_df = pd.read_parquet(rf_file)
    else:
        logger.warning("RF Default")
        rf_df = pd.DataFrame()
    
    previous_consensus = {}
    previous_prices = {}
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as out_f:
        with open(input_file, "r") as in_f:
            for line in tqdm(in_f, desc=f"Backtesting {args.agent.upper()}"):
                try:
                    day_data = json.loads(line)
                    date_str = day_data.get('date')
                    rf_rate = get_dynamic_rf_rate(date_str, rf_df)
                    
                    res = process_day(day_data, rf_rate=rf_rate, 
                                      portfolio=portfolio,
                                      executor=executor,
                                      previous_consensus=previous_consensus,
                                      previous_prices=previous_prices,
                                      agent_name=args.agent,
                                      disable_risk_manager=args.disable_risk_manager,
                                      turnover_penalty=args.turnover_penalty,
                                      decay_mode=args.decay_mode,
                                      segregate_capital=args.segregate_capital,
                                      transfer_rate=args.transfer_rate,
                                      enable_smoothing=args.enable_smoothing,
                                      enable_safety_net=args.enable_safety_net,
                                      max_daily_loss_pct=args.max_daily_loss_pct,
                                      price_matrix=price_matrix) 
                    
                    out_f.write(json.dumps(res) + "\n")
                    
                    previous_consensus = res["adjusted_consensus"].copy()
                    
                    interest_added = portfolio.add_cash_interest(rf_rate)
                    
                except Exception as e:
                    logger.error(f"Error processing a day: {e}")
            
    logger.info(f"Optimization complete. Final Portfolio Value: ${calculate_portfolio_value(portfolio, previous_prices):.2f}")

if __name__ == "__main__":
    main()
