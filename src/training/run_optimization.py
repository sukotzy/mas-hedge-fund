import json
import logging
import os
import sys
from pathlib import Path

# Fix python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import datetime
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

# Load Env Vars
load_dotenv()
os.environ["USE_LOCAL_DATA"] = "true"

from src.market.betting_market import BettingMarket
from src.schemas import Bet, MarketSignal
from src.agents.risk_manager import risk_management_agent
from src.agents.meta_manager import settle_bets
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

# Run LP is handled by solve_optimization_lp in optimizer_utils now

def get_risk_limits(date_str: str, tickers: list[str], portfolio: Portfolio, current_portfolio_value: float, disable_risk_manager: bool = False):
    """
    Pass the portfolio state and run the risk manager to get dynamic limits.
    """
    if disable_risk_manager:
        limits = {t: current_portfolio_value for t in tickers}
        return limits, {}

    portfolio_state = {
        "cash": portfolio.get_cash(),
        "margin_requirement": getattr(portfolio, 'margin_requirement', 0.5),
        "margin_used": portfolio.get_margin_used(),
        "positions": portfolio.get_positions(),
        "realized_gains": {}
    }
    
    state = {
        "messages": [],
        "data": {
            "tickers": tickers,
            "start_date": date_str,
            "end_date": date_str,
            "portfolio": portfolio_state,
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
        limits[t] = float(data.get("position_limit", data.get("reasoning", {}).get("position_limit", 0.0)))
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


def process_day(day_data: dict, rf_rate: float, portfolio: Portfolio, executor: TradeExecutor, previous_consensus: dict, agent_capital: dict, previous_bets: dict, previous_prices: dict, disable_risk_manager: bool = False, turnover_penalty: float = 0.05, decay_mode: str = "harsh", segregate_capital: float = 0.0, transfer_rate: float = 1.0, enable_smoothing: bool = False, smoothing_factor: float = 0.2, enable_safety_net: bool = False, max_daily_loss_pct: float = 0.25, price_matrix=None, active_agents: list = None, use_replicator_dynamics: bool = False, rd_eta: float = 25.0, rd_tau: float = 0.05, rd_eta_downside: float = None, rd_eta_downside_threshold: float = -0.02):
    if active_agents is None:
        active_agents = ["fundamental", "technical", "valuation", "sentiment", "virtual_cash"]
        
    date_str = day_data["date"]
    tickers = day_data["tickers"]
    rm_tickers = [t for t in tickers if t != "CASH"]
    
    active_tickers = set(rm_tickers)
    for t, pos in portfolio.get_positions().items():
        if pos["long"] > 0 or pos["short"] > 0:
            active_tickers.add(t)
            
    # FIX: Ensure we fetch prices for anything we bet on yesterday, 
    # otherwise settle_bets will use $0.0 and calculate a -100% ROI!
    if previous_bets:
        for agent, bet_data in previous_bets.items():
            for alloc in bet_data.get("allocations", []):
                bet_ticker = alloc.get("ticker")
                if bet_ticker and bet_ticker != "CASH":
                    active_tickers.add(bet_ticker)
    
    # 1. Fetch 15-day trailing prices for decay logic and get today's prices for settlement
    prices_history = {}
    dt_end = pd.Timestamp(date_str)
    dt_start = dt_end - pd.Timedelta(days=15)
    
    current_prices = {}
    if price_matrix is not None:
        # FAST PATH: Use pre-loaded PriceMatrix for O(1) lookups
        for t in active_tickers:
            price = price_matrix.get_close(t, date_str)
            current_prices[t] = price if price > 0 else previous_prices.get(t, 0.0)
            prices_history[t] = price_matrix.get_price_history_df(
                t, dt_start.strftime("%Y-%m-%d"), date_str
            )
    else:
        # SLOW PATH: Original per-ticker API calls
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
                logger.warning(f"Failed to fetch price data for {t} from {dt_start.strftime('%Y-%m-%d')} to {date_str}: {e}")
                prices_history[t] = pd.DataFrame()
                current_prices[t] = previous_prices.get(t, 0.0)

    # 1.1 Settle Yesterday's Bets (Zero-Sum Settlement)
    if previous_bets and previous_prices:
        agent_capital = settle_bets(
            agent_capital, 
            previous_bets, 
            current_prices, 
            previous_prices,
            transfer_rate=transfer_rate,
            enable_smoothing=enable_smoothing,
            smoothing_factor=smoothing_factor,
            enable_safety_net=enable_safety_net,
            max_daily_loss_pct=max_daily_loss_pct,
            use_replicator_dynamics=use_replicator_dynamics,
            rd_eta=rd_eta,
            rd_tau=rd_tau,
            risk_free_rate=rf_rate,
            rd_eta_downside=rd_eta_downside,
            rd_eta_downside_threshold=rd_eta_downside_threshold
        )
        
    # 2. Reconstruct Betting Market
    market = BettingMarket()
    agent_names = active_agents
    
    logger.info(f"\n[{date_str}] --- DAY START ---")
    
    for agent in agent_names:
        # Auto-inject virtual cash for backward compatibility with old datasets
        if agent == "virtual_cash" and agent not in day_data:
            decision = {
                "allocations": [{"ticker": "CASH", "direction": "long", "amount": 100.0}],
                "reasoning": "Injected risk-off safe haven."
            }
        elif agent not in day_data:
            continue
        else:
            decision = day_data[agent]
        
        dynamic_cap = agent_capital[agent].get("external_capital", 0) + agent_capital[agent].get("internal_capital", 0)
        
        # Extract today's ROI (if available from settle_bets)
        history = agent_capital[agent].get("roi_history", [])
        daily_roi = history[-1] if history else 0.0
        
        logger.info(f"[{date_str}] Allocator {agent.upper()} Capital: ${dynamic_cap:,.2f} "
                    f"(External: ${agent_capital[agent].get('external_capital', 0):,.2f}, "
                    f"Internal: ${agent_capital[agent].get('internal_capital', 0):,.2f}) "
                    f"[Daily ROI: {daily_roi*100:.4f}%]")
        
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
            bet_amt = (amount / 100.0) * dynamic_cap
            logger.info(f"    -> Bet: {direction.upper()} {ticker} | Weight: {amount}% | USD: ${bet_amt:,.2f}")
            
            b = Bet(
                ticker=ticker,
                direction=sig,
                amount=bet_amt,
                conviction=1.0,
                reasoning="Extracted from JSONL"
            )
            market.place_bet(b)
            
    consensus_values = market.calculate_consensus()
    
    # 3. Calculate portfolio value BEFORE optimization
    current_portfolio_value = calculate_portfolio_value(portfolio, current_prices)
    
    # 4. Get Risk Limits dynamically scaled to current value
    if price_matrix is not None and calculate_risk_limits_fast is not None:
        # FAST PATH: Inlined risk calculations using PriceMatrix
        risk_limits, rm_prices = calculate_risk_limits_fast(
            date_str, list(active_tickers), price_matrix,
            portfolio, current_portfolio_value, disable_risk_manager
        )
    else:
        # SLOW PATH: Original risk manager agent
        risk_limits, rm_prices = get_risk_limits(date_str, list(active_tickers), portfolio, current_portfolio_value, disable_risk_manager)
    
    # We will use the risk manager's provided prices primarily, but fallback to current_prices if needed
    for t in active_tickers:
        if t in rm_prices and rm_prices[t] > 0:
            current_prices[t] = rm_prices[t]
    
    # 5. Extract previous holdings (net positions) dynamically from Portfolio
    previous_holdings = {}
    positions_state = portfolio.get_positions()
    for t in active_tickers:
        if t in positions_state:
            pos = positions_state[t]
            net_holding = pos.get("long", 0) - pos.get("short", 0)
            previous_holdings[t] = net_holding
        else:
            previous_holdings[t] = 0

    logger.info(f"[{date_str}] Optimizer Inputs:")
    logger.info(f"    -> Initial Cash Flow Pool: ${current_portfolio_value:,.2f}")
    logger.info(f"    -> RF Rate: {rf_rate:.6f}")
    logger.info(f"    -> Asset Pool (Risk Managed Tickers count): {len(active_tickers)}")
    for t in list(active_tickers)[:5]:  # Print first 5 to avoid spamming the console
         logger.info(f"         - {t}: Consensus={consensus_values.get(t, 0.0):.4f}, Risk Limit QTY={risk_limits.get(t, 0)}, Prev Hold QTY={previous_holdings.get(t, 0)}")
    if len(active_tickers) > 5:
         logger.info(f"         ... and {len(active_tickers) - 5} more tickers.")

    # 6. Optimization
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
    
    logger.info(f"[{date_str}] Optimizer Output (Target Shares to Hold):")
    non_zero_targets = {t: shares for t, shares in optimal_shares.items() if shares != 0}
    for t, shares in list(non_zero_targets.items())[:10]:
        logger.info(f"    -> {t}: target={shares} shares")
    if len(non_zero_targets) > 10:
        logger.info(f"    ... and {len(non_zero_targets) - 10} more targets.")
    
    # 7. Execute Delta Trades
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
        else:  # net_holding < 0
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
                
    # 8. Calculate NEW portfolio value AFTER execution
    updated_portfolio_value = calculate_portfolio_value(portfolio, current_prices)
    
    # Extract updated positions summary
    updated_positions = {t: {"long": pos["long"], "short": pos["short"]} 
                         for t, pos in portfolio.get_positions().items() 
                         if pos["long"] > 0 or pos["short"] > 0}
                         
    # 9. Update state memory for tomorrow
    # CRITICAL MEMORY LEAK FIX:
    # Do not let previous_prices accumulate thousands of dead tickers over 9 years.
    # Only cache prices for tickers that are actively in today's pool.
    previous_prices.clear()
    for t in active_tickers:
        previous_prices[t] = current_prices.get(t, 0.0)
        
    for agent in agent_names:
        if agent == "virtual_cash" and agent not in day_data:
            previous_bets[agent] = {
                "allocations": [{"ticker": "CASH", "direction": "long", "amount": 100.0}],
                "reasoning": "Injected risk-off safe haven."
            }
        elif agent in day_data:
            previous_bets[agent] = day_data[agent]
            
    # Prune adjusted consensus as well
    pruned_adjusted_consensus = {t: v for t, v in adjusted_consensus.items() if t in active_tickers}
    
    return {
        "date": date_str,
        "portfolio_value": updated_portfolio_value,
        "executed_trades": executed_trades,
        "updated_holdings": updated_positions,
        "consensus": consensus_values,
        "adjusted_consensus": pruned_adjusted_consensus,
        "agent_capital": agent_capital,
        "prices": rm_prices,
        "risk_limits": risk_limits,
        "optimal_shares": optimal_shares,
        "objective_cash_constant": current_portfolio_value * rf_rate
    }


def main():
    parser = argparse.ArgumentParser(description="Run Offline Optimization & Backtesting Engine")
    parser.add_argument("--input-file", type=str, default="data/enriched_decisions.jsonl")
    parser.add_argument("--output-file", type=str, default="data/optimization_results_final.jsonl")
    parser.add_argument("--initial-cash", type=float, default=100000.0)
    parser.add_argument("--margin-requirement", type=float, default=0.5)
    parser.add_argument("--disable-risk-manager", action="store_true", help="Disable Risk Manager and allow full allocations")
    parser.add_argument("--fast", action="store_true", help="Use pre-loaded PriceMatrix for O(1) lookups (much faster for long backtests)")
    parser.add_argument("--turnover-penalty", type=float, default=0.05, help="L1 penalty for turnover in QP optimizer")
    parser.add_argument("--decay-mode", type=str, choices=["none", "soft", "harsh"], default="harsh", help="Configure the kinematic decay speed for legacy positions")
    parser.add_argument("--segregate-capital", type=float, default=0.0, help="Ratio (0.0 to 1.0) of capital to allocate to fresh signals vs old decayed holdings. 0.0 disables segregation.")
    parser.add_argument("--transfer-rate", type=float, default=1.0, help="Multiplier for the zero-sum capital transfer penalty/reward.")
    parser.add_argument("--enable-smoothing", action="store_true", help="Enable EMA smoothing for alpha and capital floor protection in the betting market.")
    parser.add_argument("--smoothing-factor", type=float, default=0.2, help="The EMA new information weight (default 0.2) when smoothing is enabled.")
    parser.add_argument("--enable-safety-net", action="store_true", help="Enable maximum daily loss caps and absolute bankruptcy floors in the betting market.")
    parser.add_argument("--max-daily-loss-pct", type=float, default=0.25, help="Maximum percentage of current capital an agent can lose in a single day during zero-sum settlement.")
    parser.add_argument("--use-replicator-dynamics", action="store_true", help="Use Replicator Dynamics with Uniform Mutation for zero-sum settlement.")
    parser.add_argument("--rd-eta", type=float, default=25.0, help="Exponential amplifier (learning rate) for Replicator Dynamics.")
    parser.add_argument("--rd-tau", type=float, default=0.05, help="Wealth tax / mutation rate for Replicator Dynamics.")
    parser.add_argument("--rd-eta-downside", type=float, default=None, help="Exponential amplifier for severe negative alpha (Downside Risk Aversion). Defaults to symmetric rd-eta if not provided.")
    parser.add_argument("--rd-eta-downside-threshold", type=float, default=-0.02, help="Alpha threshold to trigger the downside eta amplifier.")
    parser.add_argument("--active-agents", nargs='+', default=["fundamental", "technical", "valuation", "sentiment", "virtual_cash"], help="List of active agents participating in the Meta Manager")
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}. Please run simulate_wealth_trajectory.py first.")
        return
        
    logger.info(f"Processing optimization on {input_file}")
    
    # Initialize PriceMatrix if --fast mode is enabled
    price_matrix = None
    if args.fast:
        if PriceMatrix is not None:
            price_matrix = PriceMatrix()
        else:
            logger.warning("PriceMatrix not available. Falling back to slow mode.")
    
    # First pass: collect all unique tickers to initialize Portfolio without loading the whole file
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
        
    # Load RF Data
    rf_file = Path("data/processed/daily_risk_free_rates.parquet")
    if rf_file.exists():
        rf_df = pd.read_parquet(rf_file)
    else:
        logger.warning(f"RF data not found at {rf_file}, using fallback 5% annual.")
        rf_df = pd.DataFrame()
    
    # Memory state for the optimizer
    previous_consensus = {}
    agent_capital = {
        agent: {
            "agent_name": agent,
            "allocated_capital": 50000.0,
            "external_capital": 50000.0,
            "internal_capital": 50000.0,
            "roi_history": []
        } for agent in args.active_agents
    }
    previous_bets = {}
    previous_prices = {}
    
    # Open output file for immediate writing (Streaming Mode)
    with open(output_file, "w") as out_f:
        with open(input_file, "r") as in_f:
            for line in tqdm(in_f, desc="Backtesting"):
                try:
                    day_data = json.loads(line)
                    date_str = day_data.get('date')
                    rf_rate = get_dynamic_rf_rate(date_str, rf_df)
                    
                    # Pure functional optimization pass over state
                    res = process_day(day_data, rf_rate=rf_rate, 
                                      portfolio=portfolio,
                                      executor=executor,
                                      previous_consensus=previous_consensus,
                                      agent_capital=agent_capital,
                                      previous_bets=previous_bets,
                                      previous_prices=previous_prices,
                                      disable_risk_manager=args.disable_risk_manager,
                                      turnover_penalty=args.turnover_penalty,
                                      decay_mode=args.decay_mode,
                                      segregate_capital=args.segregate_capital,
                                      transfer_rate=args.transfer_rate,
                                      enable_smoothing=args.enable_smoothing,
                                      smoothing_factor=args.smoothing_factor,
                                      enable_safety_net=args.enable_safety_net,
                                      max_daily_loss_pct=args.max_daily_loss_pct,
                                      price_matrix=price_matrix,
                                      active_agents=args.active_agents,
                                      use_replicator_dynamics=args.use_replicator_dynamics,
                                      rd_eta=args.rd_eta,
                                      rd_tau=args.rd_tau,
                                      rd_eta_downside=args.rd_eta_downside,
                                      rd_eta_downside_threshold=args.rd_eta_downside_threshold) 
                    
                    # Write result IMMEDIATELY to prevent memory accumulation (OOM fix)
                    out_f.write(json.dumps(res) + "\n")
                    # REMOVED: out_f.flush() - This was causing a massive I/O bottleneck by forcing a sync disk write every line.
                    
                    # Update memory state after each day
                    previous_consensus = res["adjusted_consensus"].copy()
                    
                    # Apply daily interest to cash balance
                    interest_added = portfolio.add_cash_interest(rf_rate)
                    if interest_added > 0 and args.fast is False: # Limit logs in fast mode
                        logger.info(f"[{date_str}] Interest added to cash: ${interest_added:,.2f}")
                    
                except Exception as e:
                    logger.error(f"Error processing a day: {e}")
            
    logger.info(f"Optimization complete. Final Portfolio Value: ${calculate_portfolio_value(portfolio, previous_prices):.2f}")

if __name__ == "__main__":
    main()
