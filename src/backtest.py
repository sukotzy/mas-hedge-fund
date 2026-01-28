import sys
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import json
import pandas as pd
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from colorama import Fore, Style, init

from src.main import create_workflow, parse_hedge_fund_response
from src.utils.progress import progress
from src.tools.api import get_prices
from src.utils.api_key import get_api_key_from_state

init(autoreset=True)

def run_backtest(
    tickers: list[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-4o",
    model_provider: str = "OpenAI",
):
    print(f"{Fore.CYAN}Starting Backtest from {start_date} to {end_date}...{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Using Model: {model_provider} / {model_name}{Style.RESET_ALL}")
    
    # Initialize Portfolio
    portfolio = {
        "cash": initial_capital,
        "margin_requirement": 0.5,
        "margin_used": 0.0,
        "positions": {t: {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0} for t in tickers},
        "realized_gains": {t: {"long": 0.0, "short": 0.0} for t in tickers},
    }
    
    # Initialize Agent Capital
    # We let the MetaManager initialize it on the first run, then we persist it.
    agent_capital = {}
    
    # Date Loop
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Store history
    history = []
    
    # Previous state for settlement
    previous_bets = {}
    previous_prices = {}
    
    while current_date <= end_date_dt:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"\n{Fore.YELLOW}Processing Date: {date_str}{Style.RESET_ALL}")
        
        # 1. Get Prices for Today (Simulate Market Close for Settlement)
        # In a real backtest, we'd need precise timing. Here we assume 'current_date' is the trading day.
        # We need "today's close" to settle "yesterday's bets".
        # And we need "today's data" for "today's bets".
        
        # For simplicity in this demo:
        # - We run the agent on 'date_str' to get bets.
        # - We use 'date_str' prices to settle 'previous_bets'.
        # - This implies agents bet at Open/Intraday and we settle at Close? 
        # - Or we settle yesterday's bets using today's Open?
        # Let's assume: Settle yesterday's bets using Today's Close. Agents bet for Tomorrow.
        
        # Fetch prices
        current_prices = {}
        for ticker in tickers:
            prices = get_prices(ticker, date_str, date_str)
            if prices:
                current_prices[ticker] = prices[-1]["close"]
            else:
                # Mock data for verification if real data is missing
                import random
                # Generate a random price movement
                prev_price = previous_prices.get(ticker, 100.0)
                change = random.uniform(-0.05, 0.05)
                current_prices[ticker] = prev_price * (1 + change)
                print(f"Mocking price for {ticker}: {current_prices[ticker]:.2f}")
        
        if not current_prices:
            print(f"No price data for {date_str}, skipping...")
            current_date += timedelta(days=1)
            continue

        # 2. Run Workflow
        workflow = create_workflow(selected_analysts if selected_analysts else None)
        app = workflow.compile()
        
        # Construct State
        initial_state = {
            "messages": [HumanMessage(content="Make trading decisions.")],
            "data": {
                "tickers": tickers,
                "portfolio": portfolio,
                "start_date": date_str, # Agent looks at data ending here
                "end_date": date_str,
                "analyst_signals": {},
                "agent_capital": agent_capital, # Pass carried-over capital
                "previous_bets": previous_bets, # Pass for settlement
                "current_prices": current_prices, # Pass for settlement
                "previous_prices": previous_prices, # Pass for settlement
            },
            "metadata": {
                "show_reasoning": False,
                "model_name": model_name,
                "model_provider": model_provider,
            },
        }
        
        final_state = app.invoke(initial_state)
        
        # 3. Update State for Next Loop
        data = final_state["data"]
        
        # Update Capital (MetaManager has already settled and updated 'agent_capital' in 'data')
        agent_capital = data["agent_capital"]
        
        # Store Bets for next settlement
        # We need to extract bets from analyst_signals
        # Structure: analyst_signals[agent][ticker] = Bet(dict)
        new_bets = {}
        analyst_signals = data["analyst_signals"]
        for agent, signals in analyst_signals.items():
            if "risk" in agent or "portfolio" in agent: continue
            new_bets[agent] = signals # This contains the Bet dicts
            
        previous_bets = new_bets
        previous_prices = current_prices
        
        # Track Total Capital
        total_external = sum(c.get("external_capital", c.get("allocated_capital", 0)) for c in agent_capital.values())
        total_internal = sum(c.get("internal_capital", 0) for c in agent_capital.values())
        total_combined = total_external + total_internal
        
        print(f"Total Agent Capital: ${total_combined:,.2f} (Ext: ${total_external:,.2f} | Int: ${total_internal:,.2f})")
        
        history.append({
            "date": date_str,
            "total_capital": total_combined,
            "agent_capital": {
                k: {
                    "allocated_capital": v.get("allocated_capital", 0),
                    "external_capital": v.get("external_capital", 0),
                    "internal_capital": v.get("internal_capital", 0)
                } 
                for k, v in agent_capital.items()
            }
        })
        
        current_date += timedelta(days=1)

    # Save History
    with open("backtest_history.json", "w") as f:
        json.dump(history, f, indent=4)
        
    print(f"{Fore.GREEN}Backtest Complete. History saved to backtest_history.json{Style.RESET_ALL}")

if __name__ == "__main__":
    from src.cli.input import parse_cli_inputs
    
    # Parse CLI arguments
    inputs = parse_cli_inputs(
        description="Run the hedge fund backtest system",
        require_tickers=True,
        default_months_back=None,
        include_graph_flag=False,
        include_reasoning_flag=False, 
    )
    
    run_backtest(
        tickers=inputs.tickers,
        start_date=inputs.start_date,
        end_date=inputs.end_date,
        initial_capital=inputs.initial_cash,
        selected_analysts=inputs.selected_analysts,
        model_name=inputs.model_name,
        model_provider=inputs.model_provider,
    )
