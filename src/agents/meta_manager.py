from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
from src.schemas import Bet, MarketSignal
import json
import numpy as np

def meta_manager_agent(state: AgentState, agent_id: str = "meta_manager_agent"):
    """
    Meta-Manager agent that monitors performance and allocates capital to agents.
    """
    data = state["data"]
    analyst_signals = data.get("analyst_signals", {})
    agent_capital = data.get("agent_capital", {})
    
    # Initialize if empty
    if not agent_capital:
        # Default initialization for all agents found in signals
        for agent_name in analyst_signals.keys():
            if agent_name not in agent_capital:
                agent_capital[agent_name] = {
                    "total_capital": 100000.0,
                    "allocated_capital": 100000.0,
                    "roi_history": []
                }
    
    # Check if we have previous day's bets and today's prices to settle
    # This data would be injected by the backtester or carried over in state
    previous_bets = data.get("previous_bets", {})
    current_prices = data.get("current_prices", {})
    previous_prices = data.get("previous_prices", {})
    
    if previous_bets and current_prices and previous_prices:
        progress.update_status(agent_id, None, "Settling bets from previous round")
        agent_capital = settle_bets(agent_capital, previous_bets, current_prices, previous_prices)
    else:
        progress.update_status(agent_id, None, "No previous bets to settle, skipping settlement")

    # Update state
    data["agent_capital"] = agent_capital
    
    # Create message
    allocation_summary = {name: caps["allocated_capital"] for name, caps in agent_capital.items()}
    message = HumanMessage(
        content=json.dumps(allocation_summary),
        name=agent_id,
    )
    
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(allocation_summary, "Meta-Manager")
        
    return {
        "messages": [message],
        "data": data
    }

def settle_bets(agent_capital, previous_bets, current_prices, previous_prices):
    """
    Settles bets based on price changes and redistributes capital (Zero-Sum).
    """
    winners = []
    losers = []
    total_loss_pool = 0.0
    
    # 1. Determine Win/Loss for each bet
    for agent_name, bets_data in previous_bets.items():
        # Skip non-analyst agents
        if agent_name not in agent_capital:
            continue
            
        # bets_data is a dict of ticker -> bet_dump
        for ticker, bet_dump in bets_data.items():
            try:
                bet = Bet(**bet_dump)
                start_price = previous_prices.get(ticker)
                end_price = current_prices.get(ticker)
                
                if not start_price or not end_price:
                    continue
                    
                price_change_pct = (end_price - start_price) / start_price
                
                # Determine outcome
                is_win = False
                if bet.direction == MarketSignal.BULLISH and price_change_pct > 0:
                    is_win = True
                elif bet.direction == MarketSignal.BEARISH and price_change_pct < 0:
                    is_win = True
                elif bet.direction == MarketSignal.NEUTRAL and abs(price_change_pct) < 0.005: # Flat
                    is_win = True
                    
                if is_win:
                    # Winner keeps their stake + share of pool later
                    winners.append({"agent": agent_name, "bet_amount": bet.amount, "ticker": ticker})
                else:
                    # Loser loses their stake to the pool
                    loss_amount = bet.amount
                    # Cap loss at available capital just in case
                    loss_amount = min(loss_amount, agent_capital[agent_name]["allocated_capital"])
                    
                    agent_capital[agent_name]["allocated_capital"] -= loss_amount
                    total_loss_pool += loss_amount
                    losers.append({"agent": agent_name, "loss": loss_amount})
                    
            except Exception as e:
                print(f"Error settling bet for {agent_name}: {e}")

    # 2. Redistribute Loss Pool to Winners
    if total_loss_pool > 0 and winners:
        total_winning_stake = sum(w["bet_amount"] for w in winners)
        
        if total_winning_stake > 0:
            for w in winners:
                agent_name = w["agent"]
                share = w["bet_amount"] / total_winning_stake
                reward = total_loss_pool * share
                
                agent_capital[agent_name]["allocated_capital"] += reward
                
                # Update ROI history (simplified)
                roi = reward / w["bet_amount"]
                agent_capital[agent_name]["roi_history"].append(roi)
        else:
            # Edge case: Winners exist but stake is 0? Should not happen if amount > 0.
            # Refund to losers to be safe.
            for l in losers:
                agent_capital[l["agent"]]["allocated_capital"] += l["loss"]
                
    elif total_loss_pool > 0 and not winners:
        # Refund to losers
        for l in losers:
            agent_capital[l["agent"]]["allocated_capital"] += l["loss"]

    return agent_capital
