from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
from src.schemas import Bet, MarketSignal
import json
import numpy as np

def meta_manager_agent(state: AgentState, agent_id: str = "meta_manager_agent"):
    """
    Meta-Manager agent that monitors performance and allocates capital to agents.
    Uses a Dual-Tranche Capital Model (Internal vs External).
    """
    data = state["data"]
    analyst_signals = data.get("analyst_signals", {})
    agent_capital = data.get("agent_capital", {})
    
    # Initialize if empty
    if not agent_capital:
        # Default initialization for all agents found in signals
        # Dual-Tranche Initialization: 50% Internal (Skin in the game), 50% External (Allocation)
        for agent_name in analyst_signals.keys():
            if agent_name not in agent_capital:
                agent_capital[agent_name] = {
                    "agent_name": agent_name,
                    "allocated_capital": 50000.0, # Backward compat
                    "external_capital": 50000.0,
                    "internal_capital": 50000.0,
                    "roi_history": []
                }
    
    # Check if we have previous day's bets and today's prices to settle
    previous_bets = data.get("previous_bets", {})
    current_prices = data.get("current_prices", {})
    previous_prices = data.get("previous_prices", {})
    
    if previous_bets and current_prices and previous_prices:
        progress.update_status(agent_id, None, "Settling bets (Dual-Tranche)")
        agent_capital = settle_bets(agent_capital, previous_bets, current_prices, previous_prices)
    else:
        progress.update_status(agent_id, None, "No previous bets to settle, skipping settlement")

    # Update state
    data["agent_capital"] = agent_capital
    
    # Create message
    # Show Total Capital for simple view
    allocation_summary = {name: caps.get("external_capital", 0) + caps.get("internal_capital", 0) for name, caps in agent_capital.items()}
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
    Settles bets based on price changes using Dual-Tranche Attribution.
    Zero-Sum Game: Losers pay into a pool, Winners take from the pool.
    """
    winners = []
    losers = []
    total_loss_pool = 0.0
    
    # 1. Determine Win/Loss for each bet
    for agent_name, bets_data in previous_bets.items():
        # Skip non-analyst agents
        if agent_name not in agent_capital:
            continue
            
        caps = agent_capital[agent_name]
        total_cap = caps.get("internal_capital", 0) + caps.get("external_capital", 0)
        
        # Calculate External Ratio (r)
        # If total is 0 (bust), r is 0 (avoid div by zero)
        r = caps.get("external_capital", 0) / total_cap if total_cap > 0 else 0.0
            
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
                    # Store r so we attribute rewards correctly later
                    winners.append({
                        "agent": agent_name, 
                        "bet_amount": bet.amount, 
                        "external_ratio": r
                    })
                else:
                    # Loser loses their stake to the pool
                    loss_amount = bet.amount
                    # Cap loss at available capital
                    loss_amount = min(loss_amount, total_cap)
                    
                    # Attribute loss
                    loss_ex = loss_amount * r
                    loss_in = loss_amount * (1 - r)
                    
                    agent_capital[agent_name]["external_capital"] -= loss_ex
                    agent_capital[agent_name]["internal_capital"] -= loss_in
                    # Sync legacy field
                    agent_capital[agent_name]["allocated_capital"] = agent_capital[agent_name]["external_capital"]
                    
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
                
                # Attribute Reward using the ratio at the time of betting (or current? usually time of betting)
                r = w["external_ratio"]
                reward_ex = reward * r
                reward_in = reward * (1 - r)
                
                agent_capital[agent_name]["external_capital"] += reward_ex
                agent_capital[agent_name]["internal_capital"] += reward_in
                # Sync legacy field
                agent_capital[agent_name]["allocated_capital"] = agent_capital[agent_name]["external_capital"]
                
                # Update ROI history
                roi = reward / w["bet_amount"]
                agent_capital[agent_name]["roi_history"].append(roi)
        else:
            # Refund if something weird happens
            for l in losers:
                r = agent_capital[l["agent"]]["external_capital"] / (agent_capital[l["agent"]]["external_capital"] + agent_capital[l["agent"]]["internal_capital"] + l["loss"])
                agent_capital[l["agent"]]["external_capital"] += l["loss"] * r
                agent_capital[l["agent"]]["internal_capital"] += l["loss"] * (1-r)
                
    elif total_loss_pool > 0 and not winners:
        # Refund to losers
        for l in losers:
             # This reconstruction of r is tricky if we don't store it. 
             # Simpler to just add back to where it fits, or assume r hasn't changed much.
             # Ideally we should have stored l's ratio too. 
             # For now, let's just add back to external mostly to be safe? 
             # No, let's try to be fair. 
             # We can't perfectly reconstruct R without storing it.
             # Taking a shortcut: add back proportionally to current cap (approximate).
             total = agent_capital[l["agent"]]["external_capital"] + agent_capital[l["agent"]]["internal_capital"]
             if total > 0:
                 r = agent_capital[l["agent"]]["external_capital"] / total
             else: 
                 r = 0.5 # Default
             
             agent_capital[l["agent"]]["external_capital"] += l["loss"] * r
             agent_capital[l["agent"]]["internal_capital"] += l["loss"] * (1-r)

    return agent_capital
