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


def settle_bets(agent_capital, previous_bets, current_prices, previous_prices, transfer_rate: float = 1.0, enable_smoothing: bool = False, max_daily_loss_pct: float = 0.25):
    """
    Settles bets based on relative performance (Alpha) using Dual-Tranche Attribution.
    Zero-Sum Game: Agents who underperform the average pay into a pool; 
                   Agents who outperform take from the pool.
    """
    agent_returns = {}
    agent_ratios = {} # Store the external ratio (r) for each agent
    
    # 1. Calculate each agent's true daily return
    for agent_name, bets_data in previous_bets.items():
        if agent_name not in agent_capital:
            continue
            
        caps = agent_capital[agent_name]
        total_cap = caps.get("internal_capital", 0) + caps.get("external_capital", 0)
        
        # Calculate External Ratio (r)
        r = caps.get("external_capital", 0) / total_cap if total_cap > 0 else 0.0
        agent_ratios[agent_name] = r
            
        allocations = bets_data.get("allocations", [])
        total_gross_amount = sum(alloc.get("amount", 0.0) for alloc in allocations)
        
        if total_gross_amount == 0:
            agent_returns[agent_name] = 0.0
            continue
            
        daily_return = 0.0
        for alloc_dump in allocations:
            try:
                ticker = alloc_dump.get("ticker")
                direction = alloc_dump.get("direction") # "long" or "short"
                amount_pct = alloc_dump.get("amount", 0.0) # 0 to 100
                
                # Weight of this asset within the agent's portfolio
                weight = amount_pct / total_gross_amount
                
                start_price = previous_prices.get(ticker)
                end_price = current_prices.get(ticker)
                
                if not start_price or not end_price:
                    continue
                    
                price_change_pct = (end_price - start_price) / start_price
                
                if direction == "long":
                    daily_return += weight * price_change_pct
                elif direction == "short":
                    daily_return -= weight * price_change_pct
                    
            except Exception as e:
                print(f"Error calculating return for {agent_name} on {ticker}: {e}")
                
        agent_returns[agent_name] = daily_return

    if not agent_returns:
        return agent_capital

    # 2. Calculate the average return (Benchmark)
    avg_return = sum(agent_returns.values()) / len(agent_returns)
    
    # 3. Alpha-based Zero-Sum Settlement (Strict Subsidy Pool)
    
    # 3.1 Calculate Taxes (Losers pay into the pool)
    subsidy_pool = 0.0
    agent_alphas = {}
    for agent_name, ret in agent_returns.items():
        raw_alpha = ret - avg_return
        caps = agent_capital[agent_name]
        
        if enable_smoothing:
            # Initialize with raw_alpha if not exists
            if "smoothed_alpha" not in caps:
                caps["smoothed_alpha"] = raw_alpha
            else:
                # EMA: 80% history, 20% today's noise
                caps["smoothed_alpha"] = (0.8 * caps["smoothed_alpha"]) + (0.2 * raw_alpha)
            effective_alpha = caps["smoothed_alpha"]
        else:
            effective_alpha = raw_alpha
            
        agent_alphas[agent_name] = effective_alpha
        
        if effective_alpha < 0:
            total_cap = caps.get("internal_capital", 0) + caps.get("external_capital", 0)
            
            MIN_CAPITAL = 1000.0  # Absolute floor
            
            # Calculate theoretical penalty with leverage
            theoretical_penalty = total_cap * abs(effective_alpha) * transfer_rate
            
            if enable_smoothing:
                # Constrain penalty: 1) Cannot exceed max_daily_loss_pct of current wealth. 2) Cannot drop below MIN_CAPITAL.
                max_allowed_loss = total_cap * max_daily_loss_pct
                available_to_lose = max(0.0, total_cap - MIN_CAPITAL)
                
                actual_penalty = min(theoretical_penalty, max_allowed_loss, available_to_lose)
            else:
                # Original naive logic
                actual_penalty = theoretical_penalty
            
            subsidy_pool += actual_penalty
            
            # Deduct `actual_penalty` from loser
            r = agent_ratios[agent_name]
            caps["external_capital"] = max(0, caps.get("external_capital", 0) - actual_penalty * r)
            caps["internal_capital"] = max(0, caps.get("internal_capital", 0) - actual_penalty * (1 - r))
    
    # 3.2 Distribute Pool (Winners take from the pool proportionally)
    total_positive_alpha = sum(alpha for alpha in agent_alphas.values() if alpha > 0)
    
    for agent_name, alpha in agent_alphas.items():
        caps = agent_capital[agent_name]
        
        if alpha > 0 and total_positive_alpha > 0:
            # Winner's share of the pool
            share_pct = alpha / total_positive_alpha
            reward_amount = subsidy_pool * share_pct
            
            # Add to winner
            r = agent_ratios[agent_name]
            caps["external_capital"] += reward_amount * r
            caps["internal_capital"] += reward_amount * (1 - r)
            
        # Sync legacy field
        caps["allocated_capital"] = caps["external_capital"]
        
        # Update ROI history with raw return (not alpha)
        caps.setdefault("roi_history", []).append(agent_returns[agent_name])

    return agent_capital
