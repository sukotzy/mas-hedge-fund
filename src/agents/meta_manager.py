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


def settle_bets(agent_capital, previous_bets, current_prices, previous_prices, transfer_rate: float = 1.0, enable_smoothing: bool = False, smoothing_factor: float = 0.2, enable_safety_net: bool = False, max_daily_loss_pct: float = 0.25, use_replicator_dynamics: bool = False, rd_eta: float = 25.0, rd_tau: float = 0.05, risk_free_rate: float = 0.0, rd_eta_downside: float = None, rd_eta_downside_threshold: float = -0.02):
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
                
                if ticker == 'CASH':
                    daily_return += weight * risk_free_rate
                    continue
                
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
    
    # 3. Alpha-based Zero-Sum Settlement
    agent_alphas = {}
    for agent_name, ret in agent_returns.items():
        raw_alpha = ret - avg_return
        caps = agent_capital[agent_name]
        
        if enable_smoothing:
            # Initialize with raw_alpha if not exists
            if "smoothed_alpha" not in caps:
                caps["smoothed_alpha"] = raw_alpha
            else:
                # EMA: (1 - factor) history, factor * today's noise
                caps["smoothed_alpha"] = ((1.0 - smoothing_factor) * caps["smoothed_alpha"]) + (smoothing_factor * raw_alpha)
            effective_alpha = caps["smoothed_alpha"]
        else:
            effective_alpha = raw_alpha
            
        agent_alphas[agent_name] = effective_alpha
        
    if use_replicator_dynamics:
        agent_names_list = list(agent_returns.keys())
        N = len(agent_names_list)
        weights = []
        for name in agent_names_list:
            caps = agent_capital[name]
            w = caps.get("internal_capital", 0) + caps.get("external_capital", 0)
            weights.append(w)
            
        total_pool = sum(weights)
        if total_pool > 0:
            weights = np.array(weights) / total_pool
            alphas = np.array([agent_alphas[name] for name in agent_names_list])
            
            # Asymmetrical Eta (Downside Risk Aversion)
            actual_downside_eta = rd_eta_downside if rd_eta_downside is not None else rd_eta
            eta_array = np.where(alphas < rd_eta_downside_threshold, actual_downside_eta, rd_eta)
            
            # Exponential Update with clip to prevent overflow
            exp_factor = np.exp(np.clip(eta_array * alphas, -20.0, 20.0))
            w_prime = weights * exp_factor
            
            # Zero-Sum Normalization
            if np.sum(w_prime) > 0:
                w_double_prime = w_prime / np.sum(w_prime)
                
                # Mutation (Tax & UBI)
                w_final = (1.0 - rd_tau) * w_double_prime + (rd_tau / N)
                
                # Reassign external/internal capital
                for i, name in enumerate(agent_names_list):
                    new_total = total_pool * w_final[i]
                    r = agent_ratios[name]
                    caps = agent_capital[name]
                    caps["external_capital"] = new_total * r
                    caps["internal_capital"] = new_total * (1 - r)
                    caps["allocated_capital"] = caps["external_capital"]
                    
        # Update ROI history
        for agent_name in agent_names_list:
            agent_capital[agent_name].setdefault("roi_history", []).append(agent_returns[agent_name])
            
        return agent_capital
        
    else:
        # Original Linear Subsidy Logic
        # 3.1 Calculate Taxes (Losers pay into the pool)
        subsidy_pool = 0.0
        for agent_name, effective_alpha in agent_alphas.items():
            caps = agent_capital[agent_name]
            if effective_alpha < 0:
                total_cap = caps.get("internal_capital", 0) + caps.get("external_capital", 0)
                
                MIN_CAPITAL = 1000.0  # Absolute floor
                theoretical_penalty = total_cap * abs(effective_alpha) * transfer_rate
                
                if enable_safety_net:
                    max_allowed_loss = total_cap * max_daily_loss_pct
                    available_to_lose = max(0.0, total_cap - MIN_CAPITAL)
                    actual_penalty = min(theoretical_penalty, max_allowed_loss, available_to_lose)
                else:
                    actual_penalty = theoretical_penalty
                
                subsidy_pool += actual_penalty
                r = agent_ratios[agent_name]
                caps["external_capital"] = max(0, caps.get("external_capital", 0) - actual_penalty * r)
                caps["internal_capital"] = max(0, caps.get("internal_capital", 0) - actual_penalty * (1 - r))
        
        # 3.2 Distribute Pool (Winners take from the pool proportionally)
        total_positive_alpha = sum(alpha for alpha in agent_alphas.values() if alpha > 0)
        
        for agent_name, alpha in agent_alphas.items():
            caps = agent_capital[agent_name]
            if alpha > 0 and total_positive_alpha > 0:
                share_pct = alpha / total_positive_alpha
                reward_amount = subsidy_pool * share_pct
                
                r = agent_ratios[agent_name]
                caps["external_capital"] += reward_amount * r
                caps["internal_capital"] += reward_amount * (1 - r)
                
            caps["allocated_capital"] = caps["external_capital"]
            caps.setdefault("roi_history", []).append(agent_returns[agent_name])

        return agent_capital
