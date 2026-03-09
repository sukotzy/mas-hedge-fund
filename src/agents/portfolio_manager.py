import json
from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from pydantic import BaseModel, Field
from typing_extensions import Literal
from src.utils.progress import progress
import pandas as pd
from src.tools.api import get_price_data
from src.utils.optimizer_utils import calculate_optimal_portfolio

class PortfolioDecision(BaseModel):
    action: Literal["buy", "sell", "short", "cover", "hold"]
    quantity: int = Field(description="Number of shares to trade")
    confidence: int = Field(description="Confidence 0-100")
    reasoning: str = Field(description="Reasoning for the decision")


class PortfolioManagerOutput(BaseModel):
    decisions: dict[str, PortfolioDecision] = Field(description="Dictionary of ticker to trading decisions")


##### Portfolio Management Agent #####
def portfolio_management_agent(state: AgentState, agent_id: str = "portfolio_manager"):
    """Makes final trading decisions and generates orders for multiple tickers based on betting market consensus."""

    portfolio = state["data"]["portfolio"]
    analyst_signals = state["data"]["analyst_signals"]
    tickers = state["data"]["tickers"]
    consensus_values = state["data"].get("consensus_values", {})

    position_limits = {}
    current_prices = {}
    max_shares = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Processing risk limits")

        # Find the corresponding risk manager for this portfolio manager
        if agent_id.startswith("portfolio_manager_"):
            suffix = agent_id.split('_')[-1]
            risk_manager_id = f"risk_management_agent_{suffix}"
        else:
            risk_manager_id = "risk_management_agent"  # Fallback for CLI

        risk_data = analyst_signals.get(risk_manager_id, {}).get(ticker, {})
        position_limits[ticker] = risk_data.get("remaining_position_limit", 0.0)
        current_prices[ticker] = float(risk_data.get("current_price", 0.0))

        # Calculate maximum shares allowed based on position limit and price
        if current_prices[ticker] > 0:
            max_shares[ticker] = int(position_limits[ticker] // current_prices[ticker])
        else:
            max_shares[ticker] = 0

    state["data"]["current_prices"] = current_prices

    progress.update_status(agent_id, None, "Optimizing portfolio")

    # Deterministic Optimization Logic
    decisions = {}
    
    # 1. Fetch 15-day trailing prices for decay logic
    prices_history = {}
    end_date = state["data"].get("end_date", pd.Timestamp.today().strftime("%Y-%m-%d"))
    dt_end = pd.Timestamp(end_date)
    dt_start = dt_end - pd.Timedelta(days=15)
    
    for t in tickers:
        if t == "CASH":
            continue
        try:
            df = get_price_data(t, dt_start.strftime("%Y-%m-%d"), end_date)
            prices_history[t] = df
        except Exception as e:
            prices_history[t] = pd.DataFrame()

    # 2. Extract previous holdings (net positions)
    previous_holdings = {}
    positions = portfolio.get("positions", {})
    for t in tickers:
        if t == "CASH":
            continue
        if t in positions:
            pos = positions[t]
            net_holding = pos.get("long", 0) - pos.get("short", 0)
            previous_holdings[t] = net_holding
        else:
            previous_holdings[t] = 0

    # 3. Retrieve previous consensus
    previous_consensus = state["data"].get("previous_consensus", {})

    # 4. Prepare risk limits in USD
    risk_limits_usd = {}
    for t in tickers:
        if t == "CASH":
            continue
        risk_limits_usd[t] = position_limits.get(t, 0.0)

    # 5. Call Optimization
    initial_capital = portfolio.get("cash", 100000.0)
    
    optimal_shares, adjusted_consensus = calculate_optimal_portfolio(
        today_consensus=consensus_values,
        previous_consensus=previous_consensus,
        previous_holdings=previous_holdings,
        prices_history=prices_history,
        risk_limits=risk_limits_usd,
        initial_capital=initial_capital,
        risk_free_rate=0.0
    )

    # 6. Store updated consensus back in state directly
    state["data"]["previous_consensus"] = adjusted_consensus

    # 7. Generate Decisions using Delta Math
    for t in tickers:
        if t == "CASH":
            continue
            
        target_net_shares = optimal_shares.get(t, 0.0)
        net_holding = previous_holdings.get(t, 0)
        delta = target_net_shares - net_holding
        
        # We need integer shares
        delta = int(round(delta))
        
        action = "hold"
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
                
        reasoning = f"Optimized target net shares: {int(target_net_shares)}. Adjusted signal: {adjusted_consensus.get(t, 0):.2f}"
        
        decisions[t] = PortfolioDecision(
            action=action,
            quantity=quantity,
            confidence=100, # Deterministic
            reasoning=reasoning
        )

    message = HumanMessage(
        content=json.dumps({ticker: decision.model_dump() for ticker, decision in decisions.items()}),
        name=agent_id,
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning({ticker: decision.model_dump() for ticker, decision in decisions.items()},
                             "Portfolio Manager")

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }
