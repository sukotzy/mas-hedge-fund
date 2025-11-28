import json
from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from pydantic import BaseModel, Field
from typing_extensions import Literal
from src.utils.progress import progress

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
    
    # 1. Separate candidates
    long_candidates = []
    short_candidates = []
    
    for ticker in tickers:
        value = consensus_values.get(ticker, 0.0)
        price = current_prices.get(ticker, 0.0)
        limit = max_shares.get(ticker, 0) # Use max shares directly
        
        if price <= 0:
            continue
            
        if value > 0:
            long_candidates.append({"ticker": ticker, "value": value, "price": price, "limit": limit})
        elif value < 0:
            short_candidates.append({"ticker": ticker, "value": value, "price": price, "limit": limit})
            
    # 2. Sort by conviction (absolute value)
    long_candidates.sort(key=lambda x: x["value"], reverse=True)
    short_candidates.sort(key=lambda x: abs(x["value"]), reverse=True)
    
    # 3. Initial Allocation (Max out up to limit)
    long_allocations = {}
    short_allocations = {}
    total_long_amt = 0.0
    total_short_amt = 0.0
    
    for item in long_candidates:
        # Allocate max shares
        qty = item["limit"]
        long_allocations[item["ticker"]] = qty
        total_long_amt += qty * item["price"]
        
    for item in short_candidates:
        qty = item["limit"]
        short_allocations[item["ticker"]] = qty
        total_short_amt += qty * item["price"]
        
    # 4. Enforce Market Neutrality (Self-financing: Long = Short)
    # Scale down the larger side
    if total_long_amt > total_short_amt and total_short_amt > 0:
        scale = total_short_amt / total_long_amt
        for t in long_allocations:
            long_allocations[t] = int(long_allocations[t] * scale)
    elif total_short_amt > total_long_amt and total_long_amt > 0:
        scale = total_long_amt / total_short_amt
        for t in short_allocations:
            short_allocations[t] = int(short_allocations[t] * scale)
    elif total_long_amt == 0 or total_short_amt == 0:
        # If one side is empty, we can't be neutral.
        # For demo, we'll allow directional bets if one side is missing, 
        # or we could zero out. Let's allow it but maybe scale to cash?
        # Assuming cash is sufficient since we used max_shares which considers position limits.
        pass

    # 5. Generate Decisions
    for ticker in tickers:
        action = "hold"
        quantity = 0
        reasoning = "Neutral consensus or no opportunity"
        
        if ticker in long_allocations and long_allocations[ticker] > 0:
            quantity = long_allocations[ticker]
            action = "buy"
            reasoning = f"Long conviction (Value: {consensus_values.get(ticker, 0):.2f})"
        elif ticker in short_allocations and short_allocations[ticker] > 0:
            quantity = short_allocations[ticker]
            action = "short"
            reasoning = f"Short conviction (Value: {consensus_values.get(ticker, 0):.2f})"
                
        decisions[ticker] = PortfolioDecision(
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
