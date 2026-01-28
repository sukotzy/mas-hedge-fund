
from langchain_core.messages import HumanMessage
from src.schemas import PortfolioDecision
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_prices, prices_to_df
from src.utils.api_key import get_api_key_from_state
from src.utils.llm import call_llm
from src.utils.progress import progress
import json
import pandas as pd

# Import Technical Math from existing agent to ensure consistency
from src.agents.technicals import (
    calculate_trend_signals,
    calculate_mean_reversion_signals,
    calculate_momentum_signals,
    calculate_volatility_signals
)

def technical_allocator(state: AgentState, agent_id: str = "technical_allocator"):
    """
    Global Batch Technical Agent.
    Analyzes price action/momentum for the entire universe and allocates capital.
    """
    data = state.get("data", {})
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    tickers = data.get("tickers")
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    
    progress.update_status(agent_id, "ALL", "Fetching Price Data & Indicators")
    
    universe_summaries = []
    
    for ticker in tickers:
        prices = get_prices(ticker, start_date, end_date, api_key=api_key)
        if not prices:
            universe_summaries.append(f"Stock {ticker}: No price data found.")
            continue
            
        df = prices_to_df(prices)
        if df.empty or len(df) < 50:
             universe_summaries.append(f"Stock {ticker}: Insufficient price history.")
             continue
             
        # Run Math
        trend = calculate_trend_signals(df)
        reversion = calculate_mean_reversion_signals(df)
        momentum = calculate_momentum_signals(df)
        vol = calculate_volatility_signals(df)
        
        # Format Summary
        summary = (
            f"Stock {ticker}:\n"
            f"  - Trend: {trend['signal'].upper()} (Confidence {trend['confidence']:.0%})\n"
            f"  - Momentum: {momentum['signal'].upper()} (Confidence {momentum['confidence']:.0%})\n"
            f"  - Mean Reversion: {reversion['signal'].upper()} (RSI: {reversion['metrics']['rsi_14']:.1f})\n"
            f"  - Volatility: {vol['signal'].upper()} (Context: {vol['metrics']['volatility_regime']:.2f}x Normal)"
        )
        universe_summaries.append(summary)

    # Construct Context
    study_notes = "\n\n".join(universe_summaries)
    
    # Select Prompt based on A/B Config
    prompt_version = state.get("metadata", {}).get("prompt_version", "wealth")
    
    base_instruction = ""
    if prompt_version == "standard":
        base_instruction = (
            "You are a Technical Analyst Portfolio Manager. "
            "Your objective is to maximize returns by capturing strong trends and momentum.\n"
            "Allocate $100 across these assets (plus 'CASH')."
        )
    else:
        base_instruction = (
            "You are a Technical Trader with $100 capital. Your goal is to maximize your wealth.\n"
            "Your decisions have financial consequences: Betting against the trend or catching falling knives will destroy your capital.\n"
            "Risk Management is key. If charts look messy or weak, Protect your wealth by allocating to CASH.\n"
            "Allocate capital based on your conviction in the technical setup."
        )

    prompt = (
        f"{base_instruction}\n\n"
        f"Technical Analysis Data:\n{study_notes}\n\n"
        f"Constraints:\n"
        f"1. Total Amount (Stocks + CASH) must equal 100.0.\n"
        f"2. Assign a Direction (up/down/neutral). 'up' = Bullish Setup, 'down' = Bearish Setup.\n"
        f"3. Be decisive."
    )
    
    decision = call_llm(prompt, PortfolioDecision, agent_name=agent_id, state=state)
    
    # Normalize
    total = sum(a.amount for a in decision.allocations)
    if total > 0 and abs(total - 100.0) > 0.1:
        scale = 100.0 / total
        for a in decision.allocations:
            a.amount *= scale

    result = {
        "allocations": [a.model_dump() for a in decision.allocations],
        "metrics": {"original_total": total}
    }
    
    if "allocator_decisions" not in state["data"]:
        state["data"]["allocator_decisions"] = {}
    state["data"]["allocator_decisions"][agent_id] = result
    
    msg = HumanMessage(content=json.dumps(result), name=agent_id)
    if state.get("metadata", {}).get("show_reasoning"):
        show_agent_reasoning(result, "Technical Allocator")
        
    progress.update_status(agent_id, None, "Done")
    return {"messages": [msg], "data": state["data"]}
