
from langchain_core.messages import HumanMessage
from src.schemas import PortfolioDecision, Allocation
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
    calculate_volatility_signals,
    calculate_stat_arb_signals
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
    risk_free_rate = data.get("risk_free_rate", 0.0)
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
        stat_arb = calculate_stat_arb_signals(df)
        
        # Prepare Hints
        tasks = data.get("tasks", [])
        hint_map = {t['ticker']: t for t in tasks}
        
        hint_str = ""
        if ticker in hint_map:
            task = hint_map[ticker]
            action = task.get('action', 'analyze')
            if action != 'analyze':
                hint_str = f"  - Quantitative Signal: {action.upper()} (Reason: {task.get('reason', 'N/A')}). Note: Use this only as a reference. You MUST make your own judgment based on your specific strategy.\n"
        
        # Format Summary
        summary = (
            f"Stock {ticker}:\n"
            f"{hint_str}"
            f"  - Trend: {trend['signal'].upper()} (Confidence {trend['confidence']:.0%})\n"
            f"  - Momentum: {momentum['signal'].upper()} (Confidence {momentum['confidence']:.0%})\n"
            f"  - Mean Reversion: {reversion['signal'].upper()} (RSI: {reversion['metrics']['rsi_14']:.1f})\n"
            f"  - Volatility: {vol['signal'].upper()} (Context: {vol['metrics']['volatility_regime']:.2f}x Normal)\n"
            f"  - Stat Arb: {stat_arb['signal'].upper()} (Skew: {stat_arb['metrics']['skewness']:.2f}, Hurst: {stat_arb['metrics']['hurst_exponent']:.2f})"
        )
        universe_summaries.append(summary)

    # Construct Context
    annual_rf = risk_free_rate * 252
    cash_summary = (
        f"Asset CASH:\n"
        f"  - Price: $1.00\n"
        f"  - Guaranteed Daily Risk-Free Rate: {risk_free_rate:.6f} (Annualized: {annual_rf:.2%})\n"
        f"  - Note: 'long' means earning this rate. 'short' means paying this rate to borrow capital."
    )
    universe_summaries.append(cash_summary)
    study_notes = "\n\n".join(universe_summaries)
    
    # Select Prompt based on A/B Config
    prompt_version = state.get("metadata", {}).get("prompt_version", "wealth")
    
    base_instruction = ""
    if prompt_version == "standard":
        base_instruction = (
            "You are a Technical Analyst Portfolio Manager. "
            "Your objective is to maximize returns by capturing strong trends and momentum.\n"
            f"Your universe includes the provided stocks AND a 'CASH' asset (which has a known daily risk-free rate of {risk_free_rate:.6f}).\n"
            "Allocate $100 across these assets based on their technical setups."
        )
    else:
        base_instruction = (
            "You are a Technical Trader with $100 capital. Your goal is to maximize your wealth.\n"
            "Your decisions have financial consequences. "
            f"Your universe includes the provided stocks AND a 'CASH' asset (which has a known daily risk-free rate of {risk_free_rate:.6f}).\n"
            "Allocate capital based on your conviction. Treat CASH as a peer asset with a guaranteed return."
        )

    prompt = (
        f"{base_instruction}\n\n"
        f"Technical Analysis Data:\n{study_notes}\n\n"
        f"Constraints:\n"
        f"1. MATHEMATICAL RULE: The Net Exposure MUST exactly equal 100.0.\n"
        f"   Calculation: (Sum of ALL 'long' amounts) - (Sum of ALL 'short' amounts) = 100.0\n"
        f"2. IMPORTANT: Every single 'amount' MUST be a strictly POSITIVE number (e.g., 50.0, never -50.0). The 'direction' field ('long' or 'short') handles the math sign.\n"
        f"3. GROSS EXPOSURE LIMIT: To prevent excessive risk, the sum of ALL amounts (long + short) should not exceed 1000.0.\n"
        f"4. MATH EXAMPLES:\n"
        f"   - Leverage: Long Stocks $150, Short CASH $50. Math: 150 - 50 = 100.0.\n"
        f"   - Hedging: Long Stocks $120, Short Stocks $20, Long CASH $0. Math: 120 - 20 = 100.0.\n"
        f"   - Pure Cash: Long CASH $100. Math: 100 - 0 = 100.0.\n"
        f"5. For stocks: 'long' = Bullish Setup, 'short' = Bearish Setup.\n"
        f"6. For CASH: 'long' = Lending/Holding cash to earn the risk-free rate, 'short' = Borrowing cash to deploy leverage.\n"
        f"7. Do NOT allocate to an asset if your conviction is low. Be decisive.\n"
        f"8. STAYING OUT: If you decide the market is too risky and want to hold no stocks, you MUST explicitly output a single allocation: 'long' CASH for 100.0. Do NOT output an empty list.\n"
        f"9. NO SPLIT CASH: Do not split CASH into multiple allocations. Provide only ONE aggregated row for CASH (either 'long' or 'short')."
    )
    
    decision = call_llm(prompt, PortfolioDecision, agent_name=agent_id, state=state)
    
    # Record original metrics before normalization
    original_net = sum(abs(a.amount) if a.direction == 'long' else -abs(a.amount) for a in decision.allocations)
    original_gross = sum(abs(a.amount) for a in decision.allocations)

    # Normalize Net Exposure to 100.0
    net_exposure = original_net
    if net_exposure > 0 and abs(net_exposure - 100.0) > 0.1:
        scale = 100.0 / net_exposure
        for a in decision.allocations:
            a.amount = abs(a.amount) * scale
    elif net_exposure <= 0:
        # Fallback: Invalid net exposure. Override and put 100% in CASH.
        print(f"[{agent_id}] Invalid Net Exposure ({net_exposure}). Defaulting to 100% CASH.")
        decision.allocations = [
            Allocation(
                ticker="CASH",
                direction="long",
                amount=100.0,
                reasoning=f"FALLBACK: Model produced invalid Net Exposure ({net_exposure}). Preserving wealth."
            )
        ]
    else:
        for a in decision.allocations:
            a.amount = abs(a.amount)

    result = {
        "allocations": [a.model_dump() for a in decision.allocations],
        "metrics": {
            "original_net_exposure": original_net,
            "original_gross_exposure": original_gross
        }
    }
    
    if "allocator_decisions" not in state["data"]:
        state["data"]["allocator_decisions"] = {}
    state["data"]["allocator_decisions"][agent_id] = result
    
    msg = HumanMessage(content=json.dumps(result), name=agent_id)
    if state.get("metadata", {}).get("show_reasoning"):
        show_agent_reasoning(result, "Technical Allocator")
        
    progress.update_status(agent_id, None, "Done")
    return {"messages": [msg], "data": state["data"]}
