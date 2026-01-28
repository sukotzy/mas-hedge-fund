
from langchain_core.messages import HumanMessage
from src.schemas import PortfolioDecision
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_financial_metrics
from src.utils.api_key import get_api_key_from_state
from src.utils.llm import call_llm
from src.utils.progress import progress
import json
from datetime import datetime

def fundamental_allocator(state: AgentState, agent_id: str = "fundamental_allocator"):
    """
    Global Batch Fundamental Agent.
    Analyzes financial metrics (Profitability, Growth, Health, Value) for the entire universe
    and allocates capital efficiently.
    """
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    
    progress.update_status(agent_id, "ALL", "Fetching Financial Metrics")
    
    universe_summaries = []
    
    for ticker in tickers:
        # Fetch metrics ( reusing the logic from fundamentals.py )
        metrics_list = get_financial_metrics(ticker, end_date, period="ttm", limit=1, api_key=api_key)
        if not metrics_list:
            universe_summaries.append(f"Stock {ticker}: No financial data found.")
            continue
            
        m = metrics_list[0]
        
        # Format key metrics for the LLM
        # We don't need to hardcode thresholds here; let the LLM judge "Good vs Bad" based on the data.
        # But providing context (e.g. "High ROE") helps. 
        
        summary = (
            f"Stock {ticker}:\n"
            f"  - Profitability: ROE {(m.return_on_equity or 0):.1%} | Net Margin {(m.net_margin or 0):.1%} | Op Margin {(m.operating_margin or 0):.1%}\n"
            f"  - Growth: Rev Growth {(m.revenue_growth or 0):.1%} | Earnings Growth {(m.earnings_growth or 0):.1%}\n"
            f"  - Health: D/E {(m.debt_to_equity or 0):.2f} | Current Ratio {(m.current_ratio or 0):.2f}\n"
            f"  - Valuation: P/E {(m.price_to_earnings_ratio or 0):.1f} | P/B {(m.price_to_book_ratio or 0):.1f} | P/S {(m.price_to_sales_ratio or 0):.1f}"
        )
        universe_summaries.append(summary)

    # Construct Global Context
    study_notes = "\n\n".join(universe_summaries)
    
    # Select Prompt based on A/B Test Config
    prompt_version = state.get("metadata", {}).get("prompt_version", "wealth")
    
    base_instruction = ""
    if prompt_version == "standard":
        base_instruction = (
            "You are a Fundamental Portfolio Manager. "
            "Your goal is to maximize return by investing in high-quality, reasonably valued companies with growth potential.\n"
            "Allocate $100 across these assets (plus 'CASH' for weak opportunities)."
        )
    else:
        # Wealth / Skin-in-the-game
        base_instruction = (
            "You are a Fundamental Investor with $100 capital. Your goal is to maximize your personal wealth.\n"
            "Your decisions have financial consequences: Investing in poor quality or overvalued stocks will destroy your capital.\n"
            "Protect your wealth: If no stocks meet your strict quality/value criteria, allocate to CASH.\n"
            "Allocate capital based on your conviction in the fundamental strength (Quality + Value + Growth)."
        )

    prompt = (
        f"{base_instruction}\n\n"
        f"Universe Data:\n{study_notes}\n\n"
        f"Constraints:\n"
        f"1. Total Amount (Stocks + CASH) must equal 100.0.\n"
        f"2. Assign a Direction (up/down/neutral). 'up' = Long (Good Fundamentals), 'down' = Short (Bad Fundamentals/Overvalued).\n"
        f"3. Be decisive."
    )
    
    # Call LLM
    decision = call_llm(prompt, PortfolioDecision, agent_name=agent_id, state=state)
    
    # Normalize
    total = sum(a.amount for a in decision.allocations)
    if total > 0 and abs(total - 100.0) > 0.1:
        scale = 100.0 / total
        for a in decision.allocations:
            a.amount *= scale

    # Output
    result = {
        "allocations": [a.model_dump() for a in decision.allocations],
        "metrics": {"original_total": total}
    }
    
    # Store
    if "allocator_decisions" not in state["data"]:
        state["data"]["allocator_decisions"] = {}
    state["data"]["allocator_decisions"][agent_id] = result
    
    # Message for graph
    msg = HumanMessage(content=json.dumps(result), name=agent_id)
    if state.get("metadata", {}).get("show_reasoning"):
        show_agent_reasoning(result, "Fundamental Allocator")
        
    progress.update_status(agent_id, None, "Done")
    return {"messages": [msg], "data": state["data"]}
