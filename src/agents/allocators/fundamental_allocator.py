
from langchain_core.messages import HumanMessage
from src.schemas import PortfolioDecision, Allocation
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
    risk_free_rate = data.get("risk_free_rate", 0.0)
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
        
        # Prepare Hints
        tasks = data.get("tasks", [])
        hint_map = {t['ticker']: t for t in tasks}
        
        hint_str = ""
        if ticker in hint_map:
            task = hint_map[ticker]
            action = task.get('action', 'analyze')
            if action != 'analyze':
                hint_str = f"  - Quantitative Signal: {action.upper()} (Reason: {task.get('reason', 'N/A')})\n"
        
        summary = (
            f"Stock {ticker}:\n"
            f"{hint_str}"
            f"  - Profitability: ROE {(m.return_on_equity or 0):.1%} | Net Margin {(m.net_margin or 0):.1%} | Op Margin {(m.operating_margin or 0):.1%}\n"
            f"  - Growth: Rev Growth {(m.revenue_growth or 0):.1%} | Earnings Growth {(m.earnings_growth or 0):.1%} | Book Value Growth {(m.book_value_growth or 0):.1%}\n"
            f"  - Health: D/E {(m.debt_to_equity or 0):.2f} | Current Ratio {(m.current_ratio or 0):.2f} | FCF/Share ${(m.free_cash_flow_per_share or 0):.2f} vs EPS ${(m.earnings_per_share or 0):.2f}\n"
            f"  - Valuation: P/E {(m.price_to_earnings_ratio or 0):.1f} | P/B {(m.price_to_book_ratio or 0):.1f} | P/S {(m.price_to_sales_ratio or 0):.1f}"
        )
        universe_summaries.append(summary)

    # Construct Global Context
    annual_rf = risk_free_rate * 252
    cash_summary = (
        f"Asset CASH:\n"
        f"  - Price: $1.00\n"
        f"  - Guaranteed Daily Risk-Free Rate: {risk_free_rate:.6f} (Annualized: {annual_rf:.2%})\n"
        f"  - Note: 'long' means earning this rate. 'short' means paying this rate to borrow capital."
    )
    universe_summaries.append(cash_summary)
    study_notes = "\n\n".join(universe_summaries)
    
    # Select Prompt based on A/B Test Config
    prompt_version = state.get("metadata", {}).get("prompt_version", "wealth")
    
    base_instruction = ""
    if prompt_version == "standard":
        base_instruction = (
            "You are a Fundamental Portfolio Manager. "
            "Your goal is to maximize return by investing in high-quality, reasonably valued companies with growth potential.\n"
            f"Your universe includes the provided stocks AND a 'CASH' asset (which has a known daily risk-free rate of {risk_free_rate:.6f}).\n"
            "Allocate $100 across these assets based on their fundamental strength."
        )
    else:
        # Wealth / Skin-in-the-game
        base_instruction = (
            "You are a Fundamental Investor with $100 capital. Your goal is to maximize your personal wealth.\n"
            "Your decisions have financial consequences. "
            f"Your universe includes the provided stocks AND a 'CASH' asset (which has a known daily risk-free rate of {risk_free_rate:.6f}).\n"
            "Allocate capital based on your conviction. Treat CASH as a peer asset with a guaranteed return."
        )

    prompt = (
        f"{base_instruction}\n\n"
        f"Universe Data:\n{study_notes}\n\n"
        f"Constraints:\n"
        f"1. MATHEMATICAL RULE: The Net Exposure MUST exactly equal 100.0.\n"
        f"   Calculation: (Sum of ALL 'long' amounts) - (Sum of ALL 'short' amounts) = 100.0\n"
        f"2. IMPORTANT: Every single 'amount' MUST be a strictly POSITIVE number (e.g., 50.0, never -50.0). The 'direction' field ('long' or 'short') handles the math sign.\n"
        f"3. GROSS EXPOSURE LIMIT: To prevent excessive risk, the sum of ALL amounts (long + short) should not exceed 1000.0.\n"
        f"4. MATH EXAMPLES:\n"
        f"   - Leverage: Long Stocks $150, Short CASH $50. Math: 150 - 50 = 100.0.\n"
        f"   - Hedging: Long Stocks $120, Short Stocks $20, Long CASH $0. Math: 120 - 20 = 100.0.\n"
        f"   - Pure Cash: Long CASH $100. Math: 100 - 0 = 100.0.\n"
        f"5. For stocks: 'long' = Buy (Good Fundamentals), 'short' = Sell (Bad Fundamentals/Overvalued).\n"
        f"6. For CASH: 'long' = Lending/Holding cash to earn the risk-free rate, 'short' = Borrowing cash to deploy leverage.\n"
        f"7. Do NOT allocate to an asset if your conviction is low. Be decisive."
    )
    
    # Call LLM
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

    # Output
    result = {
        "allocations": [a.model_dump() for a in decision.allocations],
        "metrics": {
            "original_net_exposure": original_net,
            "original_gross_exposure": original_gross
        }
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
