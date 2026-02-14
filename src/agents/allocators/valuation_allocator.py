
from langchain_core.messages import HumanMessage
from src.schemas import PortfolioDecision
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_financial_metrics, search_line_items, get_market_cap
from src.utils.api_key import get_api_key_from_state
from src.utils.llm import call_llm
from src.utils.progress import progress
import json
import statistics

# Revert to importing from valuation.py
from src.agents.valuation import (
    calculate_owner_earnings_value,
    calculate_ev_ebitda_value,
    calculate_residual_income_value,
    calculate_wacc,
    calculate_dcf_scenarios
)

def valuation_allocator(state: AgentState, agent_id: str = "valuation_allocator"):
    """
    Global Batch Valuation Agent.
    Calculates Intrinsic Value using multiple models and allocates capital to undervalued assets.
    """
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    
    progress.update_status(agent_id, "ALL", "Running Valuation Models")
    
    universe_summaries = []
    
    for ticker in tickers:
        # Fetch Metrics
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=1, api_key=api_key)
        if not metrics:
             universe_summaries.append(f"Stock {ticker}: No financial metrics found.")
             continue
        m = metrics[0]
        
        # Fetch Line Items for DCF
        line_items = search_line_items(
            ticker=ticker, 
            line_items=["free_cash_flow", "net_income", "depreciation_and_amortization", 
                        "capital_expenditure", "working_capital", "total_debt", "cash_and_equivalents"],
            end_date=end_date, period="ttm", limit=4, api_key=api_key
        )
        if not line_items:
            universe_summaries.append(f"Stock {ticker}: No line items found.")
            continue
            
        li_curr = line_items[0]
        
        # --- Run Models ---
        # 1. Owner Earnings
        wc_change = 0
        if len(line_items) > 1 and getattr(li_curr, 'working_capital', None) and getattr(line_items[1], 'working_capital', None):
             wc_change = li_curr.working_capital - line_items[1].working_capital
             
        owner_val = calculate_owner_earnings_value(
            getattr(li_curr, 'net_income', 0), 
            getattr(li_curr, 'depreciation_and_amortization', 0), 
            getattr(li_curr, 'capital_expenditure', 0), 
            wc_change, 
            growth_rate=m.earnings_growth or 0.05
        )
        
        # 2. EV/EBITDA
        ev_val = calculate_ev_ebitda_value(metrics)
        
        # 3. DCF (Simplified for this agent wrapper, using helper)
        # Convert line items to FCF history list
        fcf_history = [getattr(li, 'free_cash_flow', 0) for li in line_items if getattr(li, 'free_cash_flow', None)]
        wacc = calculate_wacc(
            m.market_cap or 0, 
            getattr(li_curr, 'total_debt', 0), 
            getattr(li_curr, 'cash_and_equivalents', 0), 
            m.interest_coverage, 
            m.debt_to_equity
        )
        
        dcf_results = calculate_dcf_scenarios(
            fcf_history, 
            {'revenue_growth': m.revenue_growth, 'fcf_growth': m.free_cash_flow_growth, 'earnings_growth': m.earnings_growth},
            wacc, m.market_cap or 0
        )
        dcf_val = dcf_results['expected_value']
        
        # 4. Residual Income Model
        rim_val = calculate_residual_income_value(
            market_cap=m.market_cap,
            net_income=getattr(li_curr, 'net_income', 0),
            price_to_book_ratio=m.price_to_book_ratio,
            book_value_growth=m.book_value_growth or 0.03
        )

        # --- Gap Analysis ---
        market_cap = get_market_cap(ticker, end_date, api_key=api_key) or 1
        
        # Calculate Weighted Intrinsic Value
        # Weighting aligned with Valuation Analyst: DCF 35%, Owner 35%, EV 20%, RIM 10%
        intrinsic_value = (dcf_val * 0.35) + (owner_val * 0.35) + (ev_val * 0.20) + (rim_val * 0.10)
        gap = (intrinsic_value - market_cap) / market_cap
        
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
            f"  - Price (Market Cap): ${market_cap:,.0f}\n"
            f"  - Intrinsic Value: ${intrinsic_value:,.0f} (Gap: {gap:+.1%})\n"
            f"  - Breakdown: DCF ${dcf_val:,.0f} | Owner Earnings ${owner_val:,.0f} | EV/EBITDA Implied ${ev_val:,.0f} | Residual Income ${rim_val:,.0f}\n"
            f"  - Key Inputs: WACC {wacc:.1%} | Exp. Growth {(m.earnings_growth or 0):.1%}"
        )
        universe_summaries.append(summary)

    # Construct Context
    study_notes = "\n\n".join(universe_summaries)
    
    # A/B Prompts
    prompt_version = state.get("metadata", {}).get("prompt_version", "wealth")
    
    base_instruction = ""
    if prompt_version == "standard":
        base_instruction = (
            "You are a Value Investor Portfolio Manager. "
            "Your objective is to allocate capital to the most undervalued assets to maximize returns.\n"
            "Allocate $100 across these assets (plus 'CASH')."
        )
    else:
        base_instruction = (
            "You are a Value Investor (Buffett/Graham Style) with $100 capital. Your goal is to maximize your wealth long-term.\n"
            "Your decisions have financial consequences: Buying overvalued stocks is the surest way to lose money.\n"
            "Margin of Safety is mandatory. If no stock offers a significant discount (e.g. >20% Upside), Protect your wealth by allocating to CASH.\n"
            "Allocate capital based on your conviction in the valuation gap."
        )

    prompt = (
        f"{base_instruction}\n\n"
        f"Valuation Data:\n{study_notes}\n\n"
        f"Constraints:\n"
        f"1. Total Amount (Stocks + CASH) must equal 100.0.\n"
        f"2. Assign a Direction (up/down/neutral). 'up' = Undervalued (Buy), 'down' = Overvalued (Sell/Short).\n"
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
        show_agent_reasoning(result, "Valuation Allocator")
        
    progress.update_status(agent_id, None, "Done")
    return {"messages": [msg], "data": state["data"]}
