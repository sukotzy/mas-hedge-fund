
from langchain_core.messages import HumanMessage
from src.schemas import PortfolioDecision, Allocation
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_financial_metrics, search_line_items
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
    risk_free_rate = data.get("risk_free_rate", 0.0)
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    
    progress.update_status(agent_id, "ALL", "Running Valuation Models")
    
    universe_summaries = []
    
    for ticker in tickers:
        # 1. Fetch Metrics
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=1, api_key=api_key)
        if not metrics:
             universe_summaries.append(f"Stock {ticker}: No financial metrics found.")
             continue
        m = metrics[0]
        
        # Fix 1: Use existing market_cap and skip if missing or <= 0
        market_cap = m.market_cap
        if not market_cap or market_cap <= 0:
             universe_summaries.append(f"Stock {ticker}: Market Cap data missing. Cannot evaluate.")
             continue
        
        # 2. Fetch Line Items for DCF
        metrics_history = search_line_items(
            ticker=ticker, 
            line_items=[], # No longer needed by local loader
            end_date=end_date, period="ttm", limit=4, api_key=api_key
        )
        if not metrics_history:
            universe_summaries.append(f"Stock {ticker}: No financial metrics found.")
            continue
            
        vm_curr = metrics_history[0]
        vm_prev = metrics_history[1] if len(metrics_history) > 1 else None
        
        # Thoroughly filter None values for safety
        net_income = vm_curr.net_income or 0
        depreciation = vm_curr.depreciation_and_amortization or 0
        capex = vm_curr.capital_expenditure or 0
        total_debt = vm_curr.total_debt or 0
        cash_equiv = vm_curr.cash_and_equivalents or 0
        
        # --- Run Models ---
        wc_change = 0
        if vm_curr.working_capital is not None and vm_prev is not None and vm_prev.working_capital is not None:
             wc_change = vm_curr.working_capital - vm_prev.working_capital
             
        owner_val = calculate_owner_earnings_value(
            net_income, 
            depreciation, 
            capex, 
            wc_change, 
            growth_rate=m.earnings_growth or 0.05
        )
        
        # 2. EV/EBITDA
        ev_val = calculate_ev_ebitda_value(metrics)
        
        # 3. DCF
        fcf_history = [vm.free_cash_flow or 0 for vm in metrics_history]
        
        wacc = calculate_wacc(
            market_cap, 
            total_debt, 
            cash_equiv, 
            m.interest_coverage or 0, 
            m.debt_to_equity or 0
        )
        
        dcf_results = calculate_dcf_scenarios(
            fcf_history, 
            {'revenue_growth': m.revenue_growth or 0, 'fcf_growth': m.free_cash_flow_growth or 0, 'earnings_growth': m.earnings_growth or 0},
            wacc, 
            market_cap
        )
        dcf_val = dcf_results.get('expected_value', 0)
        
        # 4. Residual Income Model
        rim_val = calculate_residual_income_value(
            market_cap=market_cap,
            net_income=net_income,
            price_to_book_ratio=m.price_to_book_ratio or 0,
            book_value_growth=m.book_value_growth or 0.03
        )

        # --- Gap Analysis ---
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
                hint_str = f"  - Quantitative Signal: {action.upper()} (Reason: {task.get('reason', 'N/A')}). Note: Use this only as a reference. You MUST make your own judgment based on your specific strategy.\n"
                
        val_str = f"${intrinsic_value:,.0f}" if intrinsic_value > 0 else "N/A (Missing Financials)"
        gap_str = f"{gap:+.1%}" if intrinsic_value > 0 else "N/A"
        summary = (
            f"Stock {ticker}:\n"
            f"{hint_str}"
            f"  - Price (Market Cap): ${market_cap:,.0f}\n"
            f"  - Intrinsic Value: {val_str} (Gap: {gap_str})\n"
            f"  - Breakdown: DCF ${dcf_val:,.0f} | Owner Earnings ${owner_val:,.0f} | EV/EBITDA Implied ${ev_val:,.0f} | Residual Income ${rim_val:,.0f}\n"
            f"  - Key Inputs: WACC {wacc:.1%} | Exp. Growth {(m.earnings_growth or 0):.1%}"
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
    
    # A/B Prompts
    prompt_version = state.get("metadata", {}).get("prompt_version", "wealth")
    
    base_instruction = ""
    if prompt_version == "standard":
        base_instruction = (
            "You are a Value Investor Portfolio Manager. "
            "Your objective is to allocate capital to the most undervalued assets to maximize returns.\n"
            f"Your universe includes the provided stocks AND a 'CASH' asset (which has a known daily risk-free rate of {risk_free_rate:.6f}).\n"
            "Allocate $100 across these assets based on their valuation."
        )
    else:
        base_instruction = (
            "You are a Value Investor (Buffett/Graham Style) with $100 capital. Your goal is to maximize your wealth long-term.\n"
            "Your decisions have financial consequences. "
            f"Your universe includes the provided stocks AND a 'CASH' asset (which has a known daily risk-free rate of {risk_free_rate:.6f}).\n"
            "Allocate capital based on your conviction. Treat CASH as a peer asset with a guaranteed return."
        )

    prompt = (
        f"{base_instruction}\n\n"
        f"Valuation Data:\n{study_notes}\n\n"
        f"Constraints:\n"
        f"1. MATHEMATICAL RULE: The Net Exposure MUST exactly equal 100.0.\n"
        f"   Calculation: (Sum of ALL 'long' amounts) - (Sum of ALL 'short' amounts) = 100.0\n"
        f"2. IMPORTANT: Every single 'amount' MUST be a strictly POSITIVE number (e.g., 50.0, never -50.0). The 'direction' field ('long' or 'short') handles the math sign.\n"
        f"3. GROSS EXPOSURE LIMIT: To prevent excessive risk, the sum of ALL amounts (long + short) should not exceed 1000.0.\n"
        f"4. MATH EXAMPLES:\n"
        f"   - Leverage: Long Stocks $150, Short CASH $50. Math: 150 - 50 = 100.0.\n"
        f"   - Hedging: Long Stocks $120, Short Stocks $20, Long CASH $0. Math: 120 - 20 = 100.0.\n"
        f"   - Pure Cash: Long CASH $100. Math: 100 - 0 = 100.0.\n"
        f"5. For stocks: 'long' = Undervalued (Buy), 'short' = Overvalued (Sell/Short).\n"
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
        show_agent_reasoning(result, "Valuation Allocator")
        
    progress.update_status(agent_id, None, "Done")
    return {"messages": [msg], "data": state["data"]}
