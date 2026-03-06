from langchain_core.messages import HumanMessage
from src.schemas import PortfolioDecision, Allocation
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_company_news
from src.utils.api_key import get_api_key_from_state
from src.utils.llm import call_llm
from src.utils.progress import progress
import json
from datetime import datetime, timedelta

def news_sentiment_allocator(state: AgentState, agent_id: str = "news_sentiment_allocator"):
    """Global Batch Logic: Fetch all data, then 1 LLM Call for Decision."""
    progress.update_status(agent_id, "ALL", "Fetching Universe Data (Global Batch)")
    
    # Calculate start date (7 days lookback)
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    risk_free_rate = data.get("risk_free_rate", 0.0)

    dt_end = datetime.strptime(end_date, "%Y-%m-%d")
    dt_start = dt_end - timedelta(days=7)
    start_date = dt_start.strftime("%Y-%m-%d")

    universe_content = []
    
    # Prepare Hints
    tasks = data.get("tasks", [])
    hint_map = {t['ticker']: t for t in tasks}
    
    for ticker in tickers:
        # Hint string
        hint_str = ""
        if ticker in hint_map:
            task = hint_map[ticker]
            action = task.get('action', 'analyze')
            if action != 'analyze':
                hint_str = f"Quantitative Signal: {action.upper()} (Reason: {task.get('reason', 'N/A')}). Note: Use this only as a reference. You MUST make your own judgment based on your specific strategy.\n"

        # Pass start_date
        company_news = get_company_news(ticker, end_date, start_date=start_date, limit=5, api_key=api_key)
        if not company_news:
            universe_content.append(f"Stock {ticker}: {hint_str}No significant corporate events in the last 7 days.")
            continue
            
        events_text = "\n".join([f"- {n.date}: {n.title}" for n in company_news])
        universe_content.append(f"Stock {ticker} Events (Last 7 Days):\n{hint_str}{events_text}")

    annual_rf = risk_free_rate * 252
    cash_summary = (
        f"Stock CASH Events (Last 7 Days):\n"
        f"- Guaranteed Daily Risk-Free Rate: {risk_free_rate:.6f} (Annualized: {annual_rf:.2%}).\n"
        f"- Note: 'long' means earning this rate. 'short' means paying this rate to borrow capital."
    )
    universe_content.append(cash_summary)

    full_context = "\n\n".join(universe_content)
    
    # Select Prompt based on metadata (A/B Testing)
    prompt_version = state.get("metadata", {}).get("prompt_version", "wealth")
    
    if prompt_version == "standard":
        # Control Group: Original Prompt
        prompt = (
            f"You are a Event-Driven Trader with $100 capital. "
            f"Analyze the following universe of stocks based on their recent corporate events (Last 7 Days).\n"
            f"Your universe includes these stocks AND a 'CASH' asset (which has a known daily risk-free rate of {risk_free_rate:.6f}).\n"
            f"You must allocate capital across these assets to maximize your betting return. Treat CASH as a peer asset.\n\n"
            f"{full_context}\n\n"
            f"Constraints:\n"
            f"1. MATHEMATICAL RULE: The Net Exposure MUST exactly equal 100.0.\n"
            f"   Calculation: (Sum of ALL 'long' amounts) - (Sum of ALL 'short' amounts) = 100.0\n"
            f"2. IMPORTANT: Every single 'amount' MUST be a strictly POSITIVE number (e.g., 50.0, never -50.0). The 'direction' field ('long' or 'short') handles the math sign.\n"
            f"3. GROSS EXPOSURE LIMIT: The sum of ALL amounts (long + short) MUST NOT exceed 1000.0.\n"
            f"4. MATH EXAMPLES:\n"
            f"   - Leverage: Long Stocks $150, Short CASH $50. Math: 150 - 50 = 100.0.\n"
            f"   - Hedging: Long Stocks $120, Short Stocks $20, Long CASH $0. Math: 120 - 20 = 100.0.\n"
            f"   - All Cash: Long CASH $100. Math: 100 - 0 = 100.0.\n"
            f"5. For stocks: 'long' = Bullish, 'short' = Bearish.\n"
            f"6. For CASH: 'long' = Lending/Holding cash to earn the risk-free rate, 'short' = Borrowing cash to deploy leverage.\n"
            f"7. CONVICTION ALLOCATION: Allocate capital strictly proportionally to your conviction. If there is no clear basis for a judgment on an asset, the allocated amount should be 0.\n"
            f"8. EMPTY PORTFOLIO RULE: If you choose not to hold any stocks, you MUST explicitly output a single allocation: 'long' CASH for 100.0. Do NOT output an empty list.\n"
            f"9. NO SPLIT CASH: Do not split CASH into multiple allocations. Provide only ONE aggregated row for CASH (either 'long' or 'short')."
        )
    else:
        # Variant Group: Wealth Consequence Prompt (Default)
        prompt = (
            f"You are a Event-Driven Trader with $100 capital. Your objective is to maximize your wealth through accurate predictions.\n"
            f"Analyze the following universe of stocks based on their recent corporate events (Last 7 Days).\n"
            f"Your decisions have financial consequences: accurate bets increase your capital, while incorrect bets reduce it.\n"
            f"Your universe includes these stocks AND a 'CASH' asset (which has a known daily risk-free rate of {risk_free_rate:.6f}).\n"
            f"Allocate your capital based on your conviction in the signal strength. Treat CASH as a peer asset.\n\n"
            f"{full_context}\n\n"
            f"Constraints:\n"
            f"1. MATHEMATICAL RULE: The Net Exposure MUST exactly equal 100.0.\n"
            f"   Calculation: (Sum of ALL 'long' amounts) - (Sum of ALL 'short' amounts) = 100.0\n"
            f"2. IMPORTANT: Every single 'amount' MUST be a strictly POSITIVE number (e.g., 50.0, never -50.0). The 'direction' field ('long' or 'short') handles the math sign.\n"
            f"3. GROSS EXPOSURE LIMIT: The sum of ALL amounts (long + short) MUST NOT exceed 1000.0.\n"
            f"4. MATH EXAMPLES:\n"
            f"   - Leverage: Long Stocks $150, Short CASH $50. Math: 150 - 50 = 100.0.\n"
            f"   - Hedging: Long Stocks $120, Short Stocks $20, Long CASH $0. Math: 120 - 20 = 100.0.\n"
            f"   - All Cash: Long CASH $100. Math: 100 - 0 = 100.0.\n"
            f"5. For stocks: 'long' = Bullish, 'short' = Bearish.\n"
            f"6. For CASH: 'long' = Lending/Holding cash to earn the risk-free rate, 'short' = Borrowing cash to deploy leverage.\n"
            f"7. CONVICTION ALLOCATION: Allocate capital strictly proportionally to your conviction. If there is no clear basis for a judgment on an asset, the allocated amount should be 0.\n"
            f"8. EMPTY PORTFOLIO RULE: If you choose not to hold any stocks, you MUST explicitly output a single allocation: 'long' CASH for 100.0. Do NOT output an empty list.\n"
            f"9. NO SPLIT CASH: Do not split CASH into multiple allocations. Provide only ONE aggregated row for CASH (either 'long' or 'short')."
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
            
    result = {
        "allocations": [a.model_dump() for a in decision.allocations],
        "metrics": {
            "original_net_exposure": original_net,
            "original_gross_exposure": original_gross
        }
    }
    
    message = HumanMessage(
        content=json.dumps(result),
        name=agent_id,
    )
    
    # Optional: Display output
    if state.get("metadata", {}).get("show_reasoning"):
        show_agent_reasoning(result, "News Sentiment Allocator")

    # Store in a new key "allocator_decisions" to avoid overwriting standard signals
    if "allocator_decisions" not in state["data"]:
        state["data"]["allocator_decisions"] = {}
    state["data"]["allocator_decisions"][agent_id] = result

    progress.update_status(agent_id, None, "Done")
    
    return {
        "messages": [message],
        "data": state["data"]
    }
