
from langchain_core.messages import HumanMessage
from src.schemas import PortfolioDecision, Allocation
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_company_news
from src.utils.api_key import get_api_key_from_state
from src.utils.llm import call_llm
from src.utils.progress import progress
import json
import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal, List
from datetime import datetime, timedelta

# Re-use the Sentiment schema from the original agent logic
class Sentiment(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = "neutral"
    confidence: int = Field(description="Confidence 0-100")

def news_sentiment_allocator(state: AgentState, agent_id: str = "news_sentiment_allocator"):
    """
    Batch Agent: Analyzes news for a UNIVERSE of tickers and allocates $100 capital.
    
    Step 1: Analyze events for each ticker individually (preserving original logic).
    Step 2: Make a relative value judgment to allocate capital.
    """
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")
    risk_free_rate = data.get("risk_free_rate", 0.0)
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    
    # Store intermediate analysis to feed into the Allocator
    ticker_summaries = {} 

    # --- PHASE 1: INDIVIDUAL ANALYSIS (Dual Mode) ---
    # mode="individual": 5 calls per ticker (Original)
    # mode="batch": 1 call per ticker (Optimized)
    
    # Check for mode override in state or default to individual for now (unless user wants batch)
    analysis_mode = state.get("metadata", {}).get("allocator_mode", "individual")  # Default to "individual" for safety/preservation
    
    if analysis_mode == "global_batch":
        # Global Batch: One shot analysis + allocation
        result = _analyze_and_allocate_global(tickers, end_date, api_key, agent_id, state)
        
        # Helper to format output same as others
        message = HumanMessage(content=json.dumps(result), name=agent_id)
        if state.get("metadata", {}).get("show_reasoning"):
            show_agent_reasoning(result, "News Sentiment Allocator (Global)")
        if "allocator_decisions" not in state["data"]:
            state["data"]["allocator_decisions"] = {}
        state["data"]["allocator_decisions"][agent_id] = result
        progress.update_status(agent_id, None, "Done")
        return {"messages": [message], "data": state["data"]}
        
    elif analysis_mode == "batch":
        ticker_summaries = _analyze_events_batch(tickers, end_date, api_key, agent_id, state)
    else:
        ticker_summaries = _analyze_events_individual(tickers, end_date, api_key, agent_id, state)

    # --- PHASE 2: GLOBAL ALLOCATION (The New Logic) ---
    progress.update_status(agent_id, "ALL", "Calculating Portfolio Allocation")
    
    # Construct the Universe View
    annual_rf = risk_free_rate * 252
    cash_summary = (
        f"Signal: neutral (100%). CASH asset with Guaranteed Daily Risk-Free Rate: {risk_free_rate:.6f} "
        f"(Annualized: {annual_rf:.2%}). 'long' means earning this rate. 'short' means borrowing."
    )
    ticker_summaries["CASH"] = cash_summary
    
    universe_context = "\n".join([f"Stock {t}: {s}" for t, s in ticker_summaries.items()])
    
    allocation_prompt = (
        f"You are a Betting Agent with $100 capital. Your objective is to maximize your wealth through accurate predictions.\n"
        f"Analyze the following universe of stocks based on their recent corporate events (Last 7 Days).\n"
        f"Your decisions have financial consequences: accurate bets increase your capital, while incorrect bets reduce it.\n"
        f"Your universe includes these stocks AND a 'CASH' asset (which has a known daily risk-free rate of {risk_free_rate:.6f}).\n"
        f"Allocate your capital based on your conviction in the signal strength. Treat CASH as a peer asset.\n\n"
        f"{universe_context}\n\n"
        f"Constraints:\n"
        f"1. MATHEMATICAL RULE: The Net Exposure MUST exactly equal 100.0.\n"
        f"   Calculation: (Sum of ALL 'long' amounts) - (Sum of ALL 'short' amounts) = 100.0\n"
        f"2. IMPORTANT: Every single 'amount' MUST be a strictly POSITIVE number (e.g., 50.0, never -50.0). The 'direction' field ('long' or 'short') handles the math sign.\n"
        f"3. GROSS EXPOSURE LIMIT: To prevent excessive risk, the sum of ALL amounts (long + short) should not exceed 1000.0.\n"
        f"4. MATH EXAMPLES:\n"
        f"   - Leverage: Long Stocks $150, Short CASH $50. Math: 150 - 50 = 100.0.\n"
        f"   - Hedging: Long Stocks $120, Short Stocks $20, Long CASH $0. Math: 120 - 20 = 100.0.\n"
        f"   - Pure Cash: Long CASH $100. Math: 100 - 0 = 100.0.\n"
        f"5. For stocks: 'long' = Bullish, 'short' = Bearish.\n"
        f"6. For CASH: 'long' = Lending/Holding cash to earn the risk-free rate, 'short' = Borrowing cash to deploy leverage.\n"
        f"7. Do NOT allocate to an asset if your conviction is low. Be decisive.\n"
    )
    
    # Call LLM for the final decision
    decision = call_llm(allocation_prompt, PortfolioDecision, agent_name=agent_id, state=state)
    
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

def _analyze_events_individual(tickers, end_date, api_key, agent_id, state):
    """Original Logic: Analyze each event separately."""
    ticker_summaries = {}
    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching KeyDev events (Individual)")
        print(f"  [NewsAllocator] Analyzing {ticker} for {end_date} (Individual)...") 
        company_news = get_company_news(ticker, end_date, limit=10, api_key=api_key)
        
        if not company_news:
            ticker_summaries[ticker] = "No recent corporate events found."
            continue

        valid_events = []
        for news in company_news[:5]: # Analyze up to 5 events
            prompt = (
                f"You are a hedge fund analyst specializing in Event-Driven strategies. "
                f"Analyze the following Corporate Event for stock {ticker}.\n\n"
                f"Event Context: {news.source} (KeyDev Event Type)\n"
                f"Headline: {news.title}\n\n"
                f"Determine if this event is FUNDAMENTALLY 'positive' (bullish), 'negative' (bearish), "
                f"or 'neutral' for the stock price.\n"
                f"Also provide a confidence score (0-100).\n"
            )
            try:
                response = call_llm(prompt, Sentiment, agent_name=agent_id, state=state)
                valid_events.append({
                    "headline": news.title,
                    "type": news.source,
                    "sentiment": response.sentiment,
                    "confidence": response.confidence
                })
            except Exception as e:
                print(f"Error analyzing event for {ticker}: {e}")
        
        if valid_events:
            bullish_count = sum(1 for e in valid_events if e["sentiment"] == "positive")
            # Construct summary string
            summary = (
                f"Analyzed {len(valid_events)} events. "
                f"Bullish count: {bullish_count}. "
                f"Top Event: {valid_events[0]['headline']} ({valid_events[0]['sentiment']})"
            )
            ticker_summaries[ticker] = summary
        else:
            ticker_summaries[ticker] = "Events found but analysis failed or returned neutral."
    return ticker_summaries

def _analyze_events_batch(tickers, end_date, api_key, agent_id, state):
    """Optimized Logic: 1 LLM Call per Ticker."""
    ticker_summaries = {}
    
    # Calculate start date (7 days lookback)
    dt_end = datetime.strptime(end_date, "%Y-%m-%d")
    dt_start = dt_end - timedelta(days=7)
    start_date = dt_start.strftime("%Y-%m-%d")

    # Define a Batch Summary Schema locally
    class BatchSentiment(BaseModel):
        signal: Literal["bullish", "bearish", "neutral"] = Field(default="neutral", description="Overall signal")
        confidence: int = Field(default=0, description="Confidence score 0-100")
        summary: str = Field(default="Analysis failed or no events.", description="Synthesized summary of all events")

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching KeyDev events (Batch)")
        print(f"  [NewsAllocator] Analyzing {ticker} for {end_date} (Batch, 7-day lookback)...")
        # Pass start_date to filter strict window
        company_news = get_company_news(ticker, end_date, start_date=start_date, limit=10, api_key=api_key)
        
        events_text = ""
        if not company_news:
            events_text = "No significant corporate events found in the last 7 days."
        else:
            # Format all events into one block
            events_text = "\n".join([f"- {n.date}: {n.title} (Source: {n.source})" for n in company_news[:5]])
        
        prompt = (
            f"You are an Event-Driven Analyst. Review recent events for {ticker} (Last 7 Days) and provide a signal.\n\n"
            f"{events_text}\n\n"
            f"Is the aggregate impact Bullish, Bearish, or Neutral?\n"
            f"If 'No events found', decide if silence is neutral or meaningful (usually neutral).\n"
            f"Provide a brief summary reasoning."
        )
        
        try:
            response = call_llm(prompt, BatchSentiment, agent_id, state)
            ticker_summaries[ticker] = f"Signal: {response.signal} ({response.confidence}%). {response.summary}"
        except Exception as e:
            print(f"Error in batch analysis for {ticker}: {e}")
            ticker_summaries[ticker] = "Error in batch analysis."
            
    return ticker_summaries

def _analyze_and_allocate_global(tickers, end_date, api_key, agent_id, state):
    """Global Batch Logic: Fetch all data, then 1 LLM Call for Decision."""
    progress.update_status(agent_id, "ALL", "Fetching Universe Data (Global Batch)")
    
    # Calculate start date (7 days lookback)
    dt_end = datetime.strptime(end_date, "%Y-%m-%d")
    dt_start = dt_end - timedelta(days=7)
    start_date = dt_start.strftime("%Y-%m-%d")

    data = state.get("data", {})
    risk_free_rate = data.get("risk_free_rate", 0.0)
    
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
                hint_str = f"Quantitative Signal: {action.upper()} (Reason: {task.get('reason', 'N/A')})\\n"

        # Pass start_date
        company_news = get_company_news(ticker, end_date, start_date=start_date, limit=5, api_key=api_key)
        if not company_news:
            universe_content.append(f"Stock {ticker}: {hint_str}No significant corporate events in the last 7 days.")
            continue
            
        events_text = "\\n".join([f"- {n.date}: {n.title}" for n in company_news])
        universe_content.append(f"Stock {ticker} Events (Last 7 Days):\\n{hint_str}{events_text}")

    annual_rf = risk_free_rate * 252
    cash_summary = (
        f"Stock CASH Events (Last 7 Days):\\n"
        f"- Guaranteed Daily Risk-Free Rate: {risk_free_rate:.6f} (Annualized: {annual_rf:.2%}).\\n"
        f"- Note: 'long' means earning this rate. 'short' means paying this rate to borrow capital."
    )
    universe_content.append(cash_summary)

    full_context = "\\n\\n".join(universe_content)
    
    # Select Prompt based on metadata (A/B Testing)
    prompt_version = state.get("metadata", {}).get("prompt_version", "wealth")
    
    if prompt_version == "standard":
        # Control Group: Original Prompt
        prompt = (
            f"You are a Betting Agent with $100 capital. "
            f"Analyze the following universe of stocks based on their recent corporate events (Last 7 Days).\\n"
            f"Your universe includes these stocks AND a 'CASH' asset (which has a known daily risk-free rate of {risk_free_rate:.6f}).\\n"
            f"You must allocate capital across these assets to maximize your betting return. Treat CASH as a peer asset.\\n\\n"
            f"{full_context}\\n\\n"
            f"Constraints:\\n"
            f"1. You MUST allocate across the assets. The Net Exposure (Sum of 'long' amounts MINUS Sum of 'short' amounts) MUST exactly equal 100.0.\\n"
            f"   - Leverage Example: Long $150 in Stocks, Short $50 in CASH. Net = 150 - 50 = 100.\\n"
            f"   - Shorting Example: Short $50 in Stocks, Long $150 in CASH. Net = 150 - 50 = 100.\\n"
            f"2. Every allocated asset (including CASH) MUST have a direction: 'long' or 'short'.\\n"
            f"3. For stocks: 'long' = Bullish, 'short' = Bearish.\\n"
            f"4. For CASH: 'long' = Lending/Holding cash to earn the risk-free rate, 'short' = Borrowing cash to deploy leverage.\\n"
            f"5. Do NOT allocate to an asset if your conviction is low. Be decisive.\\n"
        )
    else:
        # Variant Group: Wealth Consequence Prompt (Default)
        prompt = (
            f"You are a Betting Agent with $100 capital. Your objective is to maximize your wealth through accurate predictions.\\n"
            f"Analyze the following universe of stocks based on their recent corporate events (Last 7 Days).\\n"
            f"Your decisions have financial consequences: accurate bets increase your capital, while incorrect bets reduce it.\\n"
            f"Your universe includes these stocks AND a 'CASH' asset (which has a known daily risk-free rate of {risk_free_rate:.6f}).\\n"
            f"Allocate your capital based on your conviction in the signal strength. Treat CASH as a peer asset.\\n\\n"
            f"{full_context}\\n\\n"
            f"Constraints:\\n"
            f"1. You MUST allocate across the assets. The Net Exposure (Sum of 'long' amounts MINUS Sum of 'short' amounts) MUST exactly equal 100.0.\\n"
            f"   - Leverage Example: Long $150 in Stocks, Short $50 in CASH. Net = 150 - 50 = 100.\\n"
            f"   - Shorting Example: Short $50 in Stocks, Long $150 in CASH. Net = 150 - 50 = 100.\\n"
            f"2. Every allocated asset (including CASH) MUST have a direction: 'long' or 'short'.\\n"
            f"3. For stocks: 'long' = Bullish, 'short' = Bearish.\\n"
            f"4. For CASH: 'long' = Lending/Holding cash to earn the risk-free rate, 'short' = Borrowing cash to deploy leverage.\\n"
            f"5. Do NOT allocate to an asset if your conviction is low. Be decisive.\\n"
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
            
    return {
        "allocations": [a.model_dump() for a in decision.allocations],
        "metrics": {
            "original_net_exposure": original_net,
            "original_gross_exposure": original_gross
        }
    }
