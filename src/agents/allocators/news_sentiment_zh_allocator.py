
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
from typing import Literal, List, Dict, Any
from datetime import datetime, timedelta

# Re-use the Sentiment schema from the original agent logic
class Sentiment(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = "neutral"
    confidence: int = Field(description="Confidence 0-100")

def news_sentiment_zh_allocator(state: Dict[str, Any], agent_id: str = "news_sentiment_zh_allocator"):
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
        f"1. 数学铁律：净敞口（Net Exposure）必须精确等于 100.0。\n"
        f"   计算公式：（所有 'long' 方向的金额总和） - （所有 'short' 方向的金额总和） = 100.0\n"
        f"2. 重要限制：每一个 'amount'（金额）必须是严格的正数（例如 50.0，绝不能是 -50.0）。数学正负号由 'direction'（方向）字段（'long' 或 'short'）来控制。\n"
        f"3. 总敞口上限：为了防止过度承担风险，所有金额的绝对值总和（做多 + 做空的总规模）不得超过 1000.0。\n"
        f"4. 数学示例：\n"
        f"   - 加杠杆：做多股票 150，做空现金 (CASH) 50。数学计算：150 - 50 = 100.0。\n"
        f"   - 做对冲：做多股票 120，做空股票 20，做多现金 0。数学计算：120 - 20 = 100.0。\n"
        f"   - 纯防守：做多现金 (CASH) 100。数学计算：100 - 0 = 100.0。\n"
        f"5. 对于股票：'long' = 看涨（利好），'short' = 看跌（利空）。\n"
        f"6. 对于现金 (CASH)：'long' = 借出/持有现金以赚取无风险利率，'short' = 借入现金以便给投资组合加杠杆。\n"
        f"7. 宁缺毋滥：如果你的确信度很低，请不要为该资产分配任何资金。决策务必果断。\n"
        f"8. 空仓防守规则：如果你认为市场风险过高，不想持有任何股票，你必须明确输出唯一的一笔资产分配：'long' CASH 100.0。绝对不要输出空列表。\n"
        f"9. 禁止拆分现金：不要将 CASH 拆分成多笔分配记录。只能提供唯一一行汇总的 CASH 分配（要么只 'long'，要么只 'short'）。\n"
        f"10. 语言要求：你可以直接阅读和分析提供的英文数据，但请务必使用中文（Chinese）来撰写你的 'reasoning'（推理逻辑）字段。"
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
                hint_str = f"  - 量化信号: {action.upper()} (理由: {task.get('reason', 'N/A')})。注意: 此信号仅供参考。你必须基于你的特定策略做出独立判断。\\n"

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
            f"你是一位拥有 $100 本金的博弈交易员。请根据以下股票近期（过去7天内）的企业事件对它们进行分析。\\n"
            f"你的投资池包括这些股票，以及一个 'CASH'（现金）资产（其已知的每日无风险收益率为 {risk_free_rate:.6f}）。\\n"
            f"你必须在这些资产之间分配资金，以最大化你的博弈回报。请将现金（CASH）视为同等级资产。\\n\\n"
            f"投资池上下文:\\n{full_context}\\n\\n"
            f"约束条件:\\n"
            f"1. 数学铁律：净敞口（Net Exposure）必须精确等于 100.0。\\n"
            f"   计算公式：（所有 'long' 方向的金额总和） - （所有 'short' 方向的金额总和） = 100.0\\n"
            f"2. 重要限制：每一个 'amount'（金额）必须是严格的正数（例如 50.0，绝不能是 -50.0）。数学正负号由 'direction'（方向）字段（'long' 或 'short'）来控制。\\n"
            f"3. 总敞口上限：为了防止过度承担风险，所有金额的绝对值总和（做多 + 做空的总规模）不得超过 1000.0。\\n"
            f"4. 数学示例：\\n"
            f"   - 加杠杆：做多股票 150，做空现金 (CASH) 50。数学计算：150 - 50 = 100.0。\\n"
            f"   - 做对冲：做多股票 120，做空股票 20，做多现金 0。数学计算：120 - 20 = 100.0。\\n"
            f"   - 纯防守：做多现金 (CASH) 100。数学计算：100 - 0 = 100.0。\\n"
            f"5. 对于股票：'long' = 看涨（利好），'short' = 看跌（利空）。\\n"
            f"6. 对于现金 (CASH)：'long' = 借出/持有现金以赚取无风险利率，'short' = 借入现金以便给投资组合加杠杆。\\n"
            f"7. 宁缺毋滥：如果你的确信度很低，请不要为该资产分配任何资金。决策务必果断。\\n"
            f"8. 空仓防守规则：如果你认为市场风险过高，不想持有任何股票，你必须明确输出唯一的一笔资产分配：'long' CASH 100.0。绝对不要输出空列表。\\n"
            f"9. 禁止拆分现金：不要将 CASH 拆分成多笔分配记录。只能提供唯一一行汇总的 CASH 分配（要么只 'long'，要么只 'short'）。\\n"
            f"10. 语言要求：你可以直接阅读和分析提供的英文数据，但请务必使用中文（Chinese）来撰写你的 'reasoning'（推理逻辑）字段。"
        )
    else:
        # Variant Group: Wealth Consequence Prompt (Default)
        prompt = (
            f"你是一位拥有 $100 本金的博弈交易员。你的目标是通过准确的预测来最大化你的个人财富。\\n"
            f"请根据以下股票近期（过去7天内）的企业事件对它们进行分析。\\n"
            f"你的决策将产生真实的财务后果：准确的押注会增加你的本金，而错误的押注会减少本金。\\n"
            f"你的投资池包括这些股票，以及一个 'CASH'（现金）资产（其已知的每日无风险收益率为 {risk_free_rate:.6f}）。\\n"
            f"请根据你对信号强度的确信度来分配资金。请将现金（CASH）视为同等级资产。\\n\\n"
            f"投资池上下文:\\n{full_context}\\n\\n"
            f"约束条件:\\n"
            f"1. 数学铁律：净敞口（Net Exposure）必须精确等于 100.0。\\n"
            f"   计算公式：（所有 'long' 方向的金额总和） - （所有 'short' 方向的金额总和） = 100.0\\n"
            f"2. 重要限制：每一个 'amount'（金额）必须是严格的正数（例如 50.0，绝不能是 -50.0）。数学正负号由 'direction'（方向）字段（'long' 或 'short'）来控制。\\n"
            f"3. 总敞口上限：为了防止过度承担风险，所有金额的绝对值总和（做多 + 做空的总规模）不得超过 1000.0。\\n"
            f"4. 数学示例：\\n"
            f"   - 加杠杆：做多股票 150，做空现金 (CASH) 50。数学计算：150 - 50 = 100.0。\\n"
            f"   - 做对冲：做多股票 120，做空股票 20，做多现金 0。数学计算：120 - 20 = 100.0。\\n"
            f"   - 纯防守：做多现金 (CASH) 100。数学计算：100 - 0 = 100.0。\\n"
            f"5. 对于股票：'long' = 看涨（利好），'short' = 看跌（利空）。\\n"
            f"6. 对于现金 (CASH)：'long' = 借出/持有现金以赚取无风险利率，'short' = 借入现金以便给投资组合加杠杆。\\n"
            f"7. 宁缺毋滥：如果你的确信度很低，请不要为该资产分配任何资金。决策务必果断。\\n"
            f"8. 空仓防守规则：如果你认为市场风险过高，不想持有任何股票，你必须明确输出唯一的一笔资产分配：'long' CASH 100.0。绝对不要输出空列表。\\n"
            f"9. 禁止拆分现金：不要将 CASH 拆分成多笔分配记录。只能提供唯一一行汇总的 CASH 分配（要么只 'long'，要么只 'short'）。\\n"
            f"10. 语言要求：你可以直接阅读和分析提供的英文数据，但请务必使用中文（Chinese）来撰写你的 'reasoning'（推理逻辑）字段。"
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
