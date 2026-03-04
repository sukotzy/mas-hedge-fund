from langchain_core.messages import HumanMessage
from src.schemas import PortfolioDecision, Allocation
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_company_news
from src.utils.api_key import get_api_key_from_state
from src.utils.llm import call_llm
from src.utils.progress import progress
import json
from datetime import datetime, timedelta

def news_sentiment_zh_allocator(state: AgentState, agent_id: str = "news_sentiment_zh_allocator"):
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
                hint_str = f"  - 量化信号: {action.upper()} (理由: {task.get('reason', 'N/A')})。注意: 此信号仅供参考。你必须基于你的特定策略做出独立判断。\n"

        # Pass start_date
        company_news = get_company_news(ticker, end_date, start_date=start_date, limit=5, api_key=api_key)
        if not company_news:
            universe_content.append(f"Stock {ticker}: {hint_str}No significant corporate events in the last 7 days.")
            continue
            
        events_text = "\n".join([f"- {n.date}: {n.title}" for n in company_news])
        universe_content.append(f"Stock {ticker} Events (Last 7 Days):\n{hint_str}{events_text}")

    annual_rf = risk_free_rate * 252
    cash_summary = (
        f"资产 CASH 近期事件 (过去7天):\n"
        f"- 已知每日无风险利率: {risk_free_rate:.6f} (年化: {annual_rf:.2%})\n"
        f"- 注意: 'long' 表示以此无风险利率赚取收益。'short' 表示以此利率借入资金。"
    )
    universe_content.append(cash_summary)

    full_context = "\n\n".join(universe_content)
    
    # Select Prompt based on metadata (A/B Testing)
    prompt_version = state.get("metadata", {}).get("prompt_version", "wealth")
    
    if prompt_version == "standard":
        # Control Group: Original Prompt
        prompt = (
            f"你是一位拥有 $100 本金的博弈交易员。请根据以下股票近期（过去7天内）的企业事件对它们进行分析。\n"
            f"你的投资池包括这些股票，以及一个 'CASH'（现金）资产（其已知的每日无风险收益率为 {risk_free_rate:.6f}）。\n"
            f"你必须在这些资产之间分配资金，以最大化你的博弈回报。请将现金（CASH）视为同等级资产。\n\n"
            f"投资池上下文:\n{full_context}\n\n"
            f"约束条件:\n"
            f"1. 数学铁律：净敞口（Net Exposure）必须精确等于 100.0。\n"
            f"   计算公式：（所有 'long' 方向的金额总和） - （所有 'short' 方向的金额总和） = 100.0\n"
            f"2. 重要限制：每一个 'amount'（金额）必须是严格的正数（例如 50.0，绝不能是 -50.0）。数学正负号由 'direction'（方向）字段（'long' 或 'short'）来控制。\n"
            f"3. 总敞口上限：所有分配金额的绝对值总和（做多规模 + 做空规模）严格不得超过 1000.0。\n"
            f"4. 数学示例：\n"
            f"   - 加杠杆：做多股票 150，做空现金 (CASH) 50。数学计算：150 - 50 = 100.0。\n"
            f"   - 做对冲：做多股票 120，做空股票 20，做多现金 0。数学计算：120 - 20 = 100.0。\n"
            f"   - 全额现金：做多现金 (CASH) 100。数学计算：100 - 0 = 100.0。\n"
            f"5. 对于股票：'long' = 看涨（Bullish），'short' = 看跌（Bearish）。\n"
            f"6. 对于现金 (CASH)：'long' = 借出/持有现金以赚取无风险利率，'short' = 借入现金以便给投资组合加杠杆。\n"
            f"7. 确信度分配：请严格根据信号的确信度成比例分配资金。如果对某资产缺乏明确的判断依据，对应的分配金额应为 0。\n"
            f"8. 空仓规则：如果你不想持有任何股票，你必须明确输出唯一的一笔资产分配：'long' CASH 100.0。绝对不要输出空列表。\n"
            f"9. 禁止拆分现金：不要将 CASH 拆分成多笔分配记录。只能提供唯一一行汇总的 CASH 分配（要么只 'long'，要么只 'short'）。\n"
            f"10. 语言要求：你可以直接阅读和分析提供的英文数据，但请务必使用中文（Chinese）来撰写你的 'reasoning'（推理逻辑）字段。"
        )
    else:
        # Variant Group: Wealth Consequence Prompt (Default)
        prompt = (
            f"你是一位拥有 $100 本金的博弈交易员。你的目标是通过准确的预测来最大化你的个人财富。\n"
            f"请根据以下股票近期（过去7天内）的企业事件对它们进行分析。\n"
            f"你的决策将产生真实的财务后果：准确的押注会增加你的本金，而错误的押注会减少本金。\n"
            f"你的投资池包括这些股票，以及一个 'CASH'（现金）资产（其已知的每日无风险收益率为 {risk_free_rate:.6f}）。\n"
            f"请根据你对信号强度的确信度来分配资金。请将现金（CASH）视为同等级资产。\n\n"
            f"投资池上下文:\n{full_context}\n\n"
            f"约束条件:\n"
            f"1. 数学铁律：净敞口（Net Exposure）必须精确等于 100.0。\n"
            f"   计算公式：（所有 'long' 方向的金额总和） - （所有 'short' 方向的金额总和） = 100.0\n"
            f"2. 重要限制：每一个 'amount'（金额）必须是严格的正数（例如 50.0，绝不能是 -50.0）。数学正负号由 'direction'（方向）字段（'long' 或 'short'）来控制。\n"
            f"3. 总敞口上限：所有分配金额的绝对值总和（做多规模 + 做空规模）严格不得超过 1000.0。\n"
            f"4. 数学示例：\n"
            f"   - 加杠杆：做多股票 150，做空现金 (CASH) 50。数学计算：150 - 50 = 100.0。\n"
            f"   - 做对冲：做多股票 120，做空股票 20，做多现金 0。数学计算：120 - 20 = 100.0。\n"
            f"   - 全额现金：做多现金 (CASH) 100。数学计算：100 - 0 = 100.0。\n"
            f"5. 对于股票：'long' = 看涨（Bullish），'short' = 看跌（Bearish）。\n"
            f"6. 对于现金 (CASH)：'long' = 借出/持有现金以赚取无风险利率，'short' = 借入现金以便给投资组合加杠杆。\n"
            f"7. 确信度分配：请严格根据信号的确信度成比例分配资金。如果对某资产缺乏明确的判断依据，对应的分配金额应为 0。\n"
            f"8. 空仓规则：如果你不想持有任何股票，你必须明确输出唯一的一笔资产分配：'long' CASH 100.0。绝对不要输出空列表。\n"
            f"9. 禁止拆分现金：不要将 CASH 拆分成多笔分配记录。只能提供唯一一行汇总的 CASH 分配（要么只 'long'，要么只 'short'）。\n"
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
