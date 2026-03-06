from langchain_core.messages import HumanMessage
from src.schemas import PortfolioDecision, Allocation
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_prices, prices_to_df
from src.utils.api_key import get_api_key_from_state
from typing import Dict, Any
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

def technical_zh_allocator(state: AgentState, agent_id: str = "technical_zh_allocator"):
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
                hint_str = f"  - 量化信号: {action.upper()} (理由: {task.get('reason', 'N/A')})。注意: 此信号仅供参考。你必须基于你的特定策略做出独立判断。\n"
        
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
        f"资产 CASH (现金):\n"
        f"  - 当前价格: $1.00\n"
        f"  - 已知每日无风险利率: {risk_free_rate:.6f} (年化: {annual_rf:.2%})\n"
        f"  - 注意: 'long' 表示以此无风险利率赚取收益。'short' 表示以此利率借入资金。"
    )
    universe_summaries.append(cash_summary)
    study_notes = "\n\n".join(universe_summaries)
    
    # Select Prompt based on A/B Config
    prompt_version = state.get("metadata", {}).get("prompt_version", "wealth")
    
    if prompt_version == "standard":
        base_instruction = (
            "你是一位技术面交易员。你的目标是通过技术分析捕捉市场动能和价格趋势来最大化投资组合的收益。\n"
            f"你的投资池包括所提供的股票'Stock'，以及一个 'CASH'（现金）资产（其已知的每日无风险收益率为 {risk_free_rate:.6f}）。请将现金（CASH）视为具有保证收益的与股票（Stock）同等级的资产。\n"
            "请独立评估每项资产的技术面形态与动能以决定投资方向，并在整个投资池中进行综合权衡，根据相对的收益空间与确信度，将 $100 的本金分配到这些资产中。"
        )
    else:
        base_instruction = (
            "你是一位技术面交易员。你的目标是通过技术分析捕捉市场动能和价格趋势来最大化你的个人财富。\n"
            "你现在拥有$100的真实本金，你的决策将产生真实的财务后果。"
            f"你的投资池包括所提供的股票'Stock'，以及一个 'CASH'（现金）资产（其已知的每日无风险收益率为 {risk_free_rate:.6f}）。请将现金（CASH）视为具有保证收益的与股票（Stock）同等级的资产。\n"
            "请独立评估每项资产的技术面形态与动能以决定投资方向，并在整个投资池中进行综合权衡，根据相对的收益空间与确信度，将 $100 的本金分配到这些资产中。"
        )

    prompt = (
        f"{base_instruction}\n\n"
        f"技术分析数据:\n{study_notes}\n\n"
        f"约束条件:\n"
        f"1. 数学铁律：净敞口（Net Exposure）必须精确等于 100.0。\n"
        f"   计算公式：（所有 'long' 方向的金额总和） - （所有 'short' 方向的金额总和） = 100.0\n"
        f"2. 重要限制：每一个 'amount'（金额）必须是严格的正数（例如 50.0，绝不能是 -50.0）。数学正负号由 'direction'（方向）字段（'long' 或 'short'）来控制。\n"
        f"3. 总敞口上限：所有分配金额的绝对值总和（做多规模 + 做空规模）严格不得超过 1000.0。\n"
        f"4. 数学示例：\n"
        f"   - 加杠杆：做多股票 150，做空现金 (CASH) 50。数学计算：150 - 50 = 100.0。\n"
        f"   - 做对冲：做多股票 120，做空股票 20，做多现金 0。数学计算：120 - 20 = 100.0。\n"
        f"   - 全额现金：做多现金 (CASH) 100。数学计算：100 - 0 = 100.0。\n"
        f"5. 对于股票：'long' = 基于技术面看涨形态，买入以建立多头头寸；'short' = 基于技术面看跌形态，融券以建立空头头寸。\n"
        f"6. 对于现金 (CASH)：'long' = 持有现金以赚取无风险利率，'short' = 借入现金以便给投资组合加杠杆。\n"
        f"7. 确信度底线：如果对某资产缺乏明确的判断依据，对应的分配金额应为 0。\n"
        f"8. 空仓规则：如果你不想持有任何股票，你必须明确输出唯一的一笔资产分配：'long' CASH 100.0。绝对不要输出空列表。\n"
        f"9. 禁止拆分现金：不要将 CASH 拆分成多笔分配记录。只能提供唯一一行汇总的 CASH 分配（要么只 'long'，要么只 'short'）。\n"
        f"10. 语言要求：你可以直接阅读和分析提供的英文数据，但请务必使用中文来撰写你的 'reasoning'（推理逻辑）字段。"
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
