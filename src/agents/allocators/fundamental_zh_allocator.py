
from langchain_core.messages import HumanMessage
from src.schemas import PortfolioDecision, Allocation
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_financial_metrics
from src.utils.api_key import get_api_key_from_state
from src.utils.llm import call_llm
from src.utils.progress import progress
import json
from datetime import datetime
from typing import Dict, Any

def fundamental_zh_allocator(state: AgentState, agent_id: str = "fundamental_zh_allocator"):
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
                hint_str = f"  - 量化信号: {action.upper()} (理由: {task.get('reason', 'N/A')})。注意: 此信号仅供参考。你必须基于你的特定策略做出独立判断。\n"
        
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
    
    if prompt_version == "standard":
        base_instruction = (
            "你是一位基本面投资组合经理。你的目标是通过投资高质量、估值合理且具有增长潜力的公司来最大化收益。\n"
            f"你的投资池包括所提供的股票，以及一个 'CASH'（现金）资产（其已知的每日无风险收益率为 {risk_free_rate:.6f}）。\n"
            "请根据这些资产的基本面强度，将 $100 本金分配到这些资产中。"
        )
    else:
        # Wealth / Skin-in-the-game
        base_instruction = (
            "你是一位拥有 $100 本金的基本面投资者。你的目标是最大化你的个人财富。\n"
            "你的决策将产生真实的财务后果。 "
            f"你的投资池包括所提供的股票，以及一个 'CASH'（现金）资产（其已知的每日无风险收益率为 {risk_free_rate:.6f}）。\n"
            "请根据你的确信度来分配资金。请将现金（CASH）视为具有保证收益的同等别资产。"
        )

    prompt = (
        f"{base_instruction}\n\n"
        f"投资池数据:\n{study_notes}\n\n"
        f"约束条件:\n"
        f"1. 数学铁律：净敞口（Net Exposure）必须精确等于 100.0。\n"
        f"   计算公式：（所有 'long' 方向的金额总和） - （所有 'short' 方向的金额总和） = 100.0\n"
        f"2. 重要限制：每一个 'amount'（金额）必须是严格的正数（例如 50.0，绝不能是 -50.0）。数学正负号由 'direction'（方向）字段（'long' 或 'short'）来控制。\n"
        f"3. 总敞口上限：为了防止过度承担风险，所有金额的绝对值总和（做多 + 做空的总规模）不得超过 1000.0。\n"
        f"4. 数学示例：\n"
        f"   - 加杠杆：做多股票 150，做空现金 (CASH) 50。数学计算：150 - 50 = 100.0。\n"
        f"   - 做对冲：做多股票 120，做空股票 20，做多现金 0。数学计算：120 - 20 = 100.0。\n"
        f"   - 纯防守：做多现金 (CASH) 100。数学计算：100 - 0 = 100.0。\n"
        f"5. 对于股票：'long' = 买入（基本面良好），'short' = 卖出/做空（基本面差或被高估）。\n"
        f"6. 对于现金 (CASH)：'long' = 借出/持有现金以赚取无风险利率，'short' = 借入现金以便给投资组合加杠杆。\n"
        f"7. 宁缺毋滥：如果你的确信度很低，请不要为该资产分配任何资金。决策务必果断。\n"
        f"8. 空仓防守规则：如果你认为市场风险过高，不想持有任何股票，你必须明确输出唯一的一笔资产分配：'long' CASH 100.0。绝对不要输出空列表。\n"
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
