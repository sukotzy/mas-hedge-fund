from langchain_core.messages import HumanMessage
from src.schemas import PortfolioDecision, Allocation
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_financial_metrics, search_line_items
from src.utils.api_key import get_api_key_from_state
from typing import List, Dict, Any
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

def valuation_zh_allocator(state: AgentState, agent_id: str = "valuation_zh_allocator"):
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
                hint_str = f"  - 量化信号: {action.upper()} (理由: {task.get('reason', 'N/A')})。注意: 此信号仅供参考。你必须基于你的特定策略做出独立判断。\n"
                
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
        f"资产 CASH (现金):\n"
        f"  - 当前价格: $1.00\n"
        f"  - 已知每日无风险利率: {risk_free_rate:.6f} (年化: {annual_rf:.2%})\n"
        f"  - 注意: 'long' 表示以此无风险利率赚取收益。'short' 表示以此利率借入资金。"
    )
    universe_summaries.append(cash_summary)
    study_notes = "\n\n".join(universe_summaries)
    
    # A/B Prompts
    prompt_version = state.get("metadata", {}).get("prompt_version", "wealth")
    
    if prompt_version == "standard":
        base_instruction = (
            "你是一位价值投资投资组合经理。你的目标是将资金分配给最被低估的资产，以最大化收益。\n"
            f"你的投资池包括所提供的股票，以及一个 'CASH'（现金）资产（其已知的每日无风险收益率为 {risk_free_rate:.6f}）。\n"
            "请根据这些资产的估值情况，将 $100 本金分配到这些资产中。"
        )
    else:
        base_instruction = (
            "你是一位拥有 $100 本金的价值投资者（巴菲特/格雷厄姆风格）。你的目标是实现长期的个人财富最大化。\n"
            "你的决策将产生真实的财务后果。 "
            f"你的投资池包括所提供的股票，以及一个 'CASH'（现金）资产（其已知的每日无风险收益率为 {risk_free_rate:.6f}）。\n"
            "请根据你的确信度来分配资金。请将现金（CASH）视为具有保证收益的同等级资产。"
        )

    prompt = (
        f"{base_instruction}\n\n"
        f"估值数据:\n{study_notes}\n\n"
        f"约束条件:\n"
        f"1. 数学铁律：净敞口（Net Exposure）必须精确等于 100.0。\n"
        f"   计算公式：（所有 'long' 方向的金额总和） - （所有 'short' 方向的金额总和） = 100.0\n"
        f"2. 重要限制：每一个 'amount'（金额）必须是严格的正数（例如 50.0，绝不能是 -50.0）。数学正负号由 'direction'（方向）字段（'long' 或 'short'）来控制。\n"
        f"3. 总敞口上限：所有分配金额的绝对值总和（做多规模 + 做空规模）严格不得超过 1000.0。\n"
        f"4. 数学示例：\n"
        f"   - 加杠杆：做多股票 150，做空现金 (CASH) 50。数学计算：150 - 50 = 100.0。\n"
        f"   - 做对冲：做多股票 120，做空股票 20，做多现金 0。数学计算：120 - 20 = 100.0。\n"
        f"   - 全额现金：做多现金 (CASH) 100。数学计算：100 - 0 = 100.0。\n"
        f"5. 对于股票：'long' = 被低估（Undervalued），'short' = 被高估（Overvalued）。\n"
        f"6. 对于现金 (CASH)：'long' = 借出/持有现金以赚取无风险利率，'short' = 借入现金以便给投资组合加杠杆。\n"
        f"7. 确信度分配：请严格根据信号的确信度成比例分配资金。如果对某资产缺乏明确的判断依据，对应的分配金额应为 0。\n"
        f"8. 空仓规则：如果你不想持有任何股票，你必须明确输出唯一的一笔资产分配：'long' CASH 100.0。绝对不要输出空列表。\n"
        f"9. 禁止拆分现金：不要将 CASH 拆分成多笔分配记录。只能提供唯一一行汇总的 CASH 分配（要么只 'long'，要么只 'short'）。\n"
        f"10. 语言要求：你可以直接阅读和分析提供的英文数据，但请务必使用中文（Chinese）来撰写你的 'reasoning'（推理逻辑）字段。"
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
