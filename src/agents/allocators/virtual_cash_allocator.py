from src.graph.state import AgentState

def virtual_cash_allocator(state: AgentState, agent_id: str = "virtual_cash_allocator"):
    """
    Dummy allocator that purely implements a risk-off mechanism.
    Bypasses LLM entirely and defaults to allocating 100% precision into CASH.
    """
    data = state["data"]
    
    # Hardcoded risk-off portfolio
    decision = {
        "allocations": [
            {
                "ticker": "CASH",
                "direction": "long",
                "amount": 100.0
            }
        ],
        "reasoning": "Virtual Cash risk-off safe haven."
    }
    
    # Store decisions
    if "allocator_decisions" not in data:
        data["allocator_decisions"] = {}
        
    data["allocator_decisions"][agent_id] = decision
    return {"data": data}
