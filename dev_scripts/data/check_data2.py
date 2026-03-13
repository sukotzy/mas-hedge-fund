import os
os.environ['USE_LOCAL_DATA'] = 'true'

from src.agents.allocators.valuation_allocator import valuation_allocator

state = {
    "data": {
        "end_date": "2020-01-02",
        "tickers": ["MA", "AMD"],
        "risk_free_rate": 0.05
    }
}

try:
    result = valuation_allocator(state)
    print("Agent executed.")
    for msg in result["messages"]:
        print(msg.content)
except Exception as e:
    import traceback
    traceback.print_exc()
