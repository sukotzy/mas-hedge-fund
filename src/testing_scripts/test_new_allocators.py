
import os
import sys
import json
from rich.console import Console
from rich.panel import Panel

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.agents.allocators.fundamental_allocator import fundamental_allocator
from src.agents.allocators.technical_allocator import technical_allocator
from src.agents.allocators.valuation_allocator import valuation_allocator
from dotenv import load_dotenv

load_dotenv()
console = Console()

def test_allocators():
    tickers = ['AAPL', 'MSFT', 'TSLA', 'NVDA']
    end_date = "2023-12-15" # Recent date
    start_date = "2023-01-01"
    
    state = {
        "data": {
            "tickers": tickers,
            "end_date": end_date,
            "start_date": start_date,
            "allocator_decisions": {}
        },
        "metadata": {
            "show_reasoning": False,
            "prompt_version": "wealth",
            "model_name": "gpt-4o", # Default
            "model_provider": "OpenAI"
        }
    }
    
    # Auto-detect Model
    if os.getenv("DASHSCOPE_API_KEY"):
        state["metadata"]["model_name"] = "qwen-max"
        state["metadata"]["model_provider"] = "Dashscope"
    elif os.getenv("GOOGLE_API_KEY"):
        state["metadata"]["model_name"] = "gemini-1.5-pro"
        state["metadata"]["model_provider"] = "Google"
    
    allocators = [
        ("Fundamental", fundamental_allocator),
        ("Technical", technical_allocator),
        ("Valuation", valuation_allocator)
    ]
    
    for name, func in allocators:
        console.print(f"\n[bold blue]Running {name} Allocator...[/bold blue]")
        try:
            result = func(state)
            # parse message
            last_msg = result["messages"][-1]
            content = json.loads(last_msg.content)
            
            # Print Summary
            console.print(Panel(f"Allocations (Total: {content['metrics']['original_total']:.1f}%):", title=name))
            for a in content["allocations"]:
                direction = a["direction"].upper()
                amount = a["amount"]
                ticker = a["ticker"]
                color = "green" if direction == "UP" else "red" if direction == "DOWN" else "yellow"
                console.print(f"  {ticker}: [{color}]{direction}[/{color}] {amount:.1f}%")
                
        except Exception as e:
            console.print(f"[bold red]Error running {name}: {e}[/bold red]")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_allocators()
