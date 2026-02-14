'''Benchmarks "Separate Batch" vs "Individual Event Analysis".'''

import os
import sys
import json
import time
from rich.console import Console
from rich.table import Table

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.graph.state import AgentState
from src.agents.allocators.news_sentiment_allocator import news_sentiment_allocator

console = Console()

def run_comparison():
    # Setup Test Data
    tickers = ["AAPL"] # Reduced to 1 ticker for speed
    end_date = "2024-01-05"
    
    console.print(f"\n[bold blue]Running Allocator Comparison Test[/bold blue]")
    console.print(f"Tickers: {tickers}")
    console.print(f"Date: {end_date}\n")
    
    # --- Mode 1: Individual Event Analysis (Original) ---
    console.print("[bold yellow]--- Mode 1: Individual Event Analysis (Original) ---[/bold yellow]")
    
    # Load env vars for API keys
    from dotenv import load_dotenv
    load_dotenv()
    
    # Auto-detect Model Config (from run_analyst_history.py)
    model_name = "gpt-4o"
    model_provider = "OpenAI"
    if os.getenv("DASHSCOPE_API_KEY"):
        model_name = "qwen-max"
        model_provider = "Dashscope"
    elif os.getenv("GOOGLE_API_KEY"):
        model_name = "gemini-1.5-pro"
        model_provider = "Google"
    
    console.print(f"[bold cyan]Using Model: {model_name} ({model_provider})[/bold cyan]")
    
    # --- Control Flags ---
    RUN_INDIVIDUAL = True
    
    # --- Mode 1: Individual Event Analysis (Original) ---
    console.print("[bold yellow]--- Mode 1: Individual Event Analysis (Original) ---[/bold yellow]")
    
    # Load env vars for API keys
    from dotenv import load_dotenv
    load_dotenv()
    
    allocations_ind = []
    duration_ind = 0
    
    if RUN_INDIVIDUAL:
        # Minimal valid state for call_llm
        state_ind = {
            "data": {
                "tickers": tickers,
                "end_date": end_date,
                "start_date": "2023-01-01"
            },
            "metadata": {
                "allocator_mode": "individual",
                "show_reasoning": False,
                "model_name": model_name,
                "model_provider": model_provider
            }
        }
        
        start_time_ind = time.time()
        result_ind = news_sentiment_allocator(state_ind)
        end_time_ind = time.time()
        duration_ind = end_time_ind - start_time_ind
        
        allocations_ind = result_ind["data"]["allocator_decisions"]["news_sentiment_allocator"]["allocations"]
    else:
        console.print("[yellow]Skipping Individual Mode (Testing Batch Fix)[/yellow]")
        allocations_ind = [] # Empty for now
    
    # --- Run Batch Mode ---
    console.print("\n[bold green]--- Mode 2: Batch Analysis (Optimized) ---[/bold green]")
    state_batch = {
        "data": {
            "tickers": tickers,
            "end_date": end_date,
            "start_date": "2023-01-01"
        },
        "metadata": {
            "allocator_mode": "batch",
            "show_reasoning": False,
            "model_name": model_name,
            "model_provider": model_provider
        }
    }
    
    start_time_batch = time.time()
    result_batch = news_sentiment_allocator(state_batch)
    end_time_batch = time.time()
    duration_batch = end_time_batch - start_time_batch
    
    allocations_batch = result_batch["data"]["allocator_decisions"]["news_sentiment_allocator"]["allocations"]
    
    # --- Comparison Report ---
    console.print("\n[bold white]Comparison Results[/bold white]")
    
    table = Table(title="Performance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Individual Mode", style="yellow")
    table.add_column("Batch Mode", style="green")
    
    table.add_row("Execution Time", f"{duration_ind:.2f}s", f"{duration_batch:.2f}s")
    # Estimated calls: (5 events * 3 tickers) + 1 global vs (1 batch * 3 tickers) + 1 global
    table.add_row("Est. LLM Calls", "~16 calls", "~4 calls") 
    console.print(table)
    
    # Allocation Table
    alloc_table = Table(title="Allocation Decisions")
    alloc_table.add_column("Ticker", style="cyan")
    alloc_table.add_column("Individual Alloc %", style="yellow")
    alloc_table.add_column("Batch Alloc %", style="green")
    
    # Helper to map ticker -> amount
    def get_map(alloc_list):
        return {a['ticker']: f"{a['direction'].upper()} {a['amount']:.1f}%" for a in alloc_list}
    
    map_ind = get_map(allocations_ind)
    map_batch = get_map(allocations_batch)
    
    all_keys = set(map_ind.keys()) | set(map_batch.keys())
    for key in sorted(all_keys):
        alloc_table.add_row(key, map_ind.get(key, "N/A"), map_batch.get(key, "N/A"))
        
    console.print(alloc_table)
    
    # Save detailed reasoning for user inspection
    report = {
        "individual": allocations_ind,
        "batch": allocations_batch
    }
    with open("data/allocator_comparison.json", "w") as f:
        json.dump(report, f, indent=2)
    console.print("\nDetailed reasoning saved to [bold]data/allocator_comparison.json[/bold]")

if __name__ == "__main__":
    run_comparison()
