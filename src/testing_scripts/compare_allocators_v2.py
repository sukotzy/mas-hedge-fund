
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

def run_comparison_v2():
    # Setup Test Data - Use a small universe to test Global Batch context
    tickers = ["AAPL", "MSFT", "GOOGL"] 
    end_date = "2024-01-05"
    
    console.print(f"\n[bold blue]Running Allocator Comparison Test V2 (Global Batch)[/bold blue]")
    console.print(f"Tickers: {tickers}")
    console.print(f"Date: {end_date}\n")
    
    # Load env vars
    from dotenv import load_dotenv
    load_dotenv()
    
    # Auto-detect Model Config
    model_name = "gpt-4o"
    model_provider = "OpenAI"
    if os.getenv("DASHSCOPE_API_KEY"):
        model_name = "qwen-max"
        model_provider = "Dashscope"
    elif os.getenv("GOOGLE_API_KEY"):
        model_name = "gemini-1.5-pro"
        model_provider = "Google"
    
    console.print(f"Model: [cyan]{model_name}[/cyan] ({model_provider})")

    modes = ["batch", "global_batch"] # We skip 'individual' for speed unless requested
    results = {}
    
    for mode in modes:
        console.print(f"\n[bold yellow]--- Testing Mode: {mode} ---[/bold yellow]")
        
        state = {
            "data": {
                "tickers": tickers,
                "end_date": end_date,
                "start_date": "2023-01-01"
            },
            "metadata": {
                "allocator_mode": mode,
                "show_reasoning": False,
                "model_name": model_name,
                "model_provider": model_provider
            }
        }
        
        start_time = time.time()
        try:
            output = news_sentiment_allocator(state)
            duration = time.time() - start_time
            
            allocs = output["data"]["allocator_decisions"]["news_sentiment_allocator"]["allocations"]
            results[mode] = {
                "duration": duration,
                "allocations": allocs
            }
            console.print(f"Status: [green]Success[/green] ({duration:.2f}s)")
        except Exception as e:
            console.print(f"Status: [red]Failed[/red] - {e}")
            results[mode] = {"duration": 0, "allocations": []}

    # --- Comparison Report ---
    console.print("\n[bold white]Comparison Results[/bold white]")
    
    # Metrics
    table = Table(title="Performance Metrics")
    table.add_column("Metric", style="cyan")
    for mode in modes:
        table.add_column(f"{mode.title()} Mode", style="green")
        
    table.add_row("Execution Time", *[f"{results[m]['duration']:.2f}s" for m in modes])
    console.print(table)
    
    # Allocations
    alloc_table = Table(title="Allocation Decisions")
    alloc_table.add_column("Ticker", style="cyan")
    for mode in modes:
        alloc_table.add_column(f"{mode.title()} Alloc", style="yellow")
        
    # Get all tickers (including CASH)
    all_tickers = set()
    for m in modes:
         for a in results[m]["allocations"]:
             all_tickers.add(a["ticker"])
             
    for tick in sorted(all_tickers):
        row = [tick]
        for mode in modes:
            # Find alloc for this ticker in this mode
            found = next((a for a in results[mode]["allocations"] if a["ticker"] == tick), None)
            if found:
                row.append(f"{found['direction'].upper()} {found['amount']:.1f}%")
            else:
                row.append("-")
        alloc_table.add_row(*row)
        
    console.print(alloc_table)

if __name__ == "__main__":
    run_comparison_v2()
