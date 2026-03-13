
import os
import sys
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.agents.allocators.news_sentiment_allocator import news_sentiment_allocator
from src.data.loader import LocalDataLoader
from dotenv import load_dotenv

load_dotenv()
console = Console()

def run_ab_test():
    console.print("[bold blue]Running A/B Test: Standard vs Wealth Prompt[/bold blue]")

    # 1. Setup Data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    loader = LocalDataLoader(data_dir)
    
    # Test Case: Mixed Signals
    tickers = ['KR', 'MBC', 'INTU']
    date_str = "2023-09-19" # Confirmed mixed signal date
    
    # Print Inputs
    console.print(Panel(f"Test Date: {date_str} | Tickers: {tickers}", style="bold yellow"))
    dt_end = datetime.strptime(date_str, "%Y-%m-%d")
    dt_start = dt_end - timedelta(days=7)
    start_date_str = dt_start.strftime("%Y-%m-%d")
    
    console.print("[dim]Input Events:[/dim]")
    for t in tickers:
        news = loader.get_company_news(t, date_str, start_date=start_date_str, limit=3)
        if news:
            console.print(f"  {t}: {len(news)} events (e.g. {news[0].title[:60]}...)")
        else:
            console.print(f"  {t}: No events")

    # 2. Run Control Group (Standard Prompt)
    console.print("\n[bold magenta]--- Run 1: Control (Standard Prompt) ---[/bold magenta]")
    res_control = run_agent(tickers, date_str, "standard")

    # 3. Run Variant Group (Wealth Prompt)
    console.print("\n[bold cyan]--- Run 2: Variant (Wealth Prompt) ---[/bold cyan]")
    res_wealth = run_agent(tickers, date_str, "wealth")

    # 4. Compare Results
    console.print("\n[bold green]=== A/B Comparison ===[/bold green]")
    table = Table(show_header=True, header_style="bold white")
    table.add_column("Asset")
    table.add_column("Standard (Control)", style="magenta")
    table.add_column("Wealth (Variant)", style="cyan")
    table.add_column("Difference")
    
    assets = tickers + ["CASH"]
    
    # Extract
    def get_alloc(res, ticker):
        allocs = res["data"]["allocator_decisions"]["news_sentiment_allocator"]["allocations"]
        # Find asset (treat cash separately if needed, but schema puts it in list)
        # Actually cash is often not in 'ticker' field but 'asset' or just ticker='CASH'
        # Let's inspect
        
        # Helper to find
        found = next((a for a in allocs if a["ticker"] == ticker), None)
        if found:
            return f"{found['direction'].upper()} {found['amount']:.1f}%"
        return "N/A"

    for asset in assets:
        val_c = get_alloc(res_control, asset)
        val_v = get_alloc(res_wealth, asset)
        diff = "SAME" if val_c == val_v else "CHANGED"
        style = "dim" if diff == "SAME" else "bold yellow"
        table.add_row(asset, val_c, val_v, f"[{style}]{diff}[/{style}]")
        
    console.print(table)
    
    # Print Reasonings
    console.print("\n[bold]Reasoning Comparison (CASH):[/bold]")
    
    def get_reason(res, ticker):
        allocs = res["data"]["allocator_decisions"]["news_sentiment_allocator"]["allocations"]
        found = next((a for a in allocs if a["ticker"] == ticker), None)
        return found["reasoning"] if found else ""

    console.print(Panel(f"[magenta]Control:[/magenta] {get_reason(res_control, 'CASH')}", title="Standard Prompt"))
    console.print(Panel(f"[cyan]Variant:[/cyan] {get_reason(res_wealth, 'CASH')}", title="Wealth Prompt"))
    

def run_agent(tickers, date_str, prompt_version):
    # Auto-detect Model
    model_name = "gpt-4o"
    model_provider = "OpenAI"
    if os.getenv("DASHSCOPE_API_KEY"):
        model_name = "qwen-max"
        model_provider = "Dashscope"
    elif os.getenv("GOOGLE_API_KEY"):
        model_name = "gemini-1.5-pro"
        model_provider = "Google"

    state = {
        "data": {
            "tickers": tickers,
            "end_date": date_str,
            "start_date": "2020-01-01" 
        },
        "metadata": {
            "allocator_mode": "global_batch",
            "show_reasoning": False,
            "model_name": model_name,
            "model_provider": model_provider,
            "prompt_version": prompt_version
        }
    }
    return news_sentiment_allocator(state)

if __name__ == "__main__":
    run_ab_test()
