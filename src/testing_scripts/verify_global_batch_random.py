
import os
import sys
import random
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

def verify_random_global():
    console.print("[bold blue]Running Random Global Batch Test (3 Stocks x 3 Days)[/bold blue]")

    # 1. Load Universe
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    loader = LocalDataLoader(data_dir)
    
    # Get all unique tickers from constituents
    if loader.constituents.empty:
        console.print("[red]Error: Constituents not loaded.[/red]")
        return
        
    all_tickers = loader.constituents['ticker'].unique().tolist()
    
    # 2. Pick 3 Random Stocks
    selected_tickers = random.sample(all_tickers, 3)
    console.print(f"Selected Tickers: [cyan]{selected_tickers}[/cyan]")
    
    # 3. Pick 3 Random Dates (Avoid weekends if possible, but loader handles it)
    # Range: 2020-01-01 to 2024-01-01
    start_dt = datetime(2020, 1, 1)
    end_dt = datetime(2024, 1, 1)
    days_range = (end_dt - start_dt).days
    
    selected_dates = []
    for _ in range(3):
        rand_days = random.randint(0, days_range)
        d = start_dt + timedelta(days=rand_days)
        # Simple weekday check
        while d.weekday() > 4: # Sat/Sun
             d -= timedelta(days=1)
        selected_dates.append(d.strftime("%Y-%m-%d"))
        
    selected_dates = sorted(selected_dates)
    console.print(f"Selected Dates: [cyan]{selected_dates}[/cyan]\n")
    
    # 4. Auto-detect Model
    model_name = "gpt-4o"
    model_provider = "OpenAI"
    if os.getenv("DASHSCOPE_API_KEY"):
        model_name = "qwen-max"
        model_provider = "Dashscope"
    elif os.getenv("GOOGLE_API_KEY"):
        model_name = "gemini-1.5-pro"
        model_provider = "Google"

    # 5. Run Tests
    for date_str in selected_dates:
        console.print(Panel(f"Testing Date: {date_str}", style="bold yellow"))
        
        # Check inputs first (to see strictly what the agent sees)
        dt_end = datetime.strptime(date_str, "%Y-%m-%d")
        dt_start = dt_end - timedelta(days=7)
        start_date_str = dt_start.strftime("%Y-%m-%d")
        
        console.print("[dim]Input Events (Last 7 Days):[/dim]")
        for t in selected_tickers:
            news = loader.get_company_news(t, date_str, start_date=start_date_str, limit=3)
            if news:
                console.print(f"  {t}: {len(news)} events (e.g. {news[0].title[:50]}...)")
            else:
                console.print(f"  {t}: No events")
        
        # Run Agent
        state = {
            "data": {
                "tickers": selected_tickers,
                "end_date": date_str,
                "start_date": "2020-01-01" 
            },
            "metadata": {
                "allocator_mode": "global_batch",
                "show_reasoning": False,
                "model_name": model_name,
                "model_provider": model_provider
            }
        }
        
        try:
            output = news_sentiment_allocator(state)
            allocs = output["data"]["allocator_decisions"]["news_sentiment_allocator"]["allocations"]
            
            # Print Table
            table = Table(show_header=True)
            table.add_column("Ticker")
            table.add_column("Bet", style="bold")
            table.add_column("Amount")
            table.add_column("Reasoning Snippet")
            
            for a in allocs:
                reason = a["reasoning"][:80] + "..." if len(a["reasoning"]) > 80 else a["reasoning"]
                table.add_row(a["ticker"], a["direction"].upper(), f"{a['amount']:.1f}%", reason)
                
            console.print(table)
            console.print("\n")
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    verify_random_global()
