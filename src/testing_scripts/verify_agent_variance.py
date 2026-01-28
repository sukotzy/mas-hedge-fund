
import os
import sys
import pandas as pd
import random
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.agents.allocators.news_sentiment_allocator import news_sentiment_allocator
from src.data.loader import LocalDataLoader
from dotenv import load_dotenv

# Load env vars
load_dotenv()

console = Console()

def verify_variance():
    ticker = "AAPL"
    console.print(f"[bold blue]Running Variance Test for {ticker} (Batch Mode)[/bold blue]")
    
    # 1. auto-detect keys
    model_name = "gpt-4o"
    model_provider = "OpenAI"
    if os.getenv("DASHSCOPE_API_KEY"):
        model_name = "qwen-max"
        model_provider = "Dashscope"
    elif os.getenv("GOOGLE_API_KEY"):
        model_name = "gemini-1.5-pro"
        model_provider = "Google"
    
    console.print(f"Model: [cyan]{model_name}[/cyan] ({model_provider})")

    # 2. Find 5 dates with actual events
    console.print("Finding active event dates...")
    loader = LocalDataLoader(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    keydev_path = os.path.join(loader.data_dir, "raw", "sp500_keydev.parquet")
    
    if not os.path.exists(keydev_path):
        console.print("[red]Error: KeyDev data not found.[/red]")
        return

    df = pd.read_parquet(keydev_path)
    # Loader initialized tables in __init__, so we can look up gvkey directly?
    # Actually, verify_variance creates a NEW loader instance above.
    # But get_permno/get_gvkey rely on self.constituents/ccm_links being loaded.
    
    # Check if we can find GVKEY via the loader's helper methods or direct DF access
    # Loader doesn't have a simple ticker->gvkey dict exposed publicly, but get_permno -> get_gvkey works.
    permno = loader.get_permno(ticker, "2024-01-01") # Use a valid date to get the ID
    gvkey = loader.get_gvkey(permno) if permno else None
    
    if not gvkey:
        console.print(f"[red]GVKEY not found for {ticker}[/red]")
        return

    # Filter for AAPL and Group by Date
    events = df[df['gvkey'] == gvkey].copy()
    events['date_str'] = pd.to_datetime(events['announcedate']).dt.strftime("%Y-%m-%d")
    
    daily_counts = events.groupby('date_str').size()
    # Filter for days with at least 1 event (preferably >1 for batter batch testing)
    active_days = daily_counts[daily_counts >= 1].index.tolist()
    
    if len(active_days) < 5:
        test_dates = active_days
    else:
        # Pick 5 random ones, but maybe prioritize high volume?
        # Let's pick 3 high volume and 2 random
        busy_days = daily_counts[daily_counts >= 2].index.tolist()
        if len(busy_days) >= 3:
            test_dates = random.sample(busy_days, 3)
            remaining = [d for d in active_days if d not in test_dates]
            test_dates.extend(random.sample(remaining, min(2, len(remaining))))
        else:
            test_dates = random.sample(active_days, 5)
    
    test_dates = sorted(test_dates)
    console.print(f"Selected Test Dates: {test_dates}\n")
    
    # 3. Run Agent for each date
    results = []
    
    for date_str in test_dates:
        console.print(Panel(f"Testing Date: {date_str}", style="bold yellow"))
        
        # Peek at events for verification
        day_events = events[events['date_str'] == date_str]
        headlines = day_events['headline'].tolist()[:3]
        console.print(f"[bold]Events ({len(day_events)} total):[/bold]")
        for h in headlines:
            console.print(f"- {h}")
        if len(headers := headlines) < len(day_events):
            console.print("...")

        # Run Agent
        state = {
            "data": {
                "tickers": [ticker],
                "end_date": date_str,
                "start_date": "2023-01-01"
            },
            "metadata": {
                "allocator_mode": "batch",
                "show_reasoning": False,
                "model_name": model_name,
                "model_provider": model_provider
            }
        }
        
        try:
            # We want to capture the INTERMEDIATE signal too (from ticker_summaries)
            # But the agent function doesn't return that directly in 'data'. It returns 'allocator_decisions'.
            # However, the reasoning in 'allocator_decisions' often contains the summary.
            output = news_sentiment_allocator(state)
            decision = output["data"]["allocator_decisions"]["news_sentiment_allocator"]
            
            # Extract allocation for AAPL
            aapl_alloc = next((a for a in decision["allocations"] if a["ticker"] == ticker), None)
            cash_alloc = next((a for a in decision["allocations"] if a["ticker"] == "CASH"), None)
            
            res = {
                "date": date_str,
                "events_count": len(day_events),
                "headlines": headlines,
                "aapl_dir": aapl_alloc["direction"] if aapl_alloc else "N/A",
                "aapl_amt": aapl_alloc["amount"] if aapl_alloc else 0,
                "cash_amt": cash_alloc["amount"] if cash_alloc else 0,
                "reasoning": aapl_alloc["reasoning"] if aapl_alloc else "No alloc found"
            }
            results.append(res)
            
            # Use upper() for display, works for UP/DOWN/NEUTRAL
            console.print(f"[bold]Result:[/bold] {res['aapl_dir'].upper()} ({res['aapl_amt']}%)")
            console.print(f"[dim]{res['reasoning']}[/dim]\n")
            
        except Exception as e:
            console.print(f"[red]Error running agent: {e}[/red]")

    # 4. Summary Table
    console.print("\n[bold white]Variance Test Summary[/bold white]")
    table = Table(show_lines=True)
    table.add_column("Date", style="cyan")
    table.add_column("Events", justify="center")
    table.add_column("Decision", style="bold")
    table.add_column("Reasoning Snippet")
    
    for r in results:
        reason_snippet = r['reasoning'][:100] + "..." if len(r['reasoning']) > 100 else r['reasoning']
        table.add_row(
            r['date'], 
            str(r['events_count']), 
            f"{r['aapl_dir'].upper()} {r['aapl_amt']}%", 
            reason_snippet
        )
    
    console.print(table)

if __name__ == "__main__":
    verify_variance()
