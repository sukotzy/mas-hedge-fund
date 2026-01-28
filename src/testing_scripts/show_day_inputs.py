
import os
import sys
from rich.console import Console
from rich.panel import Panel

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.tools.api import get_company_news
from src.utils.api_key import get_api_key_from_state

console = Console()

def show_inputs():
    tickers = ["AAPL", "MSFT", "GOOGL"]
    end_date = "2024-01-05"
    
    console.print(f"[bold blue]Agent Inputs for {end_date}[/bold blue]")
    
    # We need an API key if using the real tool, but local loader might not need it?
    # get_company_news uses the KEY to decide if it goes to API or Local.
    # But wait, our current implementation of `get_company_news` in `src.tools.api` checks for "FINANCIAL_DATASETS_API_KEY".
    # However, we are essentially using LocalLoader now? 
    # Let's check `src/tools/api.py`.
    # Actually, the quickest way is to just use LocalDataLoader directly if we want to be sure what the local agent saw.
    
    from src.data.loader import LocalDataLoader
    loader = LocalDataLoader(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    
    for ticker in tickers:
        console.print(f"\n[bold yellow]--- {ticker} ---[/bold yellow]")
        # The agent uses limit=5
        news = loader.get_company_news(ticker, end_date, limit=5)
        
        if not news:
            console.print("[dim]No events found.[/dim]")
            continue
            
        for n in news:
            console.print(f"[green]{n.date}[/green]: {n.title}")
            console.print(f"[dim]Source: {n.source}[/dim]")

if __name__ == "__main__":
    show_inputs()
