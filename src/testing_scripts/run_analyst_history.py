
import sys
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.agents.allocators.news_sentiment_allocator import news_sentiment_allocator
from src.utils.progress import progress

# Load env vars
load_dotenv()

def generate_training_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    output_file: str = "data/analyst_decisions.jsonl"
):
    print(f"Generating Analyst History for {len(tickers)} tickers from {start_date} to {end_date}...")
    
    # Generate weekly dates (every Friday) to simulate rebalancing
    dates = pd.date_range(start=start_date, end=end_date, freq="W-FRI")
    
    results = []
    
    for current_date in dates:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"\nProcessing Date: {date_str}")
        
        # Auto-detect Model Config (similar to test scripts)
        model_name = "gpt-4o"
        model_provider = "OpenAI"
        if os.getenv("DASHSCOPE_API_KEY"):
            model_name = "qwen-max"
            model_provider = "Dashscope"
        elif os.getenv("GOOGLE_API_KEY"):
            model_name = "gemini-1.5-pro"
            model_provider = "Google"

        # Mock State
        state = {
            "data": {
                "tickers": tickers,
                "end_date": date_str,
                "start_date": (current_date - timedelta(days=90)).strftime("%Y-%m-%d"),
                "analyst_signals": {}, # Standard agents (empty)
                "allocator_decisions": {} # Where our output goes
            },
            "metadata": {
                "show_reasoning": False,
                "model_name": model_name,
                "model_provider": model_provider
            }
        }
        
        try:
            # Run the Allocator
            result_state = news_sentiment_allocator(state)
            
            # Extract decision
            decisions = result_state["data"].get("allocator_decisions", {})
            news_decision = decisions.get("news_sentiment_allocator")
            
            if news_decision:
                record = {
                    "date": date_str,
                    "agent": "news_sentiment_allocator",
                    "allocations": news_decision["allocations"],
                    "metrics": news_decision["metrics"]
                }
                
                # Append to file immediately (JSONL format)
                with open(output_file, "a") as f:
                    f.write(json.dumps(record) + "\n")
                
                print(f"  Saved allocation for {date_str}")
            else:
                print(f"  No allocation produced for {date_str}")
                
        except Exception as e:
            print(f"  Error processing {date_str}: {e}")

if __name__ == "__main__":
    # Example Universe: Top Tech + some others
    universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "AMD", "NVDA", "INTC"]
    
    # Run for a test period (e.g., Q1 2024)
    # Adjust dates as needed based on your data availability
    generate_training_data(
        tickers=universe,
        start_date="2024-01-01",
        end_date="2024-03-31"
    )
