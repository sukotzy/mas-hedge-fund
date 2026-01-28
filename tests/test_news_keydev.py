
import sys
import os
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.news_sentiment import news_sentiment_agent
from src.graph.state import AgentState

# Load env vars
load_dotenv()

def test_news_agent():
    print("Testing News Sentiment Agent (KeyDev)...")
    load_dotenv() # Ensure loaded
    print(f"Current CWD: {os.getcwd()}")
    print(f"Env keys available: {[k for k in os.environ.keys() if 'API' in k or 'KEY' in k]}")
    
    # Dynamic Model Selection based on available keys
    model_name = "gpt-4o"
    model_provider = "OpenAI"
    
    if os.getenv("DASHSCOPE_API_KEY"):
        print("Detected DASHSCOPE_API_KEY, using qwen-max")
        model_name = "qwen-max"
        model_provider = "Dashscope"
    elif os.getenv("GOOGLE_API_KEY"):
        print("Detected GOOGLE_API_KEY, using gemini-1.5-pro")
        model_name = "gemini-1.5-pro"
        model_provider = "Google"
    elif os.getenv("OPENAI_API_KEY"):
         print("Detected OPENAI_API_KEY, using gpt-4o")
         model_name = "gpt-4o"
         model_provider = "OpenAI"
    else:
        print("WARNING: No recognized API key found (Dashscope, Google, OpenAI). Test may fail.")

    # Setup state
    state = {
        "data": {
            "tickers": ["AAPL", "AMD"],
            "end_date": "2024-12-31",
            "start_date": "2024-01-01",
            "analyst_signals": {}
        },
        "metadata": {
            "show_reasoning": True,
            "model_name": model_name,
            "model_provider": model_provider,
        }
    }
    
    # Run agent
    try:
        result = news_sentiment_agent(state)
        
        # Check output
        signals = result["data"]["analyst_signals"]["news_sentiment_agent"]
        print("\n\nAnalyzed Signals:")
        for ticker, data in signals.items():
            print(f"\nTicker: {ticker}")
            print(f"Signal: {data['signal']}")
            print(f"Confidence: {data['confidence']}")
            print(f"Reasoning: {data['reasoning']}")
            
    except Exception as e:
        print(f"\nTest Failed with error: {e}")
        print("Please ensure your .env file has valid keys for Dashscope, Google, or OpenAI.")

if __name__ == "__main__":
    test_news_agent()
