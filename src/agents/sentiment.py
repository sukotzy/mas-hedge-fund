from langchain_core.messages import HumanMessage
from src.schemas import Bet, MarketSignal
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
import pandas as pd
import numpy as np
import json
from src.utils.api_key import get_api_key_from_state
from src.tools.api import get_insider_trades, get_company_news


##### Sentiment Agent #####
def sentiment_analyst_agent(state: AgentState, agent_id: str = "sentiment_analyst_agent"):
    """Analyzes market sentiment and generates trading signals for multiple tickers."""
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    # Initialize sentiment analysis for each ticker
    sentiment_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching company news")

        # Get the company news (Key Developments)
        # Note: We rely on the KeyDev data via get_company_news
        company_news = get_company_news(ticker, end_date, limit=100, api_key=api_key)

        # Get the sentiment from the company news
        sentiment = pd.Series([n.sentiment for n in company_news]).dropna()
        news_signals = np.where(sentiment == "negative", "bearish", 
                              np.where(sentiment == "positive", "bullish", "neutral")).tolist()
        
        progress.update_status(agent_id, ticker, "Calculating signals")
        
        # Calculate signals (100% weight on News since Insider is unavailable)
        bullish_signals = news_signals.count("bullish")
        bearish_signals = news_signals.count("bearish")
        
        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # Calculate confidence level based on the proportion
        total_signals = len(news_signals)
        confidence = 0
        if total_signals > 0:
            confidence = round((max(bullish_signals, bearish_signals) / total_signals) * 100, 2)
        
        # Create reasoning
        reasoning = {
            "news_sentiment": {
                "signal": overall_signal,
                "confidence": confidence,
                "metrics": {
                    "total_articles": total_signals,
                    "bullish_articles": bullish_signals,
                    "bearish_articles": bearish_signals,
                    "neutral_articles": news_signals.count("neutral"),
                }
            },
            "combined_analysis": {
                 "description": "Signal derived purely from Key Developments (Corporate Events) as Insider Data is unavailable.",
                 "signal": overall_signal
            }
        }

        # Determine MarketSignal
        if overall_signal == "bullish":
            signal_enum = MarketSignal.BULLISH
        elif overall_signal == "bearish":
            signal_enum = MarketSignal.BEARISH
        else:
            signal_enum = MarketSignal.NEUTRAL

        # Get agent capital (default 100k)
        agent_capital = data.get("agent_capital", {}).get(agent_id, {}).get("allocated_capital", 100000.0)
        
        # Calculate bet amount
        # Normalize confidence to 0-1
        confidence_score = confidence / 100.0
        bet_amount = agent_capital * 0.10 * confidence_score if signal_enum != MarketSignal.NEUTRAL else 0.0

        # Create Bet object
        bet = Bet(
            ticker=ticker,
            direction=signal_enum,
            amount=bet_amount,
            conviction=confidence_score,
            reasoning=json.dumps(reasoning)
        )

        sentiment_analysis[ticker] = bet.model_dump(mode='json')

        progress.update_status(agent_id, ticker, "Done", analysis=json.dumps(reasoning, indent=4))

    # Create the sentiment message
    message = HumanMessage(
        content=json.dumps(sentiment_analysis),
        name=agent_id,
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(sentiment_analysis, "Sentiment Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"][agent_id] = sentiment_analysis

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": data,
    }
