from src.schemas import Bet, MarketSignal
from src.graph.state import AgentState, show_agent_reasoning
from langchain_core.messages import HumanMessage
import json

class BettingMarket:
    """
    Aggregates bets from multiple agents to determine consensus values.
    """
    
    def __init__(self):
        self.bets: dict[str, list[Bet]] = {}

    def place_bet(self, bet: Bet):
        """
        Accepts a bet from an agent.
        """
        if bet.ticker not in self.bets:
            self.bets[bet.ticker] = []
        self.bets[bet.ticker].append(bet)

    def calculate_consensus(self) -> dict[str, float]:
        """
        Calculates the consensus value for each ticker based on the net capital flow.
        Value = Sum(Bet_long) - Sum(Bet_short)
        """
        consensus_values = {}
        
        for ticker, ticker_bets in self.bets.items():
            total_long = sum(b.amount for b in ticker_bets if b.direction == MarketSignal.LONG)
            total_short = sum(b.amount for b in ticker_bets if b.direction == MarketSignal.SHORT)
            
            # Net capital flow
            consensus_values[ticker] = total_long - total_short
            
        return consensus_values

    def get_bet_summary(self, ticker: str) -> dict:
        """
        Returns a summary of bets for a specific ticker.
        """
        if ticker not in self.bets:
            return {"long": 0, "short": 0, "total_capital": 0, "consensus": 0}
            
        ticker_bets = self.bets[ticker]
        total_long = sum(b.amount for b in ticker_bets if b.direction == MarketSignal.LONG)
        total_short = sum(b.amount for b in ticker_bets if b.direction == MarketSignal.SHORT)

        return {
            "long": total_long,
            "short": total_short,
            "total_capital": sum(b.amount for b in ticker_bets),
            "consensus": total_long - total_short
        }

def betting_market_node(state: AgentState, agent_id: str = "betting_market"):
    """
    Node that runs the betting market aggregation.
    """
    data = state["data"]
    analyst_signals = data.get("analyst_signals", {})
    
    market = BettingMarket()
    
    # Collect bets from all analysts
    for agent_name, signal_data in analyst_signals.items():
        # Skip non-analyst signals (like risk manager if it was there)
        if "risk_management" in agent_name or "portfolio" in agent_name:
            continue
            
        # signal_data is a dict of ticker -> bet_dump
        for ticker, bet_dump in signal_data.items():
            try:
                # Reconstruct Bet object
                bet = Bet(**bet_dump)
                market.place_bet(bet)
            except Exception as e:
                print(f"Error processing bet from {agent_name} for {ticker}: {e}")
                
    # Calculate consensus
    consensus_values = market.calculate_consensus()
    
    # Store in state
    data["consensus_values"] = consensus_values
    
    # Create summary message
    summary = {ticker: market.get_bet_summary(ticker) for ticker in consensus_values}
    message = HumanMessage(
        content=json.dumps(summary),
        name=agent_id,
    )
    
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(summary, "Betting Market Consensus")
        
    return {
        "messages": [message],
        "data": data
    }
