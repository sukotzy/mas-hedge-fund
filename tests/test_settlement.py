import unittest
from src.agents.meta_manager import settle_bets
from src.schemas import Bet, MarketSignal
import json

class TestSettlement(unittest.TestCase):
    def test_zero_sum_settlement(self):
        # Initial State
        agent_capital = {
            "AgentA": {"allocated_capital": 10000.0, "roi_history": []},
            "AgentB": {"allocated_capital": 10000.0, "roi_history": []},
        }
        
        # Bets: AgentA bets BULLISH, AgentB bets BEARISH on AAPL
        previous_bets = {
            "AgentA": {
                "AAPL": {
                    "ticker": "AAPL",
                    "direction": "bullish",
                    "amount": 1000.0,
                    "conviction": 1.0,
                    "reasoning": "test"
                }
            },
            "AgentB": {
                "AAPL": {
                    "ticker": "AAPL",
                    "direction": "bearish",
                    "amount": 1000.0,
                    "conviction": 1.0,
                    "reasoning": "test"
                }
            }
        }
        
        # Scenario 1: AAPL goes UP (AgentA wins, AgentB loses)
        previous_prices = {"AAPL": 100.0}
        current_prices = {"AAPL": 110.0} # +10%
        
        new_capital = settle_bets(agent_capital, previous_bets, current_prices, previous_prices)
        
        # AgentB should lose 1000
        # AgentA should gain AgentB's 1000
        
        self.assertEqual(new_capital["AgentB"]["allocated_capital"], 9000.0)
        self.assertEqual(new_capital["AgentA"]["allocated_capital"], 11000.0)
        
        # Total capital should remain 20000
        total = sum(c["allocated_capital"] for c in new_capital.values())
        self.assertEqual(total, 20000.0)
        
        print("Zero-Sum Verification Passed!")

if __name__ == '__main__':
    unittest.main()
