
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.meta_manager import settle_bets
from src.schemas import Bet, MarketSignal

def test_settle_bets_dual_tranche_win():
    # Setup
    agent_capital = {
        "TestAgent": {
            "external_capital": 50000.0,
            "internal_capital": 50000.0,
            "allocated_capital": 50000.0,
            "roi_history": []
        }
    }
    previous_bets = {
        "TestAgent": {
            "allocations": [
                {
                    "ticker": "AAPL",
                    "direction": "long",
                    "amount": 1.0, # 1% of 100000 = 1000 dollars
                    "reasoning": "test"
                }
            ],
            "metrics": {}
        }
    }
    previous_prices = {"AAPL": 100.0}
    current_prices = {"AAPL": 110.0} # +10% gain. Win.

    # Execute
    # Since it's Zero-Sum, we need a loser to fund the winner.
    # Let's add a LoserAgent.
    agent_capital["LoserAgent"] = {
        "external_capital": 50000.0,
        "internal_capital": 50000.0, 
        "allocated_capital": 50000.0,
        "roi_history": []
    }
    previous_bets["LoserAgent"] = {
        "allocations": [
            {
                "ticker": "AAPL",
                "direction": "short",
                "amount": 1.0, # 1% of 100000 = Loses 1000
                "reasoning": "test fail"
            }
        ],
        "metrics": {}
    }
    
    # Run
    new_capital = settle_bets(agent_capital, previous_bets, current_prices, previous_prices)
    
    # Verify Loser
    # Lost 1000. Split 50/50.
    # Ext: 50000 - 500 = 49500
    # Int: 50000 - 500 = 49500
    assert new_capital["LoserAgent"]["external_capital"] == 49500.0
    assert new_capital["LoserAgent"]["internal_capital"] == 49500.0
    
    # Verify Winner
    # Won share of pool (1000). Split 50/50.
    # Ext: 50000 + 500 = 50500
    # Int: 50000 + 500 = 50500
    assert new_capital["TestAgent"]["external_capital"] == 50500.0
    assert new_capital["TestAgent"]["internal_capital"] == 50500.0

def test_settle_bets_asymmetric_split():
    # TestAgent has 90% External, 10% Internal
    agent_capital = {
        "TestAgent": {
            "external_capital": 90000.0,
            "internal_capital": 10000.0,
            "allocated_capital": 90000.0,
            "roi_history": []
        },
        "LoserAgent": {
            "external_capital": 50000.0,
            "internal_capital": 50000.0,
            "allocated_capital": 50000.0,
             "roi_history": []
        }
    }
    previous_prices = {"AAPL": 100.0}
    current_prices = {"AAPL": 110.0}
    
    previous_bets = {
        "TestAgent": {
            "allocations": [
                {"ticker": "AAPL", "direction": "long", "amount": 1.0, "reasoning": "win"}
            ],
            "metrics": {}
        },
        "LoserAgent": {
             "allocations": [
                {"ticker": "AAPL", "direction": "short", "amount": 1.0, "reasoning": "lose"}
            ],
            "metrics": {}
        }
    }
    
    new_capital = settle_bets(agent_capital, previous_bets, current_prices, previous_prices)
    
    # Winner Pool = 1000 (from Loser)
    # TestAgent wins 1000.
    # Split: 90/10.
    # Ext gain: 900. Int gain: 100.
    assert new_capital["TestAgent"]["external_capital"] == 90900.0
    assert new_capital["TestAgent"]["internal_capital"] == 10100.0

if __name__ == "__main__":
    try:
        test_settle_bets_dual_tranche_win()
        print("test_settle_bets_dual_tranche_win PASSED")
        test_settle_bets_asymmetric_split()
        print("test_settle_bets_asymmetric_split PASSED")
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
    except Exception as e:
        print(f"ERROR: {e}")
