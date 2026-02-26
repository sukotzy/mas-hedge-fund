
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.meta_manager import settle_bets
from src.schemas import Bet, MarketSignal

def test_settle_bets_alpha_based_win():
    # Setup
    agent_capital = {
        "TestAgent": {
            "external_capital": 50000.0,
            "internal_capital": 50000.0,
            "allocated_capital": 50000.0,
            "roi_history": []
        },
        "LoserAgent": {
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
                    "amount": 100.0, # 100% of portfolio
                    "reasoning": "test"
                }
            ],
            "metrics": {}
        },
        "LoserAgent": {
            "allocations": [
                {
                    "ticker": "AAPL",
                    "direction": "short",
                    "amount": 100.0, # 100% of portfolio
                    "reasoning": "test fail"
                }
            ],
            "metrics": {}
        }
    }
    previous_prices = {"AAPL": 100.0}
    current_prices = {"AAPL": 110.0} # +10% gain.

    # Execute
    # TestAgent ret: +10%
    # LoserAgent ret: -10%
    # Avg ret: 0%
    # TestAgent alpha: +10%. Transfer = 100,000 * 0.10 = +10,000
    # LoserAgent alpha: -10%. Transfer = 100,000 * -0.10 = -10,000
    new_capital = settle_bets(agent_capital, previous_bets, current_prices, previous_prices)
    
    # Verify Loser
    # Lost 10000. Split 50/50.
    # Ext: 50000 - 5000 = 45000
    # Int: 50000 - 5000 = 45000
    assert new_capital["LoserAgent"]["external_capital"] == 45000.0
    assert new_capital["LoserAgent"]["internal_capital"] == 45000.0
    
    # Verify Winner
    # Won 10000. Split 50/50.
    # Ext: 50000 + 5000 = 55000
    # Int: 50000 + 5000 = 55000
    assert new_capital["TestAgent"]["external_capital"] == 55000.0
    assert new_capital["TestAgent"]["internal_capital"] == 55000.0

def test_settle_bets_alpha_asymmetric_split():
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
                {"ticker": "AAPL", "direction": "long", "amount": 100.0, "reasoning": "win"}
            ],
            "metrics": {}
        },
        "LoserAgent": {
             "allocations": [
                {"ticker": "AAPL", "direction": "short", "amount": 100.0, "reasoning": "lose"}
            ],
            "metrics": {}
        }
    }
    
    # TestAgent ret: +10%
    # LoserAgent ret: -10%
    # Avg ret: 0%
    # TestAgent alpha: +10%. Transfer = 100,000 * 0.10 = +10,000
    new_capital = settle_bets(agent_capital, previous_bets, current_prices, previous_prices)
    
    # Split: 90/10.
    # Ext gain: 9000. Int gain: 1000.
    assert new_capital["TestAgent"]["external_capital"] == 99000.0
    assert new_capital["TestAgent"]["internal_capital"] == 11000.0

def test_settle_bets_alpha_complex_portfolio():
    # Setup
    agent_capital = {
        "AgentA": {
            "external_capital": 50000.0,
            "internal_capital": 50000.0,
            "allocated_capital": 50000.0,
            "roi_history": []
        },
        "AgentB": {
            "external_capital": 50000.0,
            "internal_capital": 50000.0,
            "allocated_capital": 50000.0,
             "roi_history": []
        }
    }
    previous_prices = {"AAPL": 100.0, "MSFT": 200.0, "CASH": 1.0}
    current_prices = {"AAPL": 110.0, "MSFT": 210.0, "CASH": 1.0} 
    # AAPL +10%
    # MSFT +5%
    # CASH +0%
    
    previous_bets = {
        "AgentA": {
            "allocations": [
                {"ticker": "AAPL", "direction": "long", "amount": 60.0, "reasoning": ""},
                {"ticker": "MSFT", "direction": "short", "amount": 40.0, "reasoning": ""}
            ],
            "metrics": {}
        },
        "AgentB": {
             "allocations": [
                {"ticker": "AAPL", "direction": "long", "amount": 30.0, "reasoning": ""},
                {"ticker": "CASH", "direction": "long", "amount": 70.0, "reasoning": ""}
            ],
            "metrics": {}
        }
    }
    
    # Execute
    # AgentA Returns:
    # AAPL: Long 60% weight * +10% = +6.0%
    # MSFT: Short 40% weight * +5% = -2.0%
    # Total AgentA = 4.0%
    
    # AgentB Returns:
    # AAPL: Long 30% weight * +10% = +3.0%
    # CASH: Long 70% weight * +0% = 0.0%
    # Total AgentB = 3.0%
    
    # Average Return (Benchmark) = (4.0% + 3.0%) / 2 = 3.5%
    
    # Alpha A = 4.0% - 3.5% = +0.5% (+0.005)
    # Alpha B = 3.0% - 3.5% = -0.5% (-0.005)
    
    # Transfers:
    # AgentA = 100,000 * 0.005 = +$500
    # AgentB = 100,000 * -0.005 = -$500
    
    new_capital = settle_bets(agent_capital, previous_bets, current_prices, previous_prices)
    
    # AgentA Split 50/50: Ext +$250, Int +$250
    assert new_capital["AgentA"]["external_capital"] == 50250.0
    assert new_capital["AgentA"]["internal_capital"] == 50250.0
    
    # AgentB Split 50/50: Ext -$250, Int -$250
    assert new_capital["AgentB"]["external_capital"] == 49750.0
    assert new_capital["AgentB"]["internal_capital"] == 49750.0


if __name__ == "__main__":
    try:
        test_settle_bets_alpha_based_win()
        print("test_settle_bets_alpha_based_win PASSED")
        test_settle_bets_alpha_asymmetric_split()
        print("test_settle_bets_alpha_asymmetric_split PASSED")
        test_settle_bets_alpha_complex_portfolio()
        print("test_settle_bets_alpha_complex_portfolio PASSED")
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
    except Exception as e:
        print(f"ERROR: {e}")
