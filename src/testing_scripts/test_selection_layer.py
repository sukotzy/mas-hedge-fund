import argparse
import sys
from src.selection.pipeline import run_selection_pipeline
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Test Data Selection Layer")
    parser.add_argument("--date", type=str, default="2024-01-05", help="Analysis End Date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    print(f"--- Testing Selection Layer for {args.date} ---")
    
    try:
        print(f"\n=== TEST 1: With Hints (Default) ===")
        result_with = run_selection_pipeline(args.date, include_hint=True)
        
        print(f"Market State: {result_with['market_state']} (NTL: {result_with.get('ntl', 0):.4f})")
        print("Tasks:")
        for t in result_with['tasks']:
            print(f"  [{t['action'].upper()}] {t['ticker']} | Reason: {t['reason']}")
            
        
        print(f"\n=== TEST 2: Without Hints (Blind) ===")
        print("Tasks:")
        for t in result_with['tasks']:
            cluster_str = t['reason'].split(':')[0]
            print(f"  [ANALYZE] {t['ticker']} | Reason: {cluster_str} Representative (Hidden)")
            
    except Exception as e:
        print(f"Error running pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
