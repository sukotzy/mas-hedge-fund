import pandas as pd
import json
import random
from pathlib import Path

def verify_candidates():
    data_dir = Path("data/processed")
    hint_path = data_dir / "daily_candidates_with_hint.parquet"
    no_hint_path = data_dir / "daily_candidates_no_hint.parquet"
    
    print("=" * 60)
    print("VERIFYING DAILY CANDIDATES")
    print("=" * 60)
    
    # 1. Verify With Hint
    print(f"\n[1] Checking: {hint_path}")
    if hint_path.exists():
        df_hint = pd.read_parquet(hint_path)
        print(f"Rows: {len(df_hint)}")
        print(f"Date Range: {df_hint.index.min()} to {df_hint.index.max()}")
        
        # Sample a random day
        if not df_hint.empty:
            sample_date = df_hint.index[0] # Take first available for consistency, or random
            # Let's try to find a day with data, ideally mid-sample
            sample_date = df_hint.index[len(df_hint)//2]
            
            print(f"\n--- Sample Day (With Hint): {sample_date} ---")
            tasks_json = df_hint.loc[sample_date, 'tasks']
            tasks = json.loads(tasks_json)
            print(json.dumps(tasks, indent=2))
            
            # Verify Structure
            actions = [t['action'] for t in tasks]
            print(f"Actions present: {set(actions)}")
    else:
        print("FILE NOT FOUND")

    # 2. Verify No Hint
    print(f"\n[2] Checking: {no_hint_path}")
    if no_hint_path.exists():
        df_no_hint = pd.read_parquet(no_hint_path)
        print(f"Rows: {len(df_no_hint)}")
        print(f"Date Range: {df_no_hint.index.min()} to {df_no_hint.index.max()}")
        
        # Sample the SAME day to compare
        if not df_no_hint.empty and 'sample_date' in locals():
            if sample_date in df_no_hint.index:
                print(f"\n--- Sample Day (No Hint): {sample_date} ---")
                tasks_json = df_no_hint.loc[sample_date, 'tasks']
                tasks = json.loads(tasks_json)
                print(json.dumps(tasks, indent=2))
                
                # Verify Structure
                actions = [t['action'] for t in tasks]
                print(f"Actions present: {set(actions)}")
            else:
                print(f"Sample date {sample_date} not found in No Hint file.")
    else:
        print("FILE NOT FOUND")

if __name__ == "__main__":
    verify_candidates()
