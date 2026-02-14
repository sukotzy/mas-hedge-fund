'''Analyzed the volume/types of news events before we built the agent.'''

import wrds
import os
import pandas as pd
from dotenv import load_dotenv

def main():
    load_dotenv()
    try:
        db = wrds.Connection(wrds_username=os.getenv("WRDS_USERNAME"))
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # 1. Inspect Dow Jones All (djones_all)
    print("\n--- Inspecting Dow Jones All (djones_all) ---")
    try:
        tables = db.list_tables(library='djones_all')
        print(f"Tables in 'djones_all': {tables}")
        
        # Check specific common tables if they exist
        target_table = None
        for t in tables:
            if 'daily' in t or 'news' in t or 'headline' in t:
                target_table = t
                break
        
        if target_table:
            print(f"Sampling table: {target_table}")
            query = f"SELECT * FROM djones_all.{target_table} LIMIT 5"
            df = db.raw_sql(query)
            print(df.columns.tolist())
            print(df.head())
        else:
             print("Could not identify main news table. Listing all tables again...")
             print(tables)

    except Exception as e:
        print(f"Error accessing djones_all: {e}")

    # 2. Inspect Capital IQ Key Developments (ciq.wrds_keydev)
    print("\n--- Inspecting Capital IQ Key Dev (ciq.wrds_keydev) ---")
    try:
        # Check columns
        query = "SELECT * FROM ciq.wrds_keydev LIMIT 5"
        df = db.raw_sql(query)
        print("Columns:", df.columns.tolist())
        # Print event types/headlines
        print(df[['keydeveventtypeid', 'keydevid', 'headline']].head())
    except Exception as e:
        print(f"Error accessing ciq.wrds_keydev: {e}")

    db.close()

if __name__ == "__main__":
    main()
