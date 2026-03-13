import wrds
import sys
import os
from dotenv import load_dotenv
import pandas as pd

def check_ravenpack():
    try:
        load_dotenv()
        username = os.getenv("WRDS_USERNAME")
        password = os.getenv("WRDS_PASSWORD")
        
        print("Connecting to WRDS...")
        if username and password:
             db = wrds.Connection(wrds_username=username, password=password)
        elif username:
            db = wrds.Connection(wrds_username=username) 
        else:
            db = wrds.Connection()
            
        print("Connected.")
        
        lib = 'rpnasamp'
        table = 'rpa_full_equities'
        
        print(f"Describing table: {lib}.{table}")
        
        # describe table to get columns
        # Note: describe_table returns a pandas dataframe
        desc = db.describe_table(library=lib, table=table)
        print("Columns found:")
        print(desc[['name', 'type', 'nullable']])
        
        # Check for meaningful sentiment columns
        sentiment_cols = ['ess', 'css', 'ens', 'event_sentiment_score', 'composite_sentiment_score']
        found_sent = desc[desc['name'].isin(sentiment_cols)]
        
        if not found_sent.empty:
            print("\nFOUND SENTIMENT COLUMNS:")
            print(found_sent)
            
            # preview data
            query = f"SELECT * FROM {lib}.{table} LIMIT 5"
            print("\nPreviewing data:")
            df = db.raw_sql(query)
            print(df.head())
            
            # Check date range
            count_query = f"SELECT min(timestamp_utc) as start_date, max(timestamp_utc) as end_date, count(*) as n_rows FROM {lib}.{table}"
            print("\nChecking date range...")
            dates = db.raw_sql(count_query)
            print(dates)
            
        else:
            print(f"\nNo obvious sentiment columns found from list: {sentiment_cols}")
            
        
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    check_ravenpack()
