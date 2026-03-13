import wrds
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_latest_date():
    try:
        print("Connecting to WRDS...")
        username = os.getenv("WRDS_USERNAME")
        if username:
            db = wrds.Connection(wrds_username=username)
        else:
            db = wrds.Connection()
            
        print("Querying latest date in CRSP Daily Stock File (crsp.dsf)...")
        # Query for the maximum date
        query = "SELECT MAX(date) as latest_date FROM crsp.dsf"
        
        result = db.raw_sql(query)
        
        if not result.empty:
            latest_date = result['latest_date'].iloc[0]
            print(f"Latest available date in CRSP DSF: {latest_date}")
        else:
            print("Query returned no results.")
            
        db.close()
        
    except Exception as e:
        print(f"Error checking WRDS data: {e}")

if __name__ == "__main__":
    check_latest_date()
