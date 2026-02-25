import wrds
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def fetch_gics():
    print("Connecting to WRDS...")
    username = os.getenv("WRDS_USERNAME")
    db = wrds.Connection(wrds_username=username) if username else wrds.Connection()
    
    print("Checking for comp.company...")
    # gsector: GICS Sector code
    # gicdesc: GICS Sector Description
    # sic: Standard Industry Classification
    # naics: North American Industry Classification
    query = """
        SELECT gvkey, conm, gsector, gicdesc, sic, naics
        FROM comp.company
    """
    
    try:
        company_info = db.raw_sql(query)
        print(f"Fetched {len(company_info)} rows from comp.company.")
        print("Sample Data:")
        print(company_info[['gvkey', 'conm', 'gsector', 'gicdesc']].head())
        
        # Save it to raw data for testing
        out_path = "data/raw/company_info.parquet"
        company_info.to_parquet(out_path)
        print(f"\nSaved to {out_path}")
        
    except Exception as e:
        print(f"Error fetching comp.company: {e}")
        
    db.close()

if __name__ == "__main__":
    fetch_gics()
