import wrds
import os
import pandas as pd
from dotenv import load_dotenv

def main():
    load_dotenv()
    print("Connecting to WRDS...")
    try:
        username = os.getenv("WRDS_USERNAME")
        db = wrds.Connection(wrds_username=username) if username else wrds.Connection()
        
        print("Fetching Company Info (GICS Sectors) from comp.company...")
        # Removed 'tic' because some WRDS setups have it differently.
        # We can map gvkey to ticker using ccm_links if needed later.
        query = """
            SELECT gvkey, gsector
            FROM comp.company
        """
        data = db.raw_sql(query)
        
        if not data.empty:
            out_dir = "data/raw"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "company_info.parquet")
            data.to_parquet(out_path)
            print(f"Success! Saved {len(data)} rows to {out_path}.")
        else:
            print("Query returned no data.")
            
        db.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
