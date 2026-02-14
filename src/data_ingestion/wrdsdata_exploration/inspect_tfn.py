'''Scripts that discovered we didn't have access to Insider Trading data (TFN).'''

import wrds
import os
from dotenv import load_dotenv

def main():
    load_dotenv()
    try:
        db = wrds.Connection(wrds_username=os.getenv("WRDS_USERNAME"))
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    print("Checking TFN columns...")
    try:
        cols = db.raw_sql("SELECT * FROM tfn.table1 LIMIT 1").columns.tolist()
        print(f"tfn.table1 columns: {cols}")
    except Exception as e:
        print(f"Error checking tfn.table1: {e}")

    db.close()

if __name__ == "__main__":
    main()
