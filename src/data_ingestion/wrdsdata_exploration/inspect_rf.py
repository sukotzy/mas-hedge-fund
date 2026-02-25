import wrds
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    print("Connecting to WRDS...")
    db = wrds.Connection(wrds_username=os.getenv("WRDS_USERNAME"))
    
    print("Tables in 'ff' library:")
    tables = db.list_tables(library='ff')
    print(tables)
    
    if "factors_daily" in tables:
        print("\nColumns in ff.factors_daily:")
        print(db.describe_table(library='ff', table='factors_daily'))
        
        print("\nSample data:")
        data = db.raw_sql("SELECT date, rf FROM ff.factors_daily ORDER BY date DESC LIMIT 5", date_cols=['date'])
        print(data)

if __name__ == "__main__":
    main()
