import wrds
import sys
import os
from dotenv import load_dotenv

def check_wrds():
    try:
        load_dotenv()
        username = os.getenv("WRDS_USERNAME")
        password = os.getenv("WRDS_PASSWORD")
        print(f"Loaded WRDS_USERNAME: {username}")
        
        # Debugging pgpass location
        appdata = os.environ.get('APPDATA')
        print(f"APPDATA: {appdata}")
        if appdata:
            pgpass_path = os.path.join(appdata, 'postgresql', 'pgpass.conf')
            if os.path.exists(pgpass_path):
                print(f"Found pgpass.conf at: {pgpass_path}")
                # Optional: Read file to ensure username matches (without printing password)
                try:
                    with open(pgpass_path, 'r') as f:
                        content = f.read()
                        if username and username in content:
                            print(f"confirmed '{username}' is present in pgpass.conf")
                        else:
                            print(f"WARNING: '{username}' NOT found in pgpass.conf")
                except Exception as e:
                    print(f"Could not read pgpass.conf: {e}")
            else:
                print(f"pgpass.conf NOT FOUND at: {pgpass_path}")
        
        print("Attempting to connect to WRDS...")
        
        # attempt to connect
        if username and password:
             db = wrds.Connection(wrds_username=username, password=password)
        elif username:
            print("Connecting with username only (expecting pgpass)...")
            db = wrds.Connection(wrds_username=username) 
        else:
            print("WARNING: WRDS_USERNAME not found in .env, falling back to default/interactive")
            db = wrds.Connection()
            
        print("Connection successful!")
        
        print("\nListing available libraries...")
        libs = db.list_libraries()
        libs.sort()
        
        sentiment_keywords = ['sentiment', 'raven', 'thomson', 'reuters', 'news', 'rpna', 'trna']
        found_libs = [l for l in libs if any(k in l.lower() for k in sentiment_keywords)]
        
        if found_libs:
            print("\nFound potential sentiment/news libraries:")
            for lib in found_libs:
                print(f"\nLibrary: {lib}")
                try:
                    tables = db.list_tables(lib)
                    print(f"  Tables ({len(tables)}): {tables[:20]}")
                except Exception as e:
                    print(f"  Error listing tables: {e}")
        else:
            print("\nNo obvious sentiment libraries found with keywords:", sentiment_keywords)
            
        print("\nChecking for specific common datasets:")
        common_sets = ['ravenpack_common_stock', 'thomson_reuters']
        for s in common_sets:
            if s in libs:
                print(f"  {s} is available.")
            else:
                print(f"  {s} NOT found.")

    except Exception as e:
        print(f"\nFailed: {e}")
        print("Note: This script requires WRDS credentials set up (e.g. .pgpass or environment variables).")

if __name__ == "__main__":
    check_wrds()
