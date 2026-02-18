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

    print("Checking for MarketPsych / TRMI Libraries...")
    libs = db.list_libraries()
    
    keywords = ['psych', 'trmi', 'marketpsych', 'refinitiv', 'sentiment']
    
    found_libs = []
    for lib in libs:
        if any(k in lib.lower() for k in keywords):
            found_libs.append(lib)
            print(f"- {lib}")

    if not found_libs:
        print("No specific MarketPsych libraries found.")
    else:
        # If found, list tables to confirm it's not empty/trial
        first_lib = found_libs[0]
        try:
            print(f"Inspecting tables in {first_lib}...")
            tables = db.list_tables(library=first_lib)
            print(tables[:10])
        except:
            print(f"Could not access {first_lib}")

    db.close()

if __name__ == "__main__":
    main()
