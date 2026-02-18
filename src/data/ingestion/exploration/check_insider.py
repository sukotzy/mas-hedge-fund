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

    print("Checking for Insider Trading Libraries...")
    libs = db.list_libraries()
    
    keywords = ['insider', 'tfn', 'thomson', 'ownership', 'transactions']
    
    found_libs = []
    for lib in libs:
        if any(k in lib.lower() for k in keywords):
            found_libs.append(lib)
            print(f"- {lib}")

    if 'tfn' in found_libs:
        print("\nChecking Thomson Reuters (tfn) tables:")
        try:
            tables = db.list_tables(library='tfn')
            print(tables[:10])
        except:
            print("Access denied to tfn")

    db.close()

if __name__ == "__main__":
    main()
