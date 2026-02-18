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

    print("Fetching library list...")
    libs = db.list_libraries()
    print(f"Total libraries: {len(libs)}")

    keywords = ['news', 'text', 'senti', 'raven', 'reuters', 'thomson', 'ciq', 'capital', 'dj', 'dow']
    
    print("\n--- Potential Unstructured/News Libraries ---")
    found_libs = []
    for lib in libs:
        if any(k in lib.lower() for k in keywords):
            found_libs.append(lib)
            print(f"- {lib}")

    print("\n--- Inspecting Specific Tables in Key Libraries ---")
    # check for Key Developments in CIQ
    if 'ciq' in libs:
        try:
            tables = db.list_tables(library='ciq')
            keydev = [t for t in tables if 'keydev' in t or 'news' in t]
            print(f"Capital IQ Tables (partial): {keydev[:10]}")
        except:
            print("Could not access 'ciq'")

    # check for Ravenpack
    raven = [l for l in libs if 'raven' in l]
    if raven:
        print(f"RavenPack libraries found: {raven}")
    
    db.close()

if __name__ == "__main__":
    main()
