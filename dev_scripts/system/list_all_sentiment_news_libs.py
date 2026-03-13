import wrds
import sys
import os
from dotenv import load_dotenv

def list_all_news_sentiment():
    try:
        load_dotenv()
        username = os.getenv("WRDS_USERNAME")
        
        print(f"Connecting to WRDS as {username}...")
        # Use simple connection relying on pgpass, as confirmed working
        db = wrds.Connection(wrds_username=username)
        print("Connected.")
        
        print("Fetching full library list...")
        libs = db.list_libraries()
        libs.sort()
        
        keywords = [
            'news', 'sentiment', 'text', 'raven', 'thomson', 'reuters', 
            'capital', 'media', 'social', 'twitter', 'stocktwits', 
            'bitly', 'google', 'search', 'trend', 'word', 'bag', 'linguistic'
        ]
        
        print(f"Scanning {len(libs)} libraries for keywords: {keywords}")
        
        matches = []
        for lib in libs:
            if any(k in lib.lower() for k in keywords):
                matches.append(lib)
                
        if matches:
            print(f"\nFound {len(matches)} potential libraries:")
            for lib in matches:
                print(f"\n--- Library: {lib} ---")
                try:
                    tables = db.list_tables(lib)
                    print(f"Tables ({len(tables)}): {tables}")
                except Exception as e:
                    print(f"Error listing tables: {e}")
        else:
            print("\nNo libraries found matching the keywords.")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    list_all_news_sentiment()
