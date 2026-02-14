'''Downloads Capital IQ Key Developments (ciq.wrds_keydev) which feeds the News Agent.'''

import wrds
import pandas as pd
import os
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# Setup paths
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"

def get_connection():
    load_dotenv()
    try:
        db = wrds.Connection(wrds_username=os.getenv("WRDS_USERNAME"))
        return db
    except Exception as e:
        print(f"Error connecting to WRDS: {e}")
        return None

def fetch_s500_permnos(db, start_date='2015-01-01', end_date='2024-12-31'):
    """Fetch S&P 500 PERMNOs and GVKEYs for filtering."""
    print("Fetching S&P 500 identifiers...")
    query = f"""
        SELECT DISTINCT a.permno, b.ticker, c.gvkey
        FROM crsp.msp500list AS a
        JOIN crsp.stocknames AS b
            ON a.permno = b.permno
        LEFT JOIN crsp.ccmxpf_linktable AS c
            ON a.permno = c.lpermno
        WHERE a.ending >= '{start_date}' 
        AND a.start <= '{end_date}'
        AND b.namedt <= a.ending
        AND b.nameenddt >= a.start
        AND c.linktype IN ('LU', 'LC') 
        AND c.linkprim IN ('P', 'C')
        AND c.usedflag = 1
    """
    df = db.raw_sql(query)
    # Deduplicate
    df = df.drop_duplicates(subset=['permno', 'gvkey'])
    print(f"Found {len(df)} unique identifiers.")
    return df

def download_keydev(db, gvkeys: list, start_date='2015-01-01', end_date='2024-12-31'):
    """
    Download Capital IQ Key Developments for specified GVKEYs.
    Filters by date and S&P 500 membership.
    """
    print("Downloading Capital IQ Key Developments via GVKEY...")
    
    if not gvkeys:
        print("No GVKEYs provided.")
        return pd.DataFrame()

    # Chunking GVKEYs (1000 at a time) to avoid SQL limits
    chunk_size = 1000
    all_keydev = []
    
    unique_gvkeys = [str(g) for g in gvkeys if pd.notna(g)]
    
    for i in range(0, len(unique_gvkeys), chunk_size):
        chunk = unique_gvkeys[i:i + chunk_size]
        gvkey_tuple = tuple(chunk)
        if len(tuple(chunk)) == 1:
             gvkey_tuple = f"('{chunk[0]}')"
        
        # Query for KeyDev
        # We want: Headline, Situation (Text), Event Type, Anomalies?
        query = f"""
            SELECT 
                k.keydevid, k.companyid, k.headline, k.situation, 
                k.keydeveventtypeid, k.announcedate, k.gvkey
            FROM ciq.wrds_keydev AS k
            WHERE k.gvkey IN {gvkey_tuple}
            AND k.announcedate >= '{start_date}'
            AND k.announcedate <= '{end_date}'
        """
        try:
            df = db.raw_sql(query)
            if not df.empty:
                all_keydev.append(df)
            print(f"Fetched {len(df)} rows for chunk {i//chunk_size + 1}")
        except Exception as e:
            print(f"Error fetching KeyDev chunk {i}: {e}")

    if all_keydev:
        final_df = pd.concat(all_keydev)
        # Type conversion
        final_df['announcedate'] = pd.to_datetime(final_df['announcedate'])
        final_df['keydeveventtypeid'] = pd.to_numeric(final_df['keydeveventtypeid'], downcast='integer')
        return final_df.drop_duplicates(subset=['keydevid'])
    
    return pd.DataFrame()

def download_insider(db, tickers: list, start_date='2015-01-01', end_date='2024-12-31'):
    """
    Download Thomson Reuters Insider Filings (tfn.s12 / tfn.table1/2)
    Using Table 1 (Stock Transactions)
    """
    print("Downloading Insider Trading Data via Tickers...")
    # TFN uses 'ticker' usually.
    
    if not tickers:
        return pd.DataFrame()
        
    chunk_size = 500
    all_insider = []
    unique_tickers = [str(t) for t in tickers if pd.notna(t)]

    # Clean tickers (remove duplicates)
    unique_tickers = list(set(unique_tickers))

    for i in range(0, len(unique_tickers), chunk_size):
        chunk = unique_tickers[i:i + chunk_size]
        ticker_tuple = tuple(chunk)
        if len(chunk) == 1:
            ticker_tuple = f"('{chunk[0]}')"
            
        # Common TFN Tables:
        # tfn.table1 -> Non-Derivative Transactions (Stock buy/sell)
        # tfn.table2 -> Derivative Transactions (Options)
        # We focus on Table 1 for open market buys/sells.
        
        # Debug: check columns first if this is the first chunk
        if i == 0:
            try:
                print("Checking TFN columns...")
                print(db.raw_sql("SELECT * FROM tfn.table1 LIMIT 1").columns.tolist())
                print(db.raw_sql("SELECT * FROM tfn.header LIMIT 1").columns.tolist())
            except Exception as e:
                print(f"Error checking columns: {e}")

        # Simplified query matching standard TFN schema
        # We need ticker, date, shares, price, transaction code. 
        # Note: 'acquisition_disposition' might be named differently (e.g., 'acqdisp')
        # We'll select fewer columns to be safe.
        query = f"""
            SELECT 
                a.ticker, a.date, a.shares, a.price, a.transaction_code, a.acqdisp,
                h.form_type, h.filedate
            FROM tfn.table1 AS a
            JOIN tfn.header AS h ON a.dcn = h.dcn
            WHERE a.ticker IN {ticker_tuple}
            AND a.date >= '{start_date}'
            AND a.date <= '{end_date}'
            AND h.form_type = '4' 
        """
        # Form 4 is the standard insider trade form
        
        try:
            df = db.raw_sql(query)
            if not df.empty:
                all_insider.append(df)
            print(f"Fetched {len(df)} insider rows for chunk {i//chunk_size + 1}")
        except Exception as e:
            print(f"Error fetching Insider chunk {i}: {e}")
            
    if all_insider:
        final_df = pd.concat(all_insider)
        final_df['date'] = pd.to_datetime(final_df['date'])
        final_df['filedate'] = pd.to_datetime(final_df['filedate'])
        return final_df
    
    return pd.DataFrame()

def main():
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)

    db = get_connection()
    if not db:
        return

    # 1. Get List
    ids = fetch_s500_permnos(db)
    if ids.empty:
        print("Failed to fetch identifiers.")
        db.close()
        return

    gvkeys = ids['gvkey'].unique().tolist()
    tickers = ids['ticker'].unique().tolist()

    # 2. KeyDev
    keydev_df = download_keydev(db, gvkeys)
    if not keydev_df.empty:
        save_path = RAW_DIR / "sp500_keydev.parquet"
        keydev_df.to_parquet(save_path)
        print(f"Saved {len(keydev_df)} KeyDev rows to {save_path}")
    else:
        print("No KeyDev data found.")

    # 3. Insider
    insider_df = download_insider(db, tickers)
    if not insider_df.empty:
        save_path = RAW_DIR / "sp500_insiders.parquet"
        insider_df.to_parquet(save_path)
        print(f"Saved {len(insider_df)} Insider rows to {save_path}")
    else:
        print("No Insider data found.")

    db.close()

if __name__ == "__main__":
    main()
