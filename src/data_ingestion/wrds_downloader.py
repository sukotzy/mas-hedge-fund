'''Downloads S&P 500 Prices (OHLCV), Constituents, and Deep Fundamentals (comp.fundq) + Ratios.'''

import wrds
import pandas as pd
import os
from datetime import datetime
import argparse
import warnings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress warnings
warnings.filterwarnings('ignore')

def connect_to_wrds():
    """Connect to WRDS."""
    try:
        print("Connecting to WRDS...")
        username = os.getenv("WRDS_USERNAME")
        if username:
            db = wrds.Connection(wrds_username=username)
        else:
            db = wrds.Connection()
        return db
    except Exception as e:
        print(f"Error connecting to WRDS: {e}")
        return None

def get_sp500_constituents(db, start_date='2015-01-01', end_date='2024-12-31'):
    """
    Get S&P 500 historical constituents.
    """
    print("Fetching S&P 500 constituents with Tickers and Names...")
    # crsp.msp500list only has permnos. We need to join with crsp.stocknames to get tickers/names.
    query = f"""
        SELECT a.*, b.ticker, b.comnam, b.ncusip
        FROM crsp.msp500list AS a
        JOIN crsp.stocknames AS b
        ON a.permno = b.permno
        WHERE a.ending >= '{start_date}' 
        AND a.start <= '{end_date}'
        AND b.namedt <= a.ending
        AND b.nameenddt >= a.start
    """
    try:
        sp500 = db.raw_sql(query)
        if not sp500.empty:
            print(f"Columns found: {sp500.columns.tolist()}")
            # Determine correct sort column (likely 'nameenddt' or similar)
            sort_col = 'nameenddt' if 'nameenddt' in sp500.columns else 'ending'
            sp500 = sp500.sort_values(sort_col, ascending=False).drop_duplicates(subset=['permno', 'start', 'ending'])
        print(f"Got {len(sp500)} constituent records with names.")
        return sp500
    except Exception as e:
        print(f"Error fetching constituents: {e}")
        return pd.DataFrame()

def get_ohlcv_data(db, permnos, start_date='2015-01-01', end_date='2024-12-31'):
    """
    Get OHLCV data for the list of PERMNOs from CRSP Daily Stock File (dsf).
    """
    print(f"Fetching OHLCV data for {len(permnos)} stocks...")
    
    if not permnos:
        return pd.DataFrame()

    # Split permnos into chunks to avoid too huge query
    chunk_size = 500
    all_data = []
    
    for i in range(0, len(permnos), chunk_size):
        chunk = permnos[i:i + chunk_size]
        print(f"Fetching OHLCV chunk {i//chunk_size + 1} of {(len(permnos)-1)//chunk_size + 1}...")
        
        permno_tuple = tuple(chunk)
        if len(permno_tuple) == 1:
            permno_tuple = f"({permno_tuple[0]})"
        
        query = f"""
            SELECT date, permno, prc, vol, openprc, askhi, bidlo
            FROM crsp.dsf
            WHERE date >= '{start_date}'
            AND date <= '{end_date}'
            AND permno IN {permno_tuple}
        """
        try:
            data = db.raw_sql(query)
            if not data.empty:
                all_data.append(data)
                print(f"  Got {len(data)} rows.")
            else:
                print("  Got 0 rows.")
        except Exception as e:
            print(f"Error fetching OHLCV for chunk: {e}")
            print(f"Query was: {query[:200]}...")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

def get_financial_ratios_firm_ratio(db, permnos, start_date='2015-01-01', end_date='2024-12-31'):
    """
    Try fetching from wrdsapps.firm_ratio
    """
    print("Attempting to fetch from wrdsapps.firm_ratio...")
    if not permnos:
        return pd.DataFrame()
    
    permno_tuple = tuple(permnos)
    if len(permno_tuple) == 1:
        permno_tuple = f"({permno_tuple[0]})"
        
    query = f"""
        SELECT public_date as date, permno, ticker, pe_exi, pe_inc, ptb
        FROM wrdsapps.firm_ratio
        WHERE public_date >= '{start_date}'
        AND public_date <= '{end_date}'
        AND permno IN {permno_tuple}
    """
    try:
        data = db.raw_sql(query)
        print(f"Got {len(data)} rows from firm_ratio.")
        return data
    except Exception as e:
        print(f"Error fetching from firm_ratio: {e}")
        return pd.DataFrame()

def get_financial_data_manual(db, permnos, start_date='2015-01-01', end_date='2024-12-31'):
    """
    Get Financial Data manually by linking CRSP permnos to Compustat GVKEYs
    and fetching fundamental data from comp.fundq.
    """
    print("Fetching Linking Table (CCM)...")
    
    if not permnos:
        return pd.DataFrame(), pd.DataFrame()
        
    permno_tuple = tuple(permnos)
    if len(permno_tuple) == 1:
        permno_tuple = f"({permno_tuple[0]})"
    
    # 1. Get Link Table
    link_query = f"""
        SELECT gvkey, lpermno as permno, linkdt, linkenddt
        FROM crsp.ccmxpf_linktable
        WHERE lpermno IN {permno_tuple}
        AND linktype IN ('LC', 'LU', 'LS')
        AND linkprim IN ('P', 'C')
    """
    try:
        links = db.raw_sql(link_query)
        print(f"Found {len(links)} links.")
    except Exception as e:
        print(f"Error fetching link table: {e}")
        return pd.DataFrame(), pd.DataFrame()
        
    if links.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    # Ensure GVKEYs are 6-digit strings
    gvkeys = [str(g).zfill(6) for g in links['gvkey'].unique().tolist() if g]
    print(f"Mapped to {len(gvkeys)} unique GVKEYs.")
    
    # 2. Fetch Fundamentals from comp.fundq
    chunk_size = 500
    fund_data_list = []
    
    # Columns to fetch
    # rdq: Release Date Quarterly (Critical for point-in-time)
    # datadate: Fiscal period end
    # niq: Net Income (Quarterly)
    # revtq: Revenue (Quarterly)
    # oiadpq: Operating Income After Depreciation (Quarterly)
    # atq: Total Assets
    # ltq: Total Liabilities
    # actq: Current Assets
    # lctq: Current Liabilities
    # dlttq: Long Term Debt Total
    # dlcq: Debt in Current Liabilities
    # seqq: Stockholders Equity - Total
    # cshoq: Common Shares Outstanding
    # epspxq: EPS (Basic) - Excluding Extraordinary Items
    
    columns = [
        "gvkey", "datadate", "rdq", "fyearq", "fqtr",
        "niq", "revtq", "oiadpq", 
        "atq", "ltq", "actq", "lctq", 
        "dlttq", "dlcq", "seqq", "cshoq", 
        "epspxq"
    ]
    cols_str = ", ".join(columns)

    for i in range(0, len(gvkeys), chunk_size):
        chunk = gvkeys[i:i + chunk_size]
        # Format as string logic for SQL IN clause
        gvkey_list_str = ",".join([f"'{g}'" for g in chunk])
        gvkey_tuple = f"({gvkey_list_str})"

        print(f"Fetching Fundamentals chunk {i//chunk_size + 1}...")
        
        fund_query = f"""
            SELECT {cols_str}
            FROM comp.fundq
            WHERE datadate >= '{start_date}'
            AND datadate <= '{end_date}'
            AND gvkey IN {gvkey_tuple}
        """
        try:
            data = db.raw_sql(fund_query)
            if not data.empty:
                fund_data_list.append(data)
                print(f"  Got {len(data)} rows.")
            else:
                print("  Got 0 rows.")
        except Exception as e:
             print(f"Error fetching fundamentals: {e}")
             print(f"Query was: {fund_query[:200]}...")

    fund_data = pd.concat(fund_data_list, ignore_index=True) if fund_data_list else pd.DataFrame()
    
    # Post-processing: Calculate Derived Metrics
    if not fund_data.empty:
        print("Calculating derived metrics...")
        # Fill missing values with 0 for calculation safety only where appropriate, 
        # but for ratios, it's better to leave as NaN if denominator is missing.
        
        # 1. ROE = Net Income / Total Equity
        fund_data['return_on_equity'] = fund_data['niq'] / fund_data['seqq']
        
        # 2. Net Margin = Net Income / Revenue
        fund_data['net_margin'] = fund_data['niq'] / fund_data['revtq']
        
        # 3. Operating Margin = Operating Income / Revenue
        fund_data['operating_margin'] = fund_data['oiadpq'] / fund_data['revtq']
        
        # 4. Current Ratio = Current Assets / Current Liabilities
        fund_data['current_ratio'] = fund_data['actq'] / fund_data['lctq']
        
        # 5. Debt to Equity = (Long Term Debt + Debt in Current Liab) / Total Equity
        # Handle cases where debt components might be NaN (assume 0 if missing? usually safe for debt but let's be careful)
        total_debt = fund_data['dlttq'].fillna(0) + fund_data['dlcq'].fillna(0)
        fund_data['debt_to_equity'] = total_debt / fund_data['seqq']
        
        # Clean up infinite values
        import numpy as np
        fund_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
    return links, fund_data

def save_data(df, filename):
    """Save DataFrame to Parquet."""
    if df.empty:
        print(f"No data to save for {filename}")
        return
    
    output_dir = 'data/raw'
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    try:
        df.to_parquet(path)
        print(f"Saved {len(df)} rows to {path}")
    except Exception as e:
        print(f"Error saving {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download S&P 500 data from WRDS")
    parser.add_argument("--start-date", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2024-12-31", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    db = connect_to_wrds()
    if not db:
        return

    # 1. Get Constituents
    sp500_hist = get_sp500_constituents(db, args.start_date, args.end_date)
    save_data(sp500_hist, 'sp500_constituents.parquet')
    
    if sp500_hist.empty:
        print("No constituents found.")
        db.close()
        return

    unique_permnos = [int(p) for p in sp500_hist['permno'].unique().tolist()]
    print(f"Found {len(unique_permnos)} unique PERMNOs in S&P 500 history.")

    # 2. Get OHLCV
    ohlcv = get_ohlcv_data(db, unique_permnos, args.start_date, args.end_date)
    save_data(ohlcv, 'sp500_ohlcv.parquet')
    
    # 3. Get Financial Ratios (PE/PB)
    ratios = get_financial_ratios_firm_ratio(db, unique_permnos, args.start_date, args.end_date)
    if not ratios.empty:
        save_data(ratios, 'sp500_ratios_firm_ratio.parquet')
    
    # 4. Get Deep Fundamentals (comp.fundq)
    # Always fetch this for deep metrics (ROE, Margins, etc.)
    print("Fetching Deep Fundamentals (comp.fundq)...")
    links, fund_data = get_financial_data_manual(db, unique_permnos, args.start_date, args.end_date)
    save_data(links, 'ccm_links.parquet')
    save_data(fund_data, 'comp_fundq.parquet')
    
    db.close()

if __name__ == "__main__":
    main()
