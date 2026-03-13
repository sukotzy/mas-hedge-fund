import pandas as pd

try:
    print("Loading comp_fundq.parquet directly (Raw WRDS data)...")
    df = pd.read_parquet('data/raw/comp_fundq.parquet')
    
    print(f"\nThere are {len(df)} total fundamental records.")
    print("Columns available:", list(df.columns))
    
    # Check if 'niq' and 'capxy' have any non-null values across the ENTIRE dataset
    niq_valid = df['niq'].notna().sum() if 'niq' in df.columns else 0
    capxy_valid = df['capxy'].notna().sum() if 'capxy' in df.columns else 0
    dpq_valid = df['dpq'].notna().sum() if 'dpq' in df.columns else 0
    
    print("\n=== SYSTEM-WIDE MISSING DATA CHECK ===")
    print(f"Total Rows: {len(df)}")
    print(f"Valid 'niq' (Net Income) records: {niq_valid} ({(niq_valid/len(df))*100:.2f}%)")
    print(f"Valid 'dpq' (Depreciation) records: {dpq_valid} ({(dpq_valid/len(df))*100:.2f}%)")
    print(f"Valid 'capxy' (Capital Ex) records: {capxy_valid} ({(capxy_valid/len(df))*100:.2f}%)")
    
    if niq_valid < len(df) * 0.1:
        print("\n🚨 WARNING: The dataset is severely missing core FCF components globally!")
        
    # Check specifically for MA (using tic if available, else gvkey string mapping)
    if 'tic' in df.columns:
        ma_df = df[df['tic'] == 'MA']
        print(f"\nFound {len(ma_df)} rows for tic='MA'")
    else:
        print("\n'tic' column missing. This dataset only uses gvkey.")
        
except Exception as e:
    import traceback
    traceback.print_exc()
    print('Failed:', e)
