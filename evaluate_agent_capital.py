import re
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def parse_log_for_capital_and_roi(logfile):
    # Regex to match the new format with or without ROI:
    # Format 1 (New): [2018-05-23] Allocator FUNDAMENTAL Capital: $104,555.13 (External... [Daily ROI: 0.1234%]
    # Format 2 (Old): [2018-05-23] Allocator FUNDAMENTAL Capital: $104,555.13 (External...
    
    pattern = re.compile(r"\[(\d{4}-\d{2}-\d{2})\].*?Allocator\s+([A-Z]+)\s+Capital:\s+\$([\d,]+\.\d{2})(?:.*?(?:\[Daily ROI:\s*([-\d\.]+)\%\]))?")
    
    data = []
    with open(logfile, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                date_str = match.group(1)
                agent_name = match.group(2)
                capital = float(match.group(3).replace(',', ''))
                
                # Check if ROI was captured
                roi_str = match.group(4)
                roi = float(roi_str) / 100.0 if roi_str else np.nan
                
                data.append({
                    "Date": date_str,
                    "Agent": agent_name,
                    "Capital": capital,
                    "ROI": roi
                })
                
    if not data:
        print(f"No capital data found in {logfile}")
        return pd.DataFrame(), pd.DataFrame()
        
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Pivot Capital
    df_cap = df.pivot_table(index='Date', columns='Agent', values='Capital', aggfunc='last')
    df_cap.ffill(inplace=True)
    df_cap.dropna(inplace=True)
    
    # Pivot ROI
    df_roi = df.pivot_table(index='Date', columns='Agent', values='ROI', aggfunc='last')
    
    return df_cap, df_roi

def main():
    parser = argparse.ArgumentParser(description="Plot Agent Capital Trajectories and ROI from Logs")
    parser.add_argument("--log", required=True, help="Path to the backtest log file (e.g., nohup.out or log.txt)")
    parser.add_argument("--output", default="agent_capital_trajectory.png", help="Output image file")
    parser.add_argument("--window", type=int, default=63, help="Rolling window size in days for ROI smoothing (default: 63 days / 1 quarter)")
    args = parser.parse_args()
    
    print(f"Parsing log file: {args.log}")
    df_cap, df_roi = parse_log_for_capital_and_roi(args.log)
    
    if df_cap.empty:
        return
        
    print(f"Successfully extracted {len(df_cap)} days of capital and ROI data.")
    
    plt.style.use('bmh')
    
    # Create a 2-panel plot if ROI data exists, otherwise just 1 panel
    has_roi = not df_roi.isna().all().all()
    
    if has_roi:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(16, 8))
        print("Note: No Daily ROI data found in logs. (Are you using an older log file?)")
    
    colors = {
        'FUNDAMENTAL': '#1f77b4',  # blue
        'TECHNICAL': '#ff7f0e',    # orange
        'VALUATION': '#2ca02c',    # green
        'SENTIMENT': '#d62728'     # red
    }
    
    # --- PANEL 1: Capital Trajectories ---
    for col in df_cap.columns:
        c = colors.get(col, None)
        ax1.plot(df_cap.index, df_cap[col], label=col, color=c, linewidth=2)
        
    ax1.set_title('Meta-Manager Allocation: Agent Capital Trajectory (Zero-Sum Settlement)', fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel('Total Capital ($)', fontsize=12)
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Add a horizontal line at Initial Capital
    initial_cap = df_cap.iloc[0].mean() # usually 100k or 50k
    ax1.axhline(y=initial_cap, color='gray', linestyle=':', alpha=0.8, label=f'Initial Capital (${initial_cap:,.0f})')
    
    # --- PANEL 2: Smoothed ROI (if available) ---
    if has_roi:
        # Calculate Rolling Average ROI (annualized for readability)
        rolling_roi = df_roi.rolling(window=args.window, min_periods=1).mean() * 252
        
        for col in rolling_roi.columns:
            c = colors.get(col, None)
            ax2.plot(rolling_roi.index, rolling_roi[col], label=col, color=c, linewidth=1.5, alpha=0.8)
            
        ax2.set_title(f'Agent Performance: {args.window}-Day Rolling Annualized ROI', fontsize=14, fontweight='bold', pad=10)
        ax2.set_ylabel('Annualized Return', fontsize=12)
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)) # Format as percentage
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # We don't need a legend twice if colors match, but it's good for clarity
        ax2.legend(loc='upper left', fontsize=10, framealpha=0.9, ncol=4)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
