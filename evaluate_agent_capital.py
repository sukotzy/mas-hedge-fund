import re
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def parse_log_for_capital(logfile):
    # Regex to match: 2026-03-15 23:14:47,260 - INFO - [2022-08-10] Allocator FUNDAMENTAL Capital: $99,606.28
    pattern = re.compile(r"\[(\d{4}-\d{2}-\d{2})\].*?Allocator\s+([A-Z]+)\s+Capital:\s+\$([\d,]+\.\d{2})")
    
    data = []
    with open(logfile, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                date_str = match.group(1)
                agent_name = match.group(2)
                # Remove commas and convert to float
                capital = float(match.group(3).replace(',', ''))
                data.append({
                    "Date": date_str,
                    "Agent": agent_name,
                    "Capital": capital
                })
                
    if not data:
        print(f"No capital data found in {logfile}")
        return pd.DataFrame()
        
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Pivot so each column is an agent and each row is a date
    df_pivot = df.pivot_table(index='Date', columns='Agent', values='Capital', aggfunc='last')
    
    # Forward fill missing dates
    df_pivot.ffill(inplace=True)
    df_pivot.dropna(inplace=True)
    return df_pivot

def main():
    parser = argparse.ArgumentParser(description="Plot Agent Capital Trajectories from Logs")
    parser.add_argument("--log", required=True, help="Path to the backtest log file (e.g., nohup.out or log.txt)")
    parser.add_argument("--output", default="agent_capital_trajectory.png", help="Output image file")
    args = parser.parse_args()
    
    print(f"Parsing log file: {args.log}")
    df = parse_log_for_capital(args.log)
    
    if df.empty:
        return
        
    print(f"Successfully extracted {len(df)} days of capital data.")
    
    plt.style.use('bmh')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = {
        'FUNDAMENTAL': '#1f77b4',  # blue
        'TECHNICAL': '#ff7f0e',    # orange
        'VALUATION': '#2ca02c',    # green
        'SENTIMENT': '#d62728'     # red
    }
    
    for col in df.columns:
        c = colors.get(col, None)
        ax.plot(df.index, df[col], label=col, color=c, linewidth=2)
        
    ax.set_title('Meta-Manager Allocation: Agent Capital Trajectory (Zero-Sum Settlement)', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('Total Capital ($)', fontsize=12)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add a horizontal line at 100,000 to show the starting point
    ax.axhline(y=100000, color='gray', linestyle=':', alpha=0.8, label='Initial Capital ($100k)')
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
