import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def parse_jsonl_to_daily_returns(file_path: Path) -> pd.Series:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if "date" in row and "portfolio_value" in row:
                    data.append({
                        "date": row["date"],
                        "portfolio_value": float(row["portfolio_value"])
                    })
            except Exception:
                pass
                
    if not data:
        return pd.Series(dtype=float)
        
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    
    # Calculate daily return from portfolio value
    daily_returns = df['portfolio_value'].pct_change().dropna()
    return daily_returns

def main():
    parser = argparse.ArgumentParser(description="Plot 1-Day Forward Alpha Comparison and Correlation")
    parser.add_argument("--files", nargs='+', required=True, help="List of .jsonl files to process")
    parser.add_argument("--labels", nargs='+', required=True, help="List of labels matching the files")
    parser.add_argument("--output", type=str, default="alpha_comparison.png", help="Output PNG path")
    
    args = parser.parse_args()
    
    if len(args.files) != len(args.labels):
        logger.error("Error: The number of --files must exactly match the number of --labels.")
        return
        
    returns_dict = {}
    
    plt.figure(figsize=(14, 8))
    
    # Process each file and plot
    for file_str, label in zip(args.files, args.labels):
        file_path = Path(file_str)
        if not file_path.exists():
            logger.warning(f"File not found, skipping: {file_path}")
            continue
            
        logger.info(f"Processing {label}...")
        daily_returns = parse_jsonl_to_daily_returns(file_path)
        
        if daily_returns.empty:
            logger.warning(f"No return data found for {label}")
            continue
            
        # Store for the correlation matrix later
        returns_dict[label] = daily_returns
        
        # Calculate cumulative pure alpha
        cumulative_return = (1 + daily_returns).cumprod()
        
        # Plot the curve
        plt.plot(cumulative_return.index, cumulative_return.values, label=label, linewidth=1.5)

    if not returns_dict:
        logger.error("No valid data to plot.")
        return

    # Finalize plot formatting
    plt.title("Pure 1-Day Forward Alpha Cumulative Returns\n(Zero Friction / 100% Daily Turnover)", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return (Base 1.0)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(args.output, dpi=300)
    logger.info(f"\n✅ Plot successfully saved to: {args.output}")
    
    # Correlation Matrix Calculation
    logger.info("\n" + "="*50)
    logger.info("   DAILY RETURN CORRELATION MATRIX   ")
    logger.info("="*50)
    
    # Combine all series into a single DataFrame on Date index
    df_returns = pd.DataFrame(returns_dict)
    
    # Compute correlation
    corr_matrix = df_returns.corr()
    
    # Print elegantly
    print(corr_matrix.round(3).to_string())
    logger.info("="*50 + "\n")

if __name__ == "__main__":
    main()
