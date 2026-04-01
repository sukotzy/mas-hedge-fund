import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

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
    parser = argparse.ArgumentParser(description="Plot N-Column Forward Alpha Comparison and Correlation")
    
    # We use action='append' to let the user specify multiple plot groups. 
    # Example: 
    # python plot.py --files A B --labels Al Bl --titles "Left" 
    #                --files C D --labels Cl Dl --titles "Right"
    parser.add_argument("--files", action='append', nargs='+', required=True, 
                        help="List of .jsonl files for one subplot. Can be passed multiple times for multiple subplots side-by-side.")
    parser.add_argument("--labels", action='append', nargs='+', required=True, 
                        help="List of labels matching the files. Must be passed equally as many times as --files.")
    parser.add_argument("--titles", action='append', nargs='+', 
                        help="Title(s) for the subplot. Must be passed equally as many times as --files.")
                        
    parser.add_argument("--output", type=str, default="alpha_comparison.png", help="Output PNG path")
    parser.add_argument("--benchmark", type=str, default="^GSPC", help="Yahoo Finance ticker for benchmark (default: ^GSPC)")
    parser.add_argument("--no-benchmark", action="store_true", help="Disable plotting the benchmark entirely")
    parser.add_argument("--suptitle", type=str, default="", help="Main title spanning the entire figure")
    parser.add_argument("--decouple", action="store_true", help="Split subplots into independent image files saved to thesis_plots/")
    
    args = parser.parse_args()
    
    # Override benchmark if disabled
    if args.no_benchmark:
        args.benchmark = None
    
    num_subplots = len(args.files)
    if len(args.labels) != num_subplots:
        logger.error(f"Error: The number of --files groups ({num_subplots}) must exactly match the number of --labels groups ({len(args.labels)}).")
        return
        
    titles = [t[0] for t in args.titles] if args.titles else [f"Subplot {i+1}" for i in range(num_subplots)]
    if len(titles) < num_subplots:
        titles.extend([f"Subplot {i+1}" for i in range(num_subplots - len(titles))])
        
    output_path = Path(args.output)
    if args.decouple:
        thesis_dir = Path("thesis_plots")
        thesis_dir.mkdir(exist_ok=True)
    else:
        fig, axes = plt.subplots(1, num_subplots, figsize=(max(8 * num_subplots, 10), 6), squeeze=False)
        axes = axes.flatten()
        if args.suptitle:
            fig.suptitle(args.suptitle, fontsize=18, fontweight='bold', y=1.05)
        
    # High-contrast color palette optimized for colorblind/B&W printing readability
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i in range(num_subplots):
        if args.decouple:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            ax = axes[i]
        
        files_group = args.files[i]
        labels_group = args.labels[i]
        title = titles[i]
        
        if len(files_group) != len(labels_group):
            logger.error(f"Group {i+1}: Output mismatch. files({len(files_group)}) != labels({len(labels_group)})")
            continue
            
        returns_dict = {}
        logger.info(f"\n" + "="*60)
        logger.info(f"   🔴 Processing Subplot {i+1} : {title}")
        logger.info("="*60)
        
        df_returns = pd.DataFrame()
        
        for idx, (file_str, label) in enumerate(zip(files_group, labels_group)):
            file_path = Path(file_str)
            if not file_path.exists():
                logger.warning(f"File not found, skipping: {file_path}")
                continue
                
            daily_returns = parse_jsonl_to_daily_returns(file_path)
            if daily_returns.empty:
                logger.warning(f"No return data found for {label}")
                continue
                
            returns_dict[label] = daily_returns
            cumulative_return = (1 + daily_returns).cumprod()
            
            # Use distinct robust colors from the palette
            c = colors[idx % len(colors)]
            ax.plot(cumulative_return.index, cumulative_return.values, label=label, linewidth=1.2, color=c, alpha=0.9)
            
        if not returns_dict:
            ax.set_title(f"{title} (No Data Found/Rendered)", fontsize=14, fontweight='bold', pad=15)
            continue
            
        df_returns = pd.DataFrame(returns_dict)
        
        # Benchmark Integration
        if args.benchmark:
            try:
                import yfinance as yf
                logger.info(f"Downloading {args.benchmark} baseline...")
                start_date = df_returns.index.min().strftime('%Y-%m-%d')
                end_date = (df_returns.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                bench_df = yf.download(args.benchmark, start=start_date, end=end_date, progress=False)
                
                if not bench_df.empty and 'Close' in bench_df.columns:
                    if isinstance(bench_df.columns, pd.MultiIndex):
                        bench_close = bench_df['Close'][args.benchmark]
                    else:
                        bench_close = bench_df['Close']
                        
                    bench_returns = bench_close.pct_change().dropna()
                    bench_aligned = bench_returns.reindex(df_returns.index).fillna(0)
                    df_returns['Benchmark (' + args.benchmark + ')'] = bench_aligned
                    
                    bench_cum = (1 + bench_aligned).cumprod()
                    ax.plot(bench_cum.index, bench_cum.values, label=f"Benchmark ({args.benchmark})", color='black', linewidth=1.5, linestyle='--')
            except Exception as e:
                logger.warning(f"Failed to fetch benchmark: {e}")
                
        # Formatting Axis and Grid
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
        ax.set_xlabel("Time Horizon", fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel("Cumulative Capital (Base 1.0)", fontsize=12, fontweight='bold', labelpad=10)
        
        # Y-Axis Formatting (comma separation, 2 decimal places precision)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        
        ax.grid(True, linestyle='--', color='#b0b0b0', alpha=0.5)
        # Background color tinting for aesthetics
        ax.set_facecolor('#fdfdfd')
        
        # Thick borders for academic graphs
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#333333')
            
        ax.tick_params(axis='both', which='major', labelsize=11, length=6, width=1.5, colors='#333333')
        
        # Elegant non-obstructive Legend
        ax.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.85, edgecolor='#333333', fontsize=11)
        
        # Output Correlation Matrix
        logger.info(f"\n   [Correlation Matrix: {title}]   ")
        corr_matrix = df_returns.corr()
        print(corr_matrix.round(3).to_string() + "\n")
        
        if args.decouple:
            plt.tight_layout()
            
            panel_letters = ['A', 'B', 'C', 'D', 'E', 'F']
            panel_suffix = f"_Panel{panel_letters[i]}" if num_subplots > 1 else ""
            
            base_name = output_path.stem + panel_suffix + output_path.suffix
            final_out = thesis_dir / base_name
            
            try:
                fig.savefig(final_out, dpi=400, bbox_inches='tight')
                logger.info(f"✅ Success! Plot elegantly saved out to: {final_out}")
            except Exception as e:
                logger.error(f"Failed writing image to disk: {e}")
                
            plt.close(fig)

    if not args.decouple:
        plt.tight_layout()
        try:
            fig.savefig(args.output, dpi=400, bbox_inches='tight')
            logger.info(f"✅ Success! Plot elegantly saved out to: {args.output}")
        except Exception as e:
            logger.error(f"Failed writing image to disk: {e}")

if __name__ == "__main__":
    main()
