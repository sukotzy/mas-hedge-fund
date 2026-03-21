import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def analyze_candidate_diversity(
    candidates_file="data/processed/daily_candidates_with_hint.parquet",
    ohlcv_file="data/raw/sp500_ohlcv.parquet",
    lookback_days=252,
    start_date=None,
    end_date=None
):
    """
    Checks if the selected candidates from Hierarchical Clustering are actually diverse
    by calculating the pairwise correlation of their returns prior to selection date.
    """
    if not Path(candidates_file).exists():
        logger.error(f"Cannot find {candidates_file}. Please run run_selection.py first.")
        return
        
    logger.info("Loading necessary data...")
    # Load candidates
    cand_df = pd.read_parquet(candidates_file)
    if cand_df.empty:
        logger.error("Candidate file is empty.")
        return
        
    # Load price data to calculate correlation
    logger.info(f"Loading prices from {ohlcv_file}...")
    ohlcv = pd.read_parquet(ohlcv_file)
    
    # WRDS OHLCV uses 'permno', need to map to 'ticker'
    constituents_file = "data/raw/sp500_constituents.parquet"
    if Path(constituents_file).exists():
        constituents = pd.read_parquet(constituents_file)
        permno_map = constituents[['permno', 'ticker']].drop_duplicates(subset='permno', keep='last')
        ohlcv = ohlcv.merge(permno_map, on='permno', how='left')
        ohlcv = ohlcv.dropna(subset=['ticker'])
        # Deduplicate multiple permnos mapping to same ticker on same date by keeping highest volume
        ohlcv = ohlcv.sort_values('vol', ascending=False).drop_duplicates(subset=['date', 'ticker'])
    else:
        logger.error(f"Cannot find {constituents_file}, needed to map permno to ticker.")
        return
        
    ohlcv['date'] = pd.to_datetime(ohlcv['date'])
    prices = ohlcv.pivot(index='date', columns='ticker', values='prc').ffill(limit=3)
    returns = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan)
    
    # Filter dates based on arguments
    if start_date:
        cand_df = cand_df[cand_df.index >= pd.to_datetime(start_date)]
    if end_date:
        cand_df = cand_df[cand_df.index <= pd.to_datetime(end_date)]
        
    sample_dates = cand_df.index
    
    logger.info(f"\n{'='*60}")
    logger.info(f"CANDIDATE DIVERSITY ANALYSIS (Ward's Hierarchical Clustering)")
    logger.info(f"{'='*60}")
    
    all_avg_corrs = []
    
    for date in sample_dates:
        date_str = date.strftime('%Y-%m-%d')
        tasks_json = cand_df.loc[date, 'tasks']
        tasks = json.loads(tasks_json)
        
        tickers = [t['ticker'] for t in tasks]
        
        # Get historical slice
        end_idx = returns.index.get_indexer([date], method='pad')[0]
        start_idx = max(0, end_idx - lookback_days + 1)
        
        # Slice the returns for the lookback window
        window_returns = returns.iloc[start_idx:end_idx + 1]
        
        # Filter for the selected tickers available in the window
        valid_tickers = [t for t in tickers if t in window_returns.columns]
        if len(valid_tickers) < 2:
            logger.warning(f"  {date_str}: Not enough return history for selected tickers.")
            continue
            
        cand_returns = window_returns[valid_tickers].dropna(how='all')
        
        # Calculate Correlation Matrix
        corr_matrix = cand_returns.corr().values
        
        # Extract upper triangle (excluding diagonal which is 1.0)
        upper_triangle_indices = np.triu_indices_from(corr_matrix, k=1)
        pairwise_corrs = corr_matrix[upper_triangle_indices]
        
        # Clean NaNs if any stock had flat prices
        pairwise_corrs = pairwise_corrs[~np.isnan(pairwise_corrs)]
        
        if len(pairwise_corrs) == 0:
            continue
            
        avg_corr = np.mean(pairwise_corrs)
        max_corr = np.max(pairwise_corrs)
        min_corr = np.min(pairwise_corrs)
        all_avg_corrs.append(avg_corr)
        
        logger.info(f"Date: {date_str} | Selected: {', '.join(valid_tickers)}")
        logger.info(f"  - Avg Pairwise Correlation: {avg_corr:.3f}")
        logger.info(f"  - Min/Max Correlation:      [{min_corr:.3f}, {max_corr:.3f}]")
        logger.info("-" * 40)
        
    overall_avg = np.mean(all_avg_corrs) if all_avg_corrs else 0
    logger.info(f"OVERALL AVERAGE CORRELATION of Clusters: {overall_avg:.3f}")
    
    if overall_avg < 0.4:
        logger.info("VERDICT: SUCCESS. 候选股票之间的相关性较低（通常<0.4），说明分层聚类成功选出了不同类型的股票，实现了分散投资组合风险的目的。")
    else:
        logger.warning("VERDICT: WARNING. 候选股票之间的相关性较高，聚类的区分度可能不足。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify candidate diversity through correlation.")
    parser.add_argument("--start_date", type=str, default=None, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end_date", type=str, default=None, help="End date in YYYY-MM-DD format")
    args = parser.parse_args()
    
    analyze_candidate_diversity(start_date=args.start_date, end_date=args.end_date)
