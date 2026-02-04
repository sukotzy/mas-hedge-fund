import pandas as pd
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
from src.selection.data import SelectionDataLoader
# Updated Imports
from src.selection.layer1_shared import TopologyFilter, get_combined_candidate_pool
from src.selection.layer2 import CandidateSelector
from src.selection.realtime.layer1_detectors import MarketRegimeDetector

# Setup Logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def batch_selection(
    processed_dir: str = "data/processed",
    output_path: str = "data/processed/daily_candidates.parquet"
):
    """
    Generate Top Candidates for every day in history using pre-computed factors.
    Output: Parquet file with JSON tasks column.
    """
    p_dir = Path(processed_dir)
    factors_path = p_dir / "stock_factors.parquet"
    regime_path = p_dir / "market_regime.parquet"
    
    if not factors_path.exists() or not regime_path.exists():
        logger.error("Pre-computed factors not found. Run factor_db.py first.")
        return

    logger.info("Loading Pre-computed Factors...")
    factors_df = pd.read_parquet(factors_path)
    regime_df = pd.read_parquet(regime_path)
    
    # Load prices for clustering (we need 30d correlation)
    # This is heavy. Maybe we load incrementally?
    # For speed, let's load full OHLCV via loader and slice in loop
    # or just read factor_df dates and iterate.
    # Loading full prices is fine given prior experience (1GB mem).
    logger.info("Loading Price History for Clustering...")
    loader = SelectionDataLoader()
    # Fetch full history (valid end_date to ensure universe coverage)
    prices_full, _, _ = loader.fetch_universe_data(end_date="2024-12-31", lookback_days=5000)
    # Calculate returns for distance matrix
    returns_full = np.log(prices_full / prices_full.shift(1))
    
    # Get unique dates from factors
    valid_dates = factors_df['date'].unique()
    valid_dates = np.sort(valid_dates)
    
    logger.info(f"Processing {len(valid_dates)} days...")
    
    results = []
    
    # Detectors/Selectors
    topo_filter = TopologyFilter()
    selector = CandidateSelector()
    regime_detector = MarketRegimeDetector()
    
    for date in tqdm(valid_dates):
        ts = pd.Timestamp(date)
        date_str = ts.strftime('%Y-%m-%d')
        
        # 0. Get Daily Factors
        daily_factors = factors_df[factors_df['date'] == date].set_index('ticker')
        if daily_factors.empty: 
            logger.debug(f"Skipping {date_str}: No factors found.")
            continue
        
        # Get Regime
        regime = "Normal_Expansion"
        if ts in regime_df.index:
            regime = regime_df.loc[ts, 'regime']
            
        # 1. The "30+30" Funnel
        # Top 30 Centrality (Hubs + Leaves derived from Degree)
        # Note: 'degree' col exists.
        # TopologyFilter.get_topology_candidates expects degrees array
        degrees = daily_factors['degree'].values
        tickers = daily_factors.index.tolist()
        
        topo_candidates = topo_filter.get_topology_candidates(degrees, tickers)
        
        # Top 30 Isolation Forest Anomalies
        anom_scores = daily_factors['anomaly_score']
        # get_combined_candidate_pool takes Series
        pool = get_combined_candidate_pool(topo_candidates, anom_scores)
        
        if not pool: 
            logger.debug(f"Skipping {date_str}: Empty pool after 30+30 filter. Topo: {len(topo_candidates)}")
            continue
        
        # Ensure pool tickers exist in price data
        pool = [t for t in pool if t in prices_full.columns]
        if len(pool) < 2: 
            logger.debug(f"Skipping {date_str}: Pool < 2 after price check.")
            continue
        
        # 2. Dynamic Clustering
        # Need distance matrix for pool tickers over [t-30 : t]
        # Find index of date in returns_full
        if ts not in returns_full.index:
            logger.debug(f"Skipping {date_str}: Date not in price history.")
            continue
            
        idx = returns_full.index.get_loc(ts)
        start_idx = max(0, idx - 30 + 1)
        
        # Slice returns for distance matrix
        returns_window = returns_full.iloc[start_idx : idx + 1][pool]
        if returns_window.shape[1] < 2: continue
        
        dist_matrix = regime_detector.compute_distance_matrix(returns_window) # uses clean pandas/numpy
        
        # Cluster
        clusters = selector.cluster_candidates(dist_matrix, pool, k=5)
        
        # 3. Dual-Track Scoring
        # Need Styles/Momentum. 
        # Momentum is 20d return.
        # Factor DB doesn't save raw momentum (it saves scores). 
        # Wait, Factor DB saves Anomaly, Degree, Panic. Does it save Momentum?
        # User requirement says: "Cols=[AnomalyScore, Centrality, PanicScore, Momentum]"
        # Check factor_db.py outputs. 
        # Current factor_db.py saves: anomaly_score, degree, panic_score. NO MOMENTUM!
        # I need to Add Momentum to factor_db.py output or calc on fly.
        # Calc on fly is cheap: (P_t / P_t-20) - 1.
        
        # Calc Momentum on fly for pool
        # prices_window for momentum [t-20 : t]
        p_window_20 = prices_full.iloc[max(0, idx - 20) : idx + 1][pool]
        if len(p_window_20) > 1:
            p_end = p_window_20.iloc[-1]
            p_start = p_window_20.iloc[0] # approx 20 days ago
            momentum = (p_end - p_start) / p_start
        else:
            momentum = pd.Series(0, index=pool)
            
        # Panic Scores (from DB)
        panic_scores = daily_factors.loc[pool, 'panic_score'].fillna(0)
        anom_scores_pool = daily_factors.loc[pool, 'anomaly_score'].fillna(0)
        degrees_pool = daily_factors.loc[pool, 'degree'].fillna(0)
        
        # Prepare Styles DataFrame (dummy fundamentals for now)
        # Calculate style factors expects 'momentum' in styles?
        # CandidateSelector.calculate_style_factors calcs momentum/vol internally from prices.
        # Let's pass pool_prices to it.
        # It needs [t-252 : t] for some styles?
        # selector.calculate_style_factors uses:
        # returns.std() (Vol), (p - p_20)/p_20 (Mom), (p - p_5)/p_5 (Rev).
        # So we need ~252d window prices.
        p_start_style = max(0, idx - 252)
        prices_style = prices_full.iloc[p_start_style : idx + 1][pool]
        styles = selector.calculate_style_factors(prices_style, pd.DataFrame())
        
        # Select
        tasks = selector.select_candidates(
            clusters=clusters,
            styles=styles,
            anomaly_scores=anom_scores_pool,
            panic_scores=panic_scores,
            degrees=degrees_pool,
            market_regime=regime,
            include_hint=True # We want explicit Long/Short/Analyzes
        )
        
        # 4. Collect Result
        import json
        results.append({
            'date': ts,
            'tasks': json.dumps(tasks)
        })
        
    # Save
    if results:
        df_out = pd.DataFrame(results).set_index('date')
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_parquet(output_file)
        logger.info(f"Saved Daily Candidates to {output_file}")
    else:
        logger.warning("No candidates generated.")

if __name__ == "__main__":
    batch_selection()
