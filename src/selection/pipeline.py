import numpy as np
import json
import logging
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path
from src.selection.data import SelectionDataLoader
# Updated Imports: Use Merged Detector
from src.selection.layer1_detectors import MarketRegimeDetector, TopologyFilter, get_combined_candidate_pool
# layer1_shared is gone
from src.selection.layer2_candidate_generation import CandidateGenerator

# Setup basic logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_batch_pipeline(end_date: str, lookback_days: int = 252, include_hint: bool = True, preloaded_data: Dict = None) -> Dict[str, Any]:
    """
    Executes the Batch/DB-Backed Selection Layer pipeline.
    
    1. Try to load Pre-computed Factors (Regime, Anomalies) from Parquet.
    2. If found, skip expensive Layer 1 calculation.
    3. If missing, LOG ERROR and return empty (Realtime fallback is removed).
    4. Layer 2: Cluster Candidates & Route to Agents.
    """
    logger.info(f"Starting Selection Pipeline (Batch Only) for {end_date}...")
    
    # --- Optimization: Try Pre-computed Data ---
    processed_dir = Path("data/processed")
    regime_path = processed_dir / "market_regime.parquet"
    factors_path = processed_dir / "stock_factors.parquet"
    
    use_precomputed = False
    precomputed_regime = None
    factors_df = None
    target_ts = pd.to_datetime(end_date)
    
    if preloaded_data:
        regime_full = preloaded_data['regime_full']
        factors_full = preloaded_data['factors_full']
        
        if target_ts in regime_full.index:
            precomputed_regime = regime_full.loc[target_ts]
            
            try:
                # If factors_full is indexed by date
                factors_df = factors_full.loc[[target_ts]]
            except KeyError:
                factors_df = pd.DataFrame()
                
            if not factors_df.empty:
                use_precomputed = True
    else:
        # Fallback to disk read
        if regime_path.exists() and factors_path.exists():
            try:
                regime_df = pd.read_parquet(regime_path)
                if target_ts in regime_df.index:
                    precomputed_regime = regime_df.loc[target_ts]
                    
                    try:
                        factors_df = pd.read_parquet(factors_path, filters=[('date', '==', target_ts)])
                    except Exception:
                        full_factors = pd.read_parquet(factors_path)
                        factors_df = full_factors[full_factors['date'] == target_ts]
                    
                    if not factors_df.empty:
                        use_precomputed = True
            except Exception as e:
                logger.warning(f"Failed to load pre-computed data: {e}")

    # If Pre-computation NOT available, return empty (No Realtime Fallback)
    if not use_precomputed:
        logger.error(f"Pre-computed data missing for {end_date}. Please run factor_db.py first.")
        return {"market_state": "Unknown", "tasks": []}

    # --- Step 1: Data Fetching (Still needed for Layer 2 Prices/Volume) ---
    loader = preloaded_data['loader'] if preloaded_data else SelectionDataLoader()
    prices, volume, tickers = loader.fetch_universe_data(end_date, lookback_days)
    
    if prices.empty:
        logger.error("No price data found. formatting empty response.")
        return {"market_state": "Unknown", "tasks": []}
        
    logger.info(f"Data loaded: {len(tickers)} tickers. Using Pre-computed Factors.")
    
    # --- Step 2: Layer 1 (Loaded from DB) ---
    regime = precomputed_regime['regime']
    ntl_current = precomputed_regime['ntl']
    
    # Factors Mapping
    valid_factors = factors_df.set_index('ticker')
    
    # Align with current tickers (fill 0 for missing)
    anom_scores = valid_factors['anomaly_score'].reindex(tickers).fillna(0)
    degrees_series = valid_factors['degree'].reindex(tickers).fillna(0)
    volatility_series = valid_factors['volatility_20d'].reindex(tickers).fillna(0) if 'volatility_20d' in valid_factors.columns else pd.Series(0, index=tickers)
    volume_ratio_series = valid_factors['volume_ratio'].reindex(tickers).fillna(1.0) if 'volume_ratio' in valid_factors.columns else pd.Series(1.0, index=tickers)
    
    # Fetch Sectors
    sector_series = loader.fetch_sectors(tickers, target_date=end_date)
    # Panic Score Removed from DB read (Optimization)
    
    # Reconstruct Topology Candidates from Degree Series
    topo_filter = TopologyFilter()
    topo_candidates = topo_filter.get_topology_candidates(degrees_series.values, tickers)
    
    logger.info(f"Loaded Regime: {regime} (NTL: {ntl_current:.4f})")

    # --- Step 2c: Filter Logic (30+30 Rule) ---
    pool = get_combined_candidate_pool(topo_candidates, anom_scores)
    logger.info(f"Layer 1 Filter: Reduced universe to {len(pool)} candidates.")
    
    # --- Step 3: Layer 2 (Diversity & Routing) ---
    logger.info("Executing Layer 2: Clustering & Selection...")
    
    generator = CandidateGenerator()
    
    # Slice Data for Layer 2
    pool_indices = [tickers.index(t) for t in pool if t in tickers]
    if not pool_indices:
        logger.warning("No candidates found in pool. Using top 50 by volume as fallback.")
        pool = volume.iloc[-1].sort_values(ascending=False).head(50).index.tolist()
        pool_indices = [tickers.index(t) for t in pool if t in tickers]
    
    # Slice Distance Matrix for Clustering
    # We still need a distance matrix for the clustering step on the subset
    # Compute on-the-fly for the filtered pool (much cheaper than N*N)
    # returns for clustering:
    returns_now = np.log(prices / prices.shift(1)).dropna()
    
    # Using MarketRegimeDetector for clustering distance since we have clean slice and want consistency
    regime_detector = MarketRegimeDetector()
    dist_matrix_full = regime_detector.compute_distance_matrix(returns_now)
    pool_dist_matrix = dist_matrix_full[np.ix_(pool_indices, pool_indices)]
    
    # Diversity Clustering
    pool_volatility = volatility_series.reindex(pool).fillna(0)
    pool_volume_ratio = volume_ratio_series.reindex(pool).fillna(1.0)
    pool_sectors = sector_series.reindex(pool).fillna(-1) # -1 implies unknown sector
    
    features_df = pd.DataFrame({
        'volatility': pool_volatility,
        'volume_ratio': pool_volume_ratio
    })
    
    clusters = generator.cluster_candidates(
        dist_matrix=pool_dist_matrix, 
        tickers=pool,
        features_df=features_df,
        sectors=pool_sectors,
        k=5
    )
    
    # Style Factors
    pool_prices = prices[pool]
    pool_volume = volume[pool]
    fundamentals = pd.DataFrame() 
    styles = generator.calculate_style_factors(pool_prices, fundamentals)
    
    # Panic Scores (On-the-fly Calculation for Candidates Only)
    logger.info("Calculating Panic Scores for candidates...")
    panic_scores = generator.calculate_panic_score(pool_prices, pool_volume)
    
    # Selection
    tasks = generator.select_candidates(
        clusters=clusters, 
        styles=styles, 
        anomaly_scores=anom_scores, 
        panic_scores=panic_scores,
        degrees=degrees_series,
        market_regime=regime,
        include_hint=include_hint
    )
    
    logger.info(f"Generated {len(tasks)} tasks.")
    
    return {
        "date": end_date,
        "market_state": regime,
        "ntl": float(ntl_current),
        "tasks": tasks
    }
