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
from src.selection.layer2 import CandidateSelector

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_batch_pipeline(end_date: str, lookback_days: int = 252, include_hint: bool = True) -> Dict[str, Any]:
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
    
    if regime_path.exists() and factors_path.exists():
        try:
            target_ts = pd.to_datetime(end_date)
            # Load Regime (Lazy load ideally, but for now read full)
            regime_df = pd.read_parquet(regime_path)
            if target_ts in regime_df.index:
                precomputed_regime = regime_df.loc[target_ts]
                
                # Load Factors using filters (fast parquet read)
                try:
                    factors_df = pd.read_parquet(factors_path, filters=[('date', '==', target_ts)])
                except Exception:
                    # Fallback if filters not supported by pyarrow engine/version
                    full_factors = pd.read_parquet(factors_path)
                    factors_df = full_factors[full_factors['date'] == target_ts]
                
                if not factors_df.empty:
                    use_precomputed = True
                    logger.info("Values found in Pre-computed DB. Skipping heavy computation.")
        except Exception as e:
            logger.warning(f"Failed to load pre-computed data: {e}")

    # If Pre-computation NOT available, return empty (No Realtime Fallback)
    if not use_precomputed:
        logger.error(f"Pre-computed data missing for {end_date}. Please run factor_db.py first.")
        return {"market_state": "Unknown", "tasks": []}

    # --- Step 1: Data Fetching (Still needed for Layer 2 Prices/Volume) ---
    loader = SelectionDataLoader()
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
    db_panic_scores = valid_factors['panic_score'].reindex(tickers).fillna(0)
    
    # Reconstruct Topology Candidates from Degree Series
    topo_filter = TopologyFilter()
    topo_candidates = topo_filter.get_topology_candidates(degrees_series.values, tickers)
    
    logger.info(f"Loaded Regime: {regime} (NTL: {ntl_current:.4f})")

    # --- Step 2c: Filter Logic (30+30 Rule) ---
    pool = get_combined_candidate_pool(topo_candidates, anom_scores)
    logger.info(f"Layer 1 Filter: Reduced universe to {len(pool)} candidates.")
    
    # --- Step 3: Layer 2 (Diversity & Routing) ---
    logger.info("Executing Layer 2: Clustering & Selection...")
    
    selector = CandidateSelector()
    
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
    clusters = selector.cluster_candidates(pool_dist_matrix, pool, k=5)
    
    # Style Factors
    pool_prices = prices[pool]
    pool_volume = volume[pool]
    fundamentals = pd.DataFrame() 
    styles = selector.calculate_style_factors(pool_prices, fundamentals)
    
    # Panic Scores (Use DB value for consistency with Batch, or Recalc?)
    # DB value is consistent with the regime calculation logic. Use DB.
    # Reindex to pool
    panic_scores = db_panic_scores.reindex(pool).fillna(0)
    
    # Selection
    tasks = selector.select_candidates(
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
