import numpy as np
import json
import logging
import pandas as pd
from typing import Dict, List, Any
from src.selection.data import SelectionDataLoader
from src.selection.layer1 import MarketRegimeDetector, AnomalyDetector
from src.selection.layer2 import CandidateSelector

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_selection_pipeline(end_date: str, lookback_days: int = 252, include_hint: bool = True) -> Dict[str, Any]:
    """
    Executes the full Data Selection Layer pipeline.
    
    1. Fetch Point-in-Time Data (Diff < 500ms usually if cached).
    2. Layer 1: Identify Regime (MST) & Detect Anomalies (iForest).
    3. Layer 2: Cluster Candidates & Route to Agents.
    4. Return JSON Task list.
    """
    logger.info(f"Starting Selection Pipeline for {end_date}...")
    
    # --- Step 1: Data Fetching ---
    loader = SelectionDataLoader()
    prices, volume, tickers = loader.fetch_universe_data(end_date, lookback_days)
    
    if prices.empty:
        logger.error("No price data found. formatting empty response.")
        return {"market_state": "Unknown", "tasks": []}
        
    logger.info(f"Data loaded: {len(tickers)} tickers, {len(prices)} days history.")
    
    # Calculate Log Returns
    returns = np.log(prices / prices.shift(1)).dropna()
    
    # --- Step 2: Layer 1 (Regime & Anomalies) ---
    logger.info("Executing Layer 1: Regime & Anomalies...")
    
    # 2a. Regime Detection
    regime_detector = MarketRegimeDetector()
    dist_matrix = regime_detector.compute_distance_matrix(returns)
    mst = regime_detector.build_mst(dist_matrix)
    ntl = regime_detector.calculate_ntl(mst)
    
    # Calculate Degree Centrality (Topology)
    from src.selection.layer1 import TopologyFilter, get_combined_candidate_pool
    topo_filter = TopologyFilter()
    degrees_array = topo_filter.compute_degree_centrality(mst)
    degrees_series = pd.Series(degrees_array, index=tickers)
    
    # Get Topology Candidates (Hubs & Leaves)
    topo_candidates = topo_filter.get_topology_candidates(degrees_array, tickers)
    
    # For NTL history, we calculate it over a rolling window (e.g., past 60 days).
    # This allows us to establish a baseline (Mean) and Volatility (Std) for the tree length.
    
    ntl_history = []
    
    # We need to iterate backwards from end_date.
    # returns dataframe index is dates.
    # Let's pick the last 60 available distinct (trading) dates.
    available_dates = returns.index.sort_values(ascending=False)
    if len(available_dates) > 60:
        history_dates = available_dates[:60] # Includes current date (index 0)
    else:
        history_dates = available_dates
        
    for d in history_dates:
        # returns for this specific day?
        # NTL is a snapshot property of the correlation matrix calculated over a WINDOW ending at d.
        # Wait, correlation is usually rolling.
        # D_t = sqrt(2(1 - rho_t)). rho_t is exp weighted moving avg or rolling window corr.
        # Our current implementation: `regime_detector.compute_distance_matrix(returns)` 
        # calculates correlation of the ENTIRE provided returns matrix (lookback 252 days).
        
        # If we passed `returns` (252 days), corr() is the static correlation over that year.
        # This is WRONG for "Dynamic" MST. MST needs to evolve.
        # The correlation matrix should be calculated on a rolling window (e.g. 30-60 days) ENDING at time t.
        
        # FIX:
        # Loop: For each date t in history (last 60 days):
        #    Slice returns for [t - window, t]
        #    Compute Corr -> Dist -> MST -> NTL
        #    Store NTL
        
        # Window size for correlation: Short term is better for regime detection? 
        # Research paper usually suggests varying windows, maybe 20-60 days. 
        # Let's use 30 days rolling correlation window.
        
        # Find index of date d
        if d not in returns.index: continue
        d_loc = returns.index.get_loc(d)
        
        CORR_WINDOW = 30
        # If we don't have enough data before d, skip
        if d_loc < CORR_WINDOW: continue
        
        # Slice: returns[d - window : d]
        # Note: iloc is exclusive on upper bound? No, typical slice.
        # get_loc returns integer index.
        window_slice = returns.iloc[d_loc - CORR_WINDOW : d_loc + 1]
        
        # Compute NTL for this window
        d_dist = regime_detector.compute_distance_matrix(window_slice)
        d_mst = regime_detector.build_mst(d_dist)
        d_ntl = regime_detector.calculate_ntl(d_mst)
        
        ntl_history.append(d_ntl)
        
    if not ntl_history:
        # Fallback
        ntl_history = [ntl]
        
    # Current NTL is the first item in our history (since we sorted desc)
    # But wait, history for Z-score should ideally exclude current? Or include?
    # Usually history is "past".
    ntl_current_dynamic = ntl_history[0] 
    ntl_past = ntl_history[1:] if len(ntl_history) > 1 else ntl_history
    
    regime = regime_detector.detect_regime(ntl_current_dynamic, ntl_past) 
    logger.info(f"Market Regime Detected: {regime} (NTL: {ntl_current_dynamic:.4f}, Mean: {np.mean(ntl_past):.4f})")

    
    # 2b. Anomaly Detection (Shorts)
    anom_detector = AnomalyDetector()
    features = anom_detector.compute_features(prices, volume)
    anom_scores = anom_detector.detect_anomalies(features)
    
    # --- Step 2c: Filter Logic (30+30 Rule) ---
    pool = get_combined_candidate_pool(topo_candidates, anom_scores)
    logger.info(f"Layer 1 Filter: Reduced universe to {len(pool)} candidates.")
    
    # --- Step 3: Layer 2 (Diversity & Routing) ---
    logger.info("Executing Layer 2: Clustering & Selection...")
    
    selector = CandidateSelector()
    
    # Slice Data for Layer 2
    # We need to maintain indices alignment for distance matrix slicing
    pool_indices = [tickers.index(t) for t in pool if t in tickers]
    # Safety check
    if not pool_indices:
        logger.warning("No candidates found in pool. Using top 50 by volume as fallback.")
        # fallback logic...
        pool = volume.iloc[-1].sort_values(ascending=False).head(50).index.tolist()
        pool_indices = [tickers.index(t) for t in pool if t in tickers]
    
    # Slice Distance Matrix for Clustering
    # dist_matrix is numpy array (N x N)
    pool_dist_matrix = dist_matrix[np.ix_(pool_indices, pool_indices)]
    
    # Diversity Clustering
    clusters = selector.cluster_candidates(pool_dist_matrix, pool, k=5)
    
    # Style Factors & Panic Score
    # Pass subset data to speed up? Or full data and let it handle?
    # Logic is usually cleaner if we pass 'pool' prices.
    pool_prices = prices[pool]
    pool_volume = volume[pool] # We have 'volume' now!
    
    # Calculate Scores
    # Note: Fundamentals still empty for now
    fundamentals = pd.DataFrame() 
    styles = selector.calculate_style_factors(pool_prices, fundamentals)
    panic_scores = selector.calculate_panic_score(pool_prices, pool_volume)
    
    # Selection
    tasks = selector.select_candidates(
        clusters=clusters, 
        styles=styles, 
        anomaly_scores=anom_scores, # Pass full or subset? Index alignment handles it.
        panic_scores=panic_scores,
        degrees=degrees_series,
        market_regime=regime,
        include_hint=include_hint
    )
    
    logger.info(f"Generated {len(tasks)} tasks.")
    
    return {
        "date": end_date,
        "market_state": regime,
        "ntl": float(ntl_current_dynamic),
        "tasks": tasks
    }

