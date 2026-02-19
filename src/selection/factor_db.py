import pandas as pd
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
from src.selection.data import get_local_loader
# Updated Imports
from src.selection.layer1_detectors import MarketRegimeDetector, AnomalyDetector, TopologyFilter
# layer1_shared is gone

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def batch_compute_factors(data_dir: str = "data", start_year: int = 2015, limit: int = None):
    """
    Batch compute all Layer 1 factors (Regime, Anomalies, Topology) for history.
    Saves to:
    - data/processed/market_regime.parquet
    - data/processed/stock_factors.parquet
    """
    logger.info("Loading Data...")
    loader = get_local_loader(data_dir)
    
    # We need the full OHLCV history to iterate
    ohlcv = loader.ohlcv.copy()
    if ohlcv.empty:
        logger.error("No OHLCV data found.")
        return

    # Merge Ticker info (Permno -> Ticker)
    constituents = loader.constituents
    if constituents.empty:
        logger.error("No constituents data found.")
        return
        
    # We need a map. Since ticker mapping changes over time, we should strictly merge on date ranges.
    # But for batch processing huge history, a simple approximate merge or just using correct Point-in-Time loader is better.
    # actually SelectionDataLoader logic is complex to replicate here for full history.
    # Let's try to do a simplified merge: Permno -> Ticker (taking most recent or dominant ticker)
    # Or better: Use 'date' and 'permno' to find valid ticker?
    # Simple approach: Permno to Ticker map from constituents (drop duplicates keeping last)
    permno_map = constituents[['permno', 'ticker']].drop_duplicates(subset='permno', keep='last')
    
    ohlcv = ohlcv.merge(permno_map, on='permno', how='left')
    ohlcv = ohlcv.dropna(subset=['ticker'])
    
    # Deduplicate: Keep entry with highest volume if multiple permnos map to same ticker on same date
    ohlcv = ohlcv.sort_values('vol', ascending=False).drop_duplicates(subset=['date', 'ticker'])
    
    prices = ohlcv.pivot(index='date', columns='ticker', values='prc').ffill(limit=3)
    volume = ohlcv.pivot(index='date', columns='ticker', values='vol').fillna(0)
    
    # Calculate Returns (Log Returns) on FULL History
    returns = np.log(prices / prices.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan)
    
    # Initialize Detectors 
    regime_detector = MarketRegimeDetector()
    topo_filter = TopologyFilter() 
    anom_detector = AnomalyDetector()
    
    # Validation
    valid_dates = returns.index[returns.index.year >= start_year]
    
    if valid_dates.empty:
        logger.error(f"No valid dates found starting from {start_year}")
        return

    logger.info(f"Processing {len(valid_dates)} days from {valid_dates[0]} to {valid_dates[-1]}...")
    
    # Results Containers

    ntl_records = []
    stock_records = []
    
    # Config - Two-Speed Architecture
    WINDOW_MOOD = 30       # Fast Track (Regime)
    WINDOW_STRUCTURE = 504 # Slow Track (Topology - 2 Years)
    WINDOW_ANOM = 252      # Anomaly (1 Year)
    RECALIBRATION_INTERVAL = 20 # Update Structure every 20 days
    NTL_WINDOW = 60 # For rolling NTL stats
    
    # State for Slow Track
    cached_degrees = None
    cached_structure_tickers = None
    
    # Iterate
    # We iterate over valid_dates, but we need the INTEGER INDEX in the 'returns' dataframe
    # to slice history correctly (including history prior to start_year)
    
    for step, date in enumerate(tqdm(valid_dates)):
        # Limit check
        if limit and step >= limit:
            logger.info(f"Limit of {limit} reached. Stopping early.")
            break
        
        # Get absolute index in the full 'returns' dataframe
        # efficient lookup if index is unique
        try:
            i = returns.index.get_loc(date)
        except KeyError:
            continue
            
        # 1. Slow Track: Structural Recalibration (Every 20 days)
        # Requirement: i >= WINDOW_STRUCTURE (need 2 years history)
        # If start_year is 2015, and data starts 2000, i will be large enough.
        
        if i >= WINDOW_STRUCTURE and (step % RECALIBRATION_INTERVAL == 0 or cached_degrees is None):
            # Slice 504-day window for Structure
            # [i - 504 + 1 : i + 1]
            window_struct = returns.iloc[i - WINDOW_STRUCTURE + 1 : i + 1].dropna(axis=1, how='any')
            
            # Reduce universe to "Active" stocks in this long window? 
            # Or just take everything that has full history?
            # 504 days is long, many stocks might drop in/out. 
            # dropna(axis=1, how='any') removes any stock with even 1 NaN. This is strict but robust for RMT.
            
            if window_struct.shape[1] > 50:
                # logger.info(f"Recalibrating Structure on {date} (Window: {WINDOW_STRUCTURE}d, Assets: {window_struct.shape[1]})...")
                cached_degrees, cached_structure_tickers = topo_filter.compute_robust_structure(window_struct)
            else:
                # Keep previous cache if window is bad, or init empty
                if cached_degrees is None:
                    cached_degrees = np.array([])
                    cached_structure_tickers = []
        
        # 2. Fast Track: Market Mood (Every Day)
        # Requirement: i >= WINDOW_MOOD
        current_ntl = np.nan
        
        if i >= WINDOW_MOOD:
             window_mst = returns.iloc[i - WINDOW_MOOD + 1 : i + 1].dropna(axis=1, how='any')
             if window_mst.shape[1] >= 50:
                 try:
                    dist_matrix = regime_detector.compute_distance_matrix(window_mst)
                    mst = regime_detector.build_mst(dist_matrix)
                    current_ntl = regime_detector.calculate_ntl(mst)
                 except Exception:
                    current_ntl = np.nan
        
        # Store NTL
        ntl_records.append({'date': date, 'ntl': current_ntl})
        
        # 3. Stock Level Factors (Anomalies & Mapping Degrees)
        # We need to output a record for "Current Valid Tickers"
        # Let's define current valid tickers as those in the Anomaly Window (252d) or Mood Window?
        # Usually Anomaly Window defines the tradeable universe filters.
        
        if i >= WINDOW_ANOM:
             # Identify current universe (using Anomaly Window or just Mood Window)
             # Let's use Mood Window (30d) to ensure very recent activity
             # But Anomaly calc needs 252d.
             
             # Mood Window Tickers are active
             current_active_tickers = window_mst.columns.tolist() if 'window_mst' in locals() and not window_mst.empty else []
             
             if not current_active_tickers and cached_structure_tickers:
                 current_active_tickers = cached_structure_tickers # Fallback
            
             if current_active_tickers:
                 # Narrow down data
                 p_end_idx = i 
                 p_start_idx = max(0, p_end_idx - WINDOW_ANOM + 1)
                 
                 # Access full prices/volume using iloc
                 # Note: prices dataframe aligns with returns dataframe?
                 # Yes, if we created returns from it and didn't drop rows from prices.
                 # returns has T rows. prices has T rows (after sync)?
                 # prices = ohlcv.pivot...
                 # returns = log(prices/shift).
                 # returns often has NaN at start.
                 # If we kept NaNs in returns, lengths match.
                 # Line 60: returns = returns.replace([inf], nan).
                 # We didn't dropna() on returns globally, so indices align.
                 
                 wp = prices.iloc[p_start_idx : p_end_idx + 1][current_active_tickers]
                 wv = volume.iloc[p_start_idx : p_end_idx + 1][current_active_tickers]
                 
                 try:
                     # Detect Anomalies
                     features = anom_detector.compute_features(wp, wv)
                     features = features.replace([np.inf, -np.inf], 0).fillna(0)
                     anom_scores = anom_detector.detect_anomalies(features)
                     
                     # Map Cached Degrees to Current Tickers
                     mapped_degrees = np.zeros(len(current_active_tickers))
                     
                     if cached_degrees is not None and len(cached_degrees) > 0:
                         # Create series for mapping
                         deg_map = pd.Series(cached_degrees, index=cached_structure_tickers)
                         # Reindex to current active -> fillna(0)
                         mapped_degrees = deg_map.reindex(current_active_tickers).fillna(0).values
                     
                     # Store
                     df_day = pd.DataFrame({
                        'date': date,
                        'ticker': current_active_tickers,
                        'anomaly_score': anom_scores.values,
                        'degree': mapped_degrees
                     })
                     stock_records.append(df_day)
                     
                 except Exception as e:
                     logger.error(f"Error computing stock factors on {date}: {e}")
    
    # --- Post-Processing (Vectorized) ---
    logger.info("Computing Rolling Stats for NTL...")
    
    if not ntl_records:
        logger.warning("No NTL records computed.")
        return

    regime_df = pd.DataFrame(ntl_records)
    regime_df['date'] = pd.to_datetime(regime_df['date'])
    regime_df.set_index('date', inplace=True)
    
    # Rolling Stats (Shifted by 1 to avoid lookahead bias!)
    # We compare Current NTL to (Mean of Previous 60 days)
    # rolling(60).mean().shift(1)
    
    regime_df['ntl_mean'] = regime_df['ntl'].rolling(window=NTL_WINDOW, min_periods=30).mean().shift(1)
    regime_df['ntl_std'] = regime_df['ntl'].rolling(window=NTL_WINDOW, min_periods=30).std().shift(1)
    
    # Calculate Z-Score
    regime_df['z_score'] = (regime_df['ntl'] - regime_df['ntl_mean']) / regime_df['ntl_std']
    
    # Determine Regime
    # Crisis if Z < -1.5
    regime_df['regime'] = np.where(regime_df['z_score'] < -1.5, 'Crisis_Contraction', 'Normal_Expansion')
    regime_df['regime'] = regime_df['regime'].where(regime_df['z_score'].notna(), 'Normal_Expansion') # Default
    
    # Save Regimes
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    regime_path = output_dir / "market_regime.parquet"
    regime_df.to_parquet(regime_path)
    logger.info(f"Saved Market Regimes to {regime_path}")
    
    # Save Stock Factors
    if stock_records:
        all_factors = pd.concat(stock_records, ignore_index=True)
        factors_path = output_dir / "stock_factors.parquet"
        all_factors.to_parquet(factors_path, index=False)
        logger.info(f"Saved Stock Factors to {factors_path} ({len(all_factors)} rows)")
    else:
        logger.warning("No stock records generated.")

if __name__ == "__main__":
    # Support CLI args for start_year and limit
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, default=2015)
    parser.add_argument("--limit", type=int, default=None)
    
    args = parser.parse_args()
    
    batch_compute_factors(start_year=args.start_year, limit=args.limit)
