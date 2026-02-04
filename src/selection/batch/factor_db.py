import pandas as pd
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
from src.selection.data import get_local_loader
# Updated Imports
from src.selection.batch.layer1_detectors import RobustMarketRegimeDetector, RobustAnomalyDetector
from src.selection.layer1_shared import TopologyFilter

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
    
    # Filter by start year
    prices = prices[prices.index.year >= start_year]
    volume = volume.loc[prices.index]
    
    # Calculate Returns (Log Returns)
    returns = np.log(prices / prices.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan)
    
    # Initialize Detectors (Using Robust Versions)
    regime_detector = RobustMarketRegimeDetector()
    topo_filter = TopologyFilter() # TopologyFilter is simple math, no need to fork
    anom_detector = RobustAnomalyDetector()
    
    # Validation
    valid_dates = returns.index
    logger.info(f"Processing {len(valid_dates)} days from {valid_dates[0]} to {valid_dates[-1]}...")
    
    # Results Containers
    ntl_records = []
    stock_records = []
    
    # Config
    CORR_WINDOW = 30 # For NTL (Rolling Correlation)
    NTL_WINDOW = 60  # For Z-Score (Rolling NTL History)
    
    # Iterate
    # We use tqdm for progress bar
    for i, date in enumerate(tqdm(valid_dates)):
        # Limit check
        if limit and i >= limit:
            logger.info(f"Limit of {limit} reached. Stopping early.")
            break
            
        # Need enough history for correlation window
        if i < CORR_WINDOW:
            ntl_records.append({'date': date, 'ntl': np.nan})
            continue
            
        # 1. Slice Windows
        # We need the window ENDING at 'date'
        
        # Window size constants
        WINDOW_MST = 30
        WINDOW_ANOM = 252
        WINDOW_PANIC = 20
        
        # Determine strict indices
        # If we don't have enough history for MST (30), skip. 
        if i < WINDOW_MST:
            ntl_records.append({'date': date, 'ntl': np.nan})
            continue
            
        # Slice MST Window (30d)
        window_mst = returns.iloc[i - WINDOW_MST + 1 : i + 1].dropna(axis=1, how='any')
        
        if window_mst.shape[1] < 50: 
             ntl_records.append({'date': date, 'ntl': np.nan})
             continue
             
        # Slice Anomaly Window (252d)
        # If unavailable (early history), use whatever is available max
        start_anom = max(0, i - WINDOW_ANOM + 1)
        # Note: Anomaly Features need prices/volume, NOT returns directly because compute_features calcs returns internally.
        # We need raw prices/vol for [t-252 : t]
        
        # prices/volume dataframe indices are same as returns (aligned by date)
        # We need to slice based on integer position 'i'
        # prices is (Date x Ticker). 
        # i corresponds to 'date' in 'returns'. 'prices' has same length + 1 (because returns loses 1 row)?
        # factor_db calculated returns = prices / dict(1). So returns index starts 1 day after prices[0].
        # let's rely on date index slicing to be safe? No, integer is faster for loop.
        # returns has length T. prices has length T+1.
        # returns[i] is return for day prices[i+1].
        # Let's align exactly:
        # window_mst = returns[i-29 : i+1] <- returns for days date-(29) to date.
        
        # For Anomaly Features (Prices/Vol):
        # We need 252 days ending at 'date'. 
        start_price_idx = max(0, i - WINDOW_ANOM + 1) 
        # But prices has 1 extra row at start. 
        # If returns[0] is from prices[0] and prices[1].
        # We generally want the prices corresponding to the returns window.
        
        # Let's just slice strictly by date index to be safe and use existing dataframes
        # This is slower but safer? No, we have integer index i.
        # prices.iloc[i] corresponds to returns.iloc[i-1]?
        # We computed: returns = np.log(prices / prices.shift(1))
        # So returns.iloc[0] is log(prices[1]/prices[0]). index is prices.index[1].
        # So if we are at loop index 'i' (which is index in 'returns'), the date is returns.index[i].
        # The corresponding price row is prices.loc[date] (which is approx prices.iloc[i+1]).
        
        # Let's keep it simple: Use .loc with date range for inputs to detectors if possible?
        # Detectors normally take full DF and calc logic? No, compute_features takes arguments.
        # Let's assume passed DF is the History Window.
        
        # Slice Prices/Volume for Anomaly (252d)
        # We need to include 'date' and 251 days prior.
        # In returns index space: i-251 to i.
        # In prices index space: we need matching rows.
        # Let's simplify: Get the tickers from MST window (active universe).
        current_tickers = window_mst.columns.tolist()
        
        # Valid window length for anomalies
        anom_len = i - start_anom + 1
        
        # 2. Compute NTL (Market Physics) [Window: 30d]
        try:
            dist_matrix = regime_detector.compute_distance_matrix(window_mst)
            mst = regime_detector.build_mst(dist_matrix)
            ntl = regime_detector.calculate_ntl(mst)
            
            # Store NTL
            ntl_records.append({'date': date, 'ntl': ntl})
            
            # 3. Compute Stock-Level Factors
            
            # A. Degree Centrality (Topology)
            degrees = topo_filter.compute_degree_centrality(mst)
            
            # B. Anomaly Scores (Isolation Forest) [Window: 252d]
            # We slice prices/volume for the ACTIVE tickers derived from MST
            # This ensures we don't process zombie stocks
            
            # Need strict slicing for input DFs
            # returns.index[i] is the current date.
            # We want prices slice from (date - 252d) to date.
            # Using slice on .loc is readable.
            # prices.iloc[start_anom : i+something?]
            # Map returns index to price index?
            # prices and returns might not be perfectly aligned if we filtered returns.
            # But we generated returns from prices in this script.
            # prices[1:] should align with returns.
            
            # Price slice for [t-251 : t] (252 days)
            # returns[i] is t. prices[i+1] is t. NO, prices[i] is t.
            # FIX: p_end_idx must be i.
            p_end_idx = i 
            p_start_idx = max(0, p_end_idx - WINDOW_ANOM + 1)
            
            # Slice is exclusive at end, so we need +1 to include p_end_idx
            window_prices = prices.iloc[p_start_idx : p_end_idx + 1].loc[:, current_tickers]
            window_vol = volume.iloc[p_start_idx : p_end_idx + 1].loc[:, current_tickers]
            
            features = anom_detector.compute_features(window_prices, window_vol)
            # Fill inf/nan
            features = features.replace([np.inf, -np.inf], 0).fillna(0)
            
            # Detect
            anom_scores = anom_detector.detect_anomalies(features)
            
            # C. Panic Score [Window: 20d]
            # Window: [t-20 : t]
            # We need Volume Mean of [t-20 : t-1] (20 days prior to t)
            # And Ret of t.
            
            # Slice last 21 days from window_prices (if available) to get Returns and Volume
            # Or reuse window_prices since it covers 252d.
            # We need strictly last 21 days logic.
            
            # Current day values
            v_t = window_vol.iloc[-1]
            
            # Past 20 days volume (excluding current)
            if len(window_vol) >= 21:
                v_hist = window_vol.iloc[-21:-1]
                v_avg = v_hist.mean()
            elif len(window_vol) > 1:
                v_hist = window_vol.iloc[:-1] # Use whatever history we have
                v_avg = v_hist.mean()
            else:
                v_avg = v_t # Avoid div by zero?
            
            v_avg = v_avg.replace(0, 1) # Avoid div by zero
            
            # Price Return t
            # p_t = prices[t], p_prev = prices[t-1]
            p_t = window_prices.iloc[-1]
            p_prev = window_prices.iloc[-2]
            daily_ret = (p_t - p_prev) / p_prev
            
            vol_ratio = v_t / v_avg
            
            panic_mask = (daily_ret < 0).values
            panic_scores = np.zeros(len(current_tickers))
            
            # Calculate scores using numpy directly
            # vol_ratio and daily_ret are series, convert to numpy
            scores = (vol_ratio * daily_ret.abs() * 100).values
            
            panic_scores[panic_mask] = scores[panic_mask]
            
            # Store Data
            # Format: Date | Ticker | Anomaly | Degree | Panic
            df_day = pd.DataFrame({
                'date': date,
                'ticker': current_tickers,
                'anomaly_score': anom_scores.values,
                'degree': degrees,
                'panic_score': panic_scores # Already numpy array
            })
            stock_records.append(df_day)
            
        except Exception as e:
            logger.error(f"Error on {date}: {e}")
            import traceback
            traceback.print_exc()
            ntl_records.append({'date': date, 'ntl': np.nan})

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
