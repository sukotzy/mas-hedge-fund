import pandas as pd
import numpy as np
import logging
import time
from src.selection.layer1_detectors import TopologyFilter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_on_real_data():
    logger.info("Loading Real Data (sp500_ohlcv.parquet)...")
    try:
        # Load Raw Data
        df = pd.read_parquet("data/raw/sp500_ohlcv.parquet")
        logger.info(f"Loaded {len(df)} rows.")
        
        # Pivot to Wide Format (Title x Date)
        # Assuming 'prc' is column
        logger.info("Pivoting data to (Date x Ticker)...")
        prices = df.pivot(index='date', columns='ticker', values='prc')
        
        # Fill missing
        prices = prices.ffill(limit=3)
        
        # Calculate Returns
        logger.info("Calculating Log Returns...")
        returns = np.log(prices / prices.shift(1))
        
        # Slice a 504-day window from the end
        if len(returns) < 504:
            logger.warning("Not enough history for 504-day window test.")
            return

        # Take last 504 days
        window = returns.iloc[-504:].dropna(axis=1, how='any')
        logger.info(f"Test Window Shape: {window.shape}")
        
        if window.shape[1] < 10:
            logger.warning("Not enough assets in window.")
            return
            
        # Run Robust Structure
        logger.info("Running compute_robust_structure...")
        topo = TopologyFilter()
        
        start_time = time.time()
        degrees, tickers = topo.compute_robust_structure(window)
        end_time = time.time()
        
        logger.info(f"Computation finished in {end_time - start_time:.4f} seconds.")
        logger.info(f"Generated {len(degrees)} degree values.")
        
        # Basic check
        if len(degrees) == len(tickers) and len(degrees) > 0:
            logger.info("SUCCESS: Structure computed correctly on REAL data.")
            logger.info(f"Sample Degrees: {degrees[:5]}")
        else:
            logger.error("FAILURE: Degree count mismatch or empty.")

    except Exception as e:
        logger.error(f"TEST FAILED with Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_on_real_data()
