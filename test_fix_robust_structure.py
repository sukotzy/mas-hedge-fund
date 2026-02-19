import pandas as pd
import numpy as np
import logging
from src.selection.layer1_detectors import TopologyFilter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_robust_structure_with_object_dtype():
    logger.info("Testing compute_robust_structure with object dtype inputs...")
    
    # 1. Create a DataFrame with object dtype (simulating the issue)
    # Even if values are numbers, dtype=object causes np.corrcoef to fail or behave oddly
    data = {
        'A': [0.01, 0.02, -0.01, 0.03, 0.01] * 105, # > 504 rows needed? No, func checks T/N
        'B': [0.02, 0.01, 0.00, -0.01, 0.02] * 105,
        'C': [0.05, 0.05, 0.05, 0.05, 0.05] * 105
    }
    df = pd.DataFrame(data)
    
    # Force object dtype
    df = df.astype(object)
    
    # Verify dtype
    if df['A'].dtype == 'object':
        logger.info("Confirmed input DataFrame has object dtype.")
    else:
        logger.warning("Failed to create object dtype DataFrame.")
        
    # Instantiate Filter
    topo = TopologyFilter()
    
    try:
        # Run method
        logger.info("Calling compute_robust_structure...")
        degrees, tickers = topo.compute_robust_structure(df)
        
        logger.info("Successfully computed structure.")
        logger.info(f"Degrees: {degrees}")
        
        if len(degrees) == 3:
            logger.info("TEST PASSED: Output dimensions correct.")
        else:
            logger.error("TEST FAILED: Incorrect output dimensions.")
            
    except Exception as e:
        logger.error(f"TEST FAILED with Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_robust_structure_with_object_dtype()
