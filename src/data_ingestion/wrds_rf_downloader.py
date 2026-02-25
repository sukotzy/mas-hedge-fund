import wrds
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def connect_to_wrds():
    """Connect to WRDS using credentials from environment."""
    load_dotenv()
    try:
        username = os.getenv("WRDS_USERNAME")
        if username:
            db = wrds.Connection(wrds_username=username)
        else:
            db = wrds.Connection()
        return db
    except Exception as e:
        logger.error(f"Failed to connect to WRDS: {e}")
        return None

def download_rf_data(output_dir: str = "data/processed"):
    """
    Downloads Fama French daily factors (ff.factors_daily) to get the point-in-time Risk Free rate (rf).
    Saves the data as a parquet file.
    """
    db = connect_to_wrds()
    if not db:
        logger.error("Cannot proceed without WRDS connection.")
        return

    logger.info("Connected to WRDS. Fetching ff.factors_daily...")
    
    # Query to fetch Date and RF.
    # RF in Fama-French is usually provided as a raw decimal for the *daily* rate, but we should inspect it.
    # For example, if it's 0.0001 it means 0.01% daily return.
    sql = """
        SELECT 
            date, 
            rf as risk_free_rate
        FROM 
            ff.factors_daily
        ORDER BY 
            date ASC
    """
    
    try:
        df = db.raw_sql(sql, date_cols=['date'])
        
        if df is None or df.empty:
            logger.error("No data fetched from ff.factors_daily.")
            return
            
        # Ensure date index
        df.set_index('date', inplace=True)
        
        # Save to parquet
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        file_path = out_path / "daily_risk_free_rates.parquet"
        df.to_parquet(file_path)
        logger.info(f"Successfully saved {len(df)} risk free rate records to {file_path}")
        
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    download_rf_data()
