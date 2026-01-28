import duckdb
import argparse
import os
import sys
import time

def convert_csv_to_parquet(csv_path: str, parquet_path: str = None, compression: str = 'snappy'):
    """
    Convert a CSV file to Parquet format using DuckDB for high performance and low memory usage.
    
    Args:
        csv_path (str): Path to the input CSV file.
        parquet_path (str, optional): Path to the output Parquet file. 
                                      If None, replaces .csv extension with .parquet.
        compression (str): Compression method (default: 'snappy').
    """
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    if parquet_path is None:
        parquet_path = os.path.splitext(csv_path)[0] + '.parquet'

    start_time = time.time()
    try:
        print(f"Converting {csv_path} to {parquet_path} using DuckDB...")
        
        # DuckDB can query CSV files directly and export to Parquet efficiently
        # It handles large files by streaming, avoiding OOM errors common with pandas
        # sample_size=-1 forces scanning the whole file for type inference
        query = f"""
            COPY (SELECT * FROM read_csv_auto('{csv_path}', sample_size=-1)) 
            TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION '{compression}');
        """
        
        duckdb.sql(query)
        
        elapsed = time.time() - start_time
        print(f"Conversion successful! Time taken: {elapsed:.2f} seconds")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV file to Parquet format using DuckDB.")
    parser.add_argument("csv_file", help="Path to the source CSV file")
    parser.add_argument("--output", "-o", help="Path to the destination Parquet file (optional)", default=None)
    parser.add_argument("--compression", "-c", help="Compression algorithm (default: snappy)", default="snappy")
    
    args = parser.parse_args()
    
    convert_csv_to_parquet(args.csv_file, args.output, args.compression)