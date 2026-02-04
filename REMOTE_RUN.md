# Remote Execution Guide (ETH Euler)

This guide helps you deploy the Batch Processing Pipeline to a remote server (High-Performance Cluster) to generate the 10-year training dataset efficiently.

## 1. Preparation

Ensure your local environment is clean and `requirements.txt` is ready.

```bash
# Verify requirements exist
dir requirements.txt
```

## 2. Upload to Euler

Use `scp` (or `rsync`) to upload the project code and data.
Replace `your_username@euler.ethz.ch` with your actual login.

```powershell
# In PowerShell (Local)
# Create a folder on server
ssh your_username@euler.ethz.ch "mkdir -p ~/mas_hedge_fund/data/raw"

# Upload Source Code
scp -r src your_username@euler.ethz.ch:~/mas_hedge_fund/
scp requirements.txt your_username@euler.ethz.ch:~/mas_hedge_fund/

# Upload Raw Data (Only need raw parquet files)
# Assuming your local data is in d:\...\data\raw
scp d:\path\to\data\raw\*.parquet your_username@euler.ethz.ch:~/mas_hedge_fund/data/raw/
```

## 3. Execution on Server

Login to the server and run the processing.

```bash
# Login
ssh your_username@euler.ethz.ch

# Go to folder
cd ~/mas_hedge_fund

# Setup Python Environment (Load module or use venv)
# Option A: Standard Venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Option B: Load module (if available on Euler)
# module load python/3.x

# --- STEP 1: Generate Factors (Heavy) ---
# This computes MST/Isolation Forest for 10 years
# (~1-2 Hours)
python -m src.selection.batch.factor_db

# --- STEP 2: Generate Candidates (Light) ---
# This generates the daily selection JSONs
# (~20 Minutes)
python -m src.selection.batch.batch_selection
```

## 4. Download Results

Once finished, download the processed files back to your local machine.

```powershell
# In PowerShell (Local)
scp your_username@euler.ethz.ch:~/mas_hedge_fund/data/processed/*.parquet d:\path\to\local\data\processed\
```

## Output Files
You will receive:
1.  `daily_candidates_with_hint.parquet`: Contains `{"action": "long/short"}`.
2.  `daily_candidates_no_hint.parquet`: Contains `{"action": "analyze"}`.
