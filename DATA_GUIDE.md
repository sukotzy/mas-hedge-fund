# Data Guide for MAS Hedge Fund

This document explains the data requirements for the Multi-Agent System (MAS) Hedge Fund, covering both the Production (WRDS) pipeline and the Legacy (CSV) testing mode.

## 1. System Logic: Backtest Accumulation

**Question:** "If I choose 2024-10-20 to 2024-10-25, on the last day, the asset each agent holds would be accumulated from 10-20, right?"

**Answer:** **Yes.**

The backtest works as follows:
1.  **Initialization (Day 0)**: The system starts with an initial capital allocation (e.g., $100,000 total, split among agents).
2.  **Daily Loop (10-20 to 10-25)**:
    *   **Betting**: Agents place bets based on data available *up to* that day.
    *   **Settlement**: At the end of each day, bets are settled against market price movements.
    *   **Accumulation**: Gains are added to the agent's capital, and losses are deducted.
3.  **Final State (10-25)**: On the last day, the capital shown is the cumulative result of performance.

---

## 2. WRDS Production Data (Parquet)

This is the primary data source for high-fidelity backtesting. Data is downloaded from WRDS (Wharton Research Data Services) and stored in `data/raw/` as Parquet files.

### 2.1 Constituents (`sp500_constituents.parquet`)
*   **Source:** `crsp.msp500list`
*   **Usage:** Defines the investment universe (S&P 500 historical members).
*   **Key Columns:** `permno`, `start`, `ending`, `ticker`.

### 2.2 Prices & Volume (`sp500_ohlcv.parquet`)
*   **Source:** CRSP Daily Stock File (`crsp.dsf`)
*   **Frequency:** **Daily**
*   **Bias Prevention:** Point-in-time daily market data.
*   **Key Columns:** 
    *   `date`: Trading date
    *   `permno`: Unique CRSP identifier
    *   `prc`: Closing Price
    *   `vol`: Volume
    *   `openprc`, `askhi`, `bidlo`: OHLC components

### 2.3 Financial Ratios (`sp500_ratios_firm_ratio.parquet`)
*   **Source:** WRDS Financial Ratios Suite (`wrdsapps.firm_ratio`)
*   **Frequency:** **Monthly** (e.g., 2016-01-31, 2016-02-29).
*   **Bias Prevention:** Uses **`public_date`** (the date the data became public knowledge), NOT the fiscal reference date. This ensures no look-ahead bias.
*   **Key Columns:**
    *   `date` (mapped from `public_date`): The effective date for trading.
    *   `pe_exi`: P/E Ratio (Excl. Extraordinary Items)
    *   `ptb`: Price-to-Book Ratio

### 2.4 Deep Fundamentals (`comp_fundq.parquet`)
*   **Source:** Compustat Quarterly Fundamentals (`comp.fundq`)
*   **Frequency:** **Quarterly**
*   **Bias Prevention:** STRICTLY uses **`rdq` (Release Date Quarterly)**. This is the exact date the 10-Q/10-K was filed with the SEC.
    *   *Note:* If a fiscal quarter ends on March 31 but the report is filed on April 25, the data becomes available to agents ONLY on April 25.
*   **Key Columns:**
    *   **Timestamp:** `rdq` (used as index), `datadate` (fiscal refernece).
    *   **Raw Items:** `niq` (Net Income), `revtq` (Revenue), `atq` (Total Assets), `seqq` (Shareholder Equity), `dlttq` (Long-term Debt).
    *   **Calculated Metrics:** 
        *   `return_on_equity` (ROE)
        *   `net_margin`
        *   `operating_margin`
        *   `current_ratio`
        *   `debt_to_equity`

### 2.5 Metadata Linkage (`ccm_links.parquet`)
*   **Source:** CRSP-Compustat Merged Link Table
*   **Usage:** Maps CRSP `permno` (for Prices) to Compustat `gvkey` (for Fundamentals).

---

## 3. Data Ingestion Commands

### To Download WRDS Data:
```bash
python src/data_ingestion/wrds_downloader.py --start-date 2015-01-01 --end-date 2024-12-31
```
*Requires `WRDS_USERNAME` in `.env` and `pgpass.conf` logic.*

### To Generate Mock Data (Testing):
```bash
python generate_sample_data.py
```

---

## 4. Legacy/Testing Data (CSV Mode)

Used for local testing without WRDS access. Files are stored in `data/`.

#### 4.1 Directory Structure
```
data/
├── prices/              # CSVs: time, ticker, open, close...
├── financial_metrics/   # CSVs: pe_ratio, price_to_book...
├── news/                # CSVs: title, sentiment...
└── insider_trades/      # CSVs: transaction details...
```

#### 4.2 Format Examples
*   **Prices**: `time,ticker,open,close,high,low,volume`
*   **Metrics**: `ticker,report_period,pe_ratio,price_to_book_ratio...`
