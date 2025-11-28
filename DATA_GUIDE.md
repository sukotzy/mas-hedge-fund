# Data Guide for MAS Hedge Fund

This document explains the data requirements for the Multi-Agent System (MAS) Hedge Fund and how to generate sample data for backtesting.

## 1. System Logic: Backtest Accumulation

**Question:** "If I choose 2024-10-20 to 2024-10-25, on the last day, the asset each agent holds would be accumulated from 10-20, right?"

**Answer:** **Yes.**

The backtest works as follows:
1.  **Initialization (Day 0)**: The system starts with an initial capital allocation (e.g., $100,000 total, split among agents).
2.  **Daily Loop (10-20 to 10-25)**:
    *   **Betting**: Agents place bets based on data available up to that day.
    *   **Settlement**: At the end of each day (or start of next), bets are settled against market price movements.
    *   **Accumulation**: Gains are added to the agent's capital, and losses are deducted.
3.  **Carry-Over**: The updated capital balance is carried over to the next day.
4.  **Final State (10-25)**: On the last day, the capital shown for each agent is the cumulative result of their performance from the start date (10-20) through the end date (10-25).

## 2. Data Requirements

Each agent type requires specific data to function. The system uses a `LocalDataLoader` (in `src/data/local_loader.py`) to read CSV files from the `data/` directory when `USE_LOCAL_DATA=true`.

### Directory Structure
```
data/
├── prices/              # Historical price data (OHLCV)
├── financial_metrics/   # Fundamental ratios (PE, Margins, etc.)
├── news/                # News headlines and sentiment
└── insider_trades/      # Insider transaction records
```

### Data Formats

#### 1. Prices (`data/prices/{TICKER}.csv`)
Required by: **Technical Analyst**, **Valuation Analyst**, **Risk Manager**
Format:
```csv
time,ticker,open,close,high,low,volume
2024-10-01T00:00:00Z,AAPL,150.25,152.30,153.10,149.80,50000000
...
```

#### 2. Financial Metrics (`data/financial_metrics/{TICKER}.csv`)
Required by: **Fundamental Analyst**, **Valuation Analyst**, **Warren Buffett**
Format:
```csv
ticker,report_period,market_cap,pe_ratio,price_to_book_ratio,return_on_equity,net_margin,operating_margin,revenue_growth,earnings_growth,current_ratio,debt_to_equity_ratio
AAPL,2024-09-30,2500000000000,28.5,45.2,0.45,0.25,0.30,0.10,0.12,1.2,0.8
...
```

#### 3. News (`data/news/{TICKER}.csv`)
Required by: **Sentiment Analyst**, **News Sentiment Analyst**
Format:
```csv
ticker,date,title,text,source,url
AAPL,2024-10-01T10:00:00Z,Apple releases new iPhone,Full text...,Reuters,https://...
...
```

#### 4. Insider Trades (`data/insider_trades/{TICKER}.csv`)
Required by: **Fundamental Analyst** (optional signal)
Format:
```csv
ticker,filing_date,transaction_date,owner_name,is_director,is_officer,shares,transaction_type
AAPL,2024-09-30,2024-09-28,Tim Cook,True,True,10000,Sale
...
```

## 3. Generating Sample Data

A script `generate_sample_data.py` has been created to generate realistic mock data for `AAPL`, `MSFT`, and `GOOGL` for the period `2023-01-01` to `2024-12-31`.

To run it:
```bash
.\hf2\Scripts\python.exe generate_sample_data.py
```

This will populate the `data/` directory with the necessary CSV files to run the backtest locally.
