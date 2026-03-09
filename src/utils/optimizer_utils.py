import logging
import pandas as pd
import numpy as np
from scipy.optimize import linprog

logger = logging.getLogger(__name__)

def solve_optimization_lp(
    consensus_values: dict[str, float],
    current_prices: dict[str, float],
    risk_limits: dict[str, float],
    initial_capital: float,
    risk_free_rate: float
) -> dict[str, float]:
    """
    Runs LP Optimization.
    Objective: Maximize Sum(Value_i * Action_i) + (Initial_Capital * RiskFreeRate)
    Constraints:
    1. |Action_i * Price_i| <= Risk_Limit_i
    2. Sum(Action_i * Price_i) = 0 (Self-financing via collateral)
    """
    tickers = list(current_prices.keys())
    if "CASH" not in tickers:
        tickers.append("CASH")
        
    c = []
    prices = []
    bounds = []
    
    for t in tickers:
        value = consensus_values.get(t, 0.0)
        c.append(-value) # Minimize negative value
        
        if t == "CASH":
            price = 1.0 # Cash is always $1
            prices.append(price)
            bounds.append((-initial_capital, initial_capital))
        else:
            price = current_prices[t]
            prices.append(price)
            
            limit_usd = risk_limits.get(t, 0.0)
            
            if price > 0:
                max_shares = limit_usd / price
                bounds.append((-max_shares, max_shares))
            else:
                bounds.append((0, 0))
            
    if not c or all(val == 0.0 for val in c):
        logger.warning("All consensus values are zero (no bets). Returning zero allocations.")
        return {t: 0.0 for t in tickers}
        
    A_eq = [prices]
    b_eq = [0.0]
    
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    results = {}
    if res.success:
        for i, t in enumerate(tickers):
            results[t] = res.x[i]
    else:
        logger.warning(f"Optimization failed: {res.message}")
        for t in tickers:
            results[t] = 0.0
                
    return results

def calculate_optimal_portfolio(
    today_consensus: dict[str, float],
    previous_consensus: dict[str, float],
    previous_holdings: dict[str, float],
    prices_history: dict[str, pd.DataFrame],
    risk_limits: dict[str, float],
    initial_capital: float,
    risk_free_rate: float
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Stateful, Four-Tier Kinematic Risk Model optimizer.
    
    Part A: Signal Routing (Replacement vs. Decay)
    Part B: Four-Tier Kinematic Filter (for missing signals)
    Part C: LP Optimization
    
    Arguments:
    - today_consensus: dict of net consensus today per ticker.
    - previous_consensus: dict of consensus from the previous day.
    - previous_holdings: dict of net holdings (shares) from the previous day per ticker.
    - prices_history: dict mapping ticker -> pd.DataFrame of at least last 6 days of OHLCV prices (must contain 'close' column, sorted by date).
    - risk_limits: dict of max USD risk allowed per ticker.
    - initial_capital: total fund wealth or starting capital.
    - risk_free_rate: risk free rate for the period.
    
    Returns:
    - optimal_shares: dict of target net shares for each ticker and CASH.
    - adjusted_consensus: the actual consensus values used, incorporating decay.
    """
    adjusted_consensus = {}
    tickers = set()
    tickers.update(today_consensus.keys())
    tickers.update(previous_holdings.keys())
    tickers.discard("CASH")

    # Determine current prices (latest close)
    current_prices = {}
    for ticker in tickers:
        df = prices_history.get(ticker)
        if df is not None and not df.empty and "close" in df.columns:
            # Drop NaN and get the last valid close
            valid_closes = df["close"].dropna()
            if not valid_closes.empty:
                current_prices[ticker] = float(valid_closes.iloc[-1])
            else:
                current_prices[ticker] = 0.0
        else:
            current_prices[ticker] = 0.0

    for ticker in tickers:
        has_signal_today = ticker in today_consensus and today_consensus[ticker] != 0.0
        
        if has_signal_today:
            # Part A.1: Strictly REPLACE
            adjusted_consensus[ticker] = today_consensus[ticker]
        else:
            # Part A.2: Missing/zero signal but we have holdings
            holdings = previous_holdings.get(ticker, 0.0)
            if holdings != 0.0:
                prev_cons = previous_consensus.get(ticker, 0.0)
                
                # Part B: Four-Tier Kinematic Filter
                decay_factor = 0.0
                df = prices_history.get(ticker)
                
                if df is not None and len(df) >= 6 and "close" in df.columns:
                    closes = df["close"].dropna().values
                    
                    if len(closes) >= 6:
                        # Latest 6 days: t-5, t-4, t-3, t-2, t-1, t
                        latest_close = closes[-1]
                        prev_close_1 = closes[-2]
                        prev_close_2 = closes[-3]
                        
                        # Calculate metrics
                        ma5 = np.mean(closes[-5:])
                        price_delta_1d = (latest_close - prev_close_1) / prev_close_1 if prev_close_1 > 0 else 0.0
                        
                        # 3-day momentum: today close - 3 days ago close (which is closes[-4])
                        # Wait, "today close - 3 days ago close"
                        price_delta_3d = latest_close - closes[-4]
                        
                        # Daily returns
                        R_t_minus_1 = (prev_close_1 - prev_close_2) / prev_close_2 if prev_close_2 > 0 else 0.0
                        prev_close_3 = closes[-4]
                        R_t_minus_2 = (prev_close_2 - prev_close_3) / prev_close_3 if prev_close_3 > 0 else 0.0
                        
                        # Determine tier
                        # Tier 1: Absolute Defense (Hard Cutoff)
                        cond_A_long = holdings > 0 and price_delta_1d <= -0.08
                        cond_A_short = holdings < 0 and price_delta_1d >= 0.08
                        
                        # Crosses MA5 for 2 consecutive days adversely
                        # Adverse: Long -> price < MA5; Short -> price > MA5
                        prev_ma5 = np.mean(closes[-6:-1]) # ma5 of t-6 to t-2
                        cond_B_long = holdings > 0 and (latest_close < ma5) and (prev_close_1 < prev_ma5)
                        cond_B_short = holdings < 0 and (latest_close > ma5) and (prev_close_1 > prev_ma5)
                        
                        if cond_A_long or cond_A_short or cond_B_long or cond_B_short:
                            decay_factor = 0.0
                        else:
                            # Tiers 2, 3, 4
                            if holdings * price_delta_3d > 0:
                                # Tier 2: Tailwind Soft Landing
                                decay_factor = 0.8
                            elif holdings * price_delta_3d < 0:
                                # It's a headwind
                                is_long = holdings > 0
                                if is_long:
                                    # Adverse return accelerating: R_t_minus_1 < R_t_minus_2 < 0
                                    accel = (R_t_minus_1 < R_t_minus_2) and (R_t_minus_2 < 0)
                                else:
                                    # For short: price going up is adverse. 
                                    # Accelerating: R_t_minus_1 > R_t_minus_2 > 0
                                    accel = (R_t_minus_1 > R_t_minus_2) and (R_t_minus_2 > 0)
                                    
                                if accel:
                                    # Tier 3: Headwind Acceleration
                                    decay_factor = 0.2
                                else:
                                    # Tier 4: Headwind Deceleration
                                    decay_factor = 0.5
                            else:
                                # price_delta_3d == 0 or holdings == 0
                                decay_factor = 0.5
                else:
                    # Incomplete price history -> conservative decay
                    decay_factor = 0.0
                    
                adjusted_consensus[ticker] = prev_cons * decay_factor
            else:
                # No signal and holding is 0
                adjusted_consensus[ticker] = 0.0

    # Clean up small residuals
    for t in list(adjusted_consensus.keys()):
        if abs(adjusted_consensus[t]) < 1e-6:
            adjusted_consensus[t] = 0.0

    # Part C: LP Optimization
    optimal_shares = solve_optimization_lp(
        consensus_values=adjusted_consensus,
        current_prices=current_prices,
        risk_limits=risk_limits,
        initial_capital=initial_capital,
        risk_free_rate=risk_free_rate
    )

    return optimal_shares, adjusted_consensus
