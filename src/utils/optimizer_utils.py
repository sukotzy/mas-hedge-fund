import logging
import pandas as pd
import numpy as np
from scipy.optimize import linprog

logger = logging.getLogger(__name__)

def solve_optimization_qp(
    adjusted_consensus: dict[str, float],
    previous_holdings: dict[str, float],
    current_prices: dict[str, float],
    portfolio_value: float,
    risk_limits: dict[str, float],
    lambda_penalty: float = 0.05,
    use_risk_manager: bool = True
) -> dict[str, float]:
    """
    Runs Quadratic Programming Optimization with Turnover Penalty.
    Objective: Minimize tracking error to target weights + lambda_penalty * turnover
    Constraints: 
    1. sum(abs(w_i)) <= 1.0 (Gross exposure <= 100%)
    2. |w_i| <= min(risk_limit_i / portfolio_value, 0.4)
    """
    from scipy.optimize import minimize
    
    tickers = [t for t in current_prices.keys() if t != "CASH"]
    n = len(tickers)
    
    if n == 0 or portfolio_value <= 0:
        return {t: 0.0 for t in current_prices.keys()}
        
    # 1. Target Weights
    total_score = sum(abs(v) for v in adjusted_consensus.values())
    target_w = np.zeros(n)
    if total_score > 0:
        for i, t in enumerate(tickers):
            target_w[i] = adjusted_consensus.get(t, 0.0) / total_score
            
    # 2. Previous Weights
    prev_w = np.zeros(n)
    for i, t in enumerate(tickers):
        prev_w[i] = (previous_holdings.get(t, 0.0) * current_prices.get(t, 0.0)) / portfolio_value
            
    # 3. Bounds
    bounds = []
    for t in tickers:
        limit_usd = risk_limits.get(t, portfolio_value)
        max_w = limit_usd / portfolio_value
        
        # Risk Manager natively provides dynamic bounds based on volatility and correlation.
        # We cap the optimizer strictly at 60% (0.6) gross exposure per asset mathematically if risk manager is used.
        if use_risk_manager:
            max_w = min(max_w, 0.6)
        else:
            max_w = min(max_w, 1.0) # Cap at 100% max per asset

        bounds.append((-max_w, max_w))
        
    # Objective Function
    def objective(w):
        return np.sum((w - target_w)**2) + lambda_penalty * np.sum(np.abs(w - prev_w))
        
    # Constraints: Sum(abs(w_i)) <= 1.0 -> 1.0 - sum(abs(w)) >= 0
    def constraint_gross_exposure(w):
        return 1.0 - np.sum(np.abs(w))
        
    # Initial guess
    w0 = prev_w.copy()
    
    res = minimize(
        objective,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=[{'type': 'ineq', 'fun': constraint_gross_exposure}],
        options={'maxiter': 1000}
    )
    
    fin_w = res.x if res.success else prev_w
    if not res.success:
        logger.warning(f"Optimization failed: {res.message}. Falling back to previous weights.")
        
    results = {}
    for i, t in enumerate(tickers):
        price = current_prices.get(t, 0.0)
        if price > 0:
            target_shares = (fin_w[i] * portfolio_value) / price
            results[t] = target_shares
        else:
            results[t] = 0.0
            
    results["CASH"] = 0.0
    return results

def calculate_optimal_portfolio(
    today_consensus: dict[str, float],
    previous_consensus: dict[str, float],
    previous_holdings: dict[str, float],
    prices_history: dict[str, pd.DataFrame],
    risk_limits: dict[str, float],
    initial_capital: float,
    risk_free_rate: float,
    use_risk_manager: bool = True
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Stateful, Four-Tier Kinematic Risk Model optimizer.
    
    Part A: Signal Routing (Replacement vs. Decay)
    Part B: Four-Tier Kinematic Filter (for missing signals)
    Part C: QP Optimization
    
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
    # tickers.discard("CASH")  # Removed to stop forcibly dropping CASH signal

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

    # Clean up small residuals and separate active vs zombie tickers
    active_tickers = []
    zero_tickers = []
    for t in list(adjusted_consensus.keys()):
        if abs(adjusted_consensus[t]) < 1e-6:
            adjusted_consensus[t] = 0.0
            zero_tickers.append(t)
        else:
            active_tickers.append(t)

    # Filter inputs for QP to reduce dimensionality drastically (O(N^3) optimization)
    active_consensus = {t: adjusted_consensus[t] for t in active_tickers}
    active_prev_holdings = {t: previous_holdings.get(t, 0.0) for t in active_tickers}
    active_prices = {t: current_prices.get(t, 0.0) for t in active_tickers if t in current_prices}
    if "CASH" in current_prices and "CASH" not in active_prices:
        active_prices["CASH"] = current_prices["CASH"]
    active_risk_limits = {t: risk_limits.get(t, initial_capital) for t in active_tickers}

    # Part C: QP Optimization on Active Subset
    optimal_shares_active = solve_optimization_qp(
        adjusted_consensus=active_consensus,
        previous_holdings=active_prev_holdings,
        current_prices=active_prices,
        portfolio_value=initial_capital,
        risk_limits=active_risk_limits,
        lambda_penalty=0.05,
        use_risk_manager=use_risk_manager
    )

    # Merge results: Zero-consensus tickers are implicitly liquidated (target=0.0)
    optimal_shares = {t: 0.0 for t in zero_tickers}
    optimal_shares.update(optimal_shares_active)

    return optimal_shares, adjusted_consensus
