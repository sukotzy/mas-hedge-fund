import logging
import pandas as pd
import numpy as np
from scipy.optimize import linprog

logger = logging.getLogger(__name__)

# Global Deadband threshold for floating-point noise truncation
# A weight < 1e-6 corresponds to < $0.1 on a $100k account, purely numerical debris
WEIGHT_EPSILON = 1e-6

def solve_optimization_qp(
    adjusted_consensus: dict[str, float],
    previous_holdings: dict[str, float],
    current_prices: dict[str, float],
    portfolio_value: float,
    risk_limits: dict[str, float],
    today_consensus: dict[str, float],
    lambda_penalty: float = 0.05,
    use_risk_manager: bool = True,
    segregate_capital: float = 0.0
) -> dict[str, float]:
    """
    Runs Quadratic Programming Optimization with Turnover Penalty.
    Objective: Minimize tracking error to target weights + lambda_penalty * turnover
    Constraints: 
    1. sum(abs(w_i)) <= 1.0 (Gross exposure <= 100%)
    2. |w_i| <= min(risk_limit_i / portfolio_value, 0.4)
    """
    from scipy.optimize import minimize
    # Make iteration deterministic to prevent floating-point chaos
    tickers = sorted([t for t in current_prices.keys() if t != "CASH"])
    n = len(tickers)
    
    if n == 0 or portfolio_value <= 0:
        return {t: 0.0 for t in current_prices.keys()}
        
    # 1. Target Weights Calculation
    target_w = np.zeros(n)
    
    # --- NEW FIX: 提取 CASH 的比例，压缩风险资产池 ---
    total_global_score = sum(abs(v) for v in adjusted_consensus.values())
    cash_score = abs(adjusted_consensus.get("CASH", 0.0))
    cash_ratio = (cash_score / total_global_score) if total_global_score > 0 else 0.0
    risky_alloc_budget = 1.0 - cash_ratio # 留给股票的实际仓位上限
    
    if segregate_capital > 0.0:
        # --- NEW LOGIC: Capital Segregation ---
        # Separate today's fresh signals from old decayed holdings
        fresh_tickers = [t for t in tickers if t in today_consensus and today_consensus[t] != 0.0]
        old_tickers = [t for t in tickers if t not in fresh_tickers]
        
        fresh_score = sum(abs(adjusted_consensus.get(t, 0.0)) for t in fresh_tickers)
        old_score = sum(abs(adjusted_consensus.get(t, 0.0)) for t in old_tickers)
        
        # Determine bucket allocations 
        if fresh_score > 0 and old_score > 0:
            fresh_bucket_weight = segregate_capital * risky_alloc_budget
            old_bucket_weight = (1.0 - segregate_capital) * risky_alloc_budget
        elif fresh_score > 0 and old_score == 0:
            fresh_bucket_weight = 1.0 * risky_alloc_budget
            old_bucket_weight = 0.0
        elif fresh_score == 0 and old_score > 0:
            fresh_bucket_weight = 0.0
            old_bucket_weight = 1.0 * risky_alloc_budget
        else:
            fresh_bucket_weight = 0.0
            old_bucket_weight = 0.0

        for i, t in enumerate(tickers):
            score = abs(adjusted_consensus.get(t, 0.0))
            if score == 0:
                target_w[i] = 0.0
                continue
                
            if t in fresh_tickers:
                target_w[i] = (score / fresh_score) * fresh_bucket_weight
            else:
                target_w[i] = (score / old_score) * old_bucket_weight
    else:
        # --- ORIGINAL LOGIC: Proportional Allocation ---
        if total_global_score > 0:
            for i, t in enumerate(tickers):
                target_w[i] = abs(adjusted_consensus.get(t, 0.0)) / total_global_score
            
    # 2. Previous Weights
    prev_w = np.zeros(n)
    for i, t in enumerate(tickers):
        prev_w[i] = (previous_holdings.get(t, 0.0) * current_prices.get(t, 0.0)) / portfolio_value
        
    target_w = np.round(target_w, 6)
    prev_w = np.round(prev_w, 6)
            
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

        bounds.append((-round(max_w, 6), round(max_w, 6)))
        
    # Objective Function
    def objective(w):
        tracking_error = np.sum((w - target_w)**2)
        # Pseudo-Huber Loss for smooth, differentiable L1 approximation
        turnover = np.sum(np.sqrt((w - prev_w)**2 + 1e-8)) 
        return tracking_error + lambda_penalty * turnover
        
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
        
    fin_w = np.round(fin_w, 5) # <--- ADD THIS TRUNCATION BEFORE CALCULATING SHARES
        
    # --- EPSILON DEADBAND TRUNCATION & RE-NORMALIZATION ---
    # 1. Truncate absolute floating point garbage to pure 0.0
    fin_w = np.where(np.abs(fin_w) < WEIGHT_EPSILON, 0.0, fin_w)
    
    # 2. Re-normalize remaining weights to conserve intended scale
    current_sum = np.sum(np.abs(fin_w))
    original_sum = np.sum(np.abs(res.x if res.success else prev_w))
    if current_sum > 0 and original_sum > 0:
        fin_w = fin_w * (original_sum / current_sum)
    # ------------------------------------------------------
        
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
    use_risk_manager: bool = True,
    turnover_penalty: float = 0.05,
    decay_mode: str = "harsh",
    segregate_capital: float = 0.0
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
    - use_risk_manager: whether to use the risk manager for limits.
    - turnover_penalty: L1 regularization penalty for trading turnover.
    
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
        elif ticker == "CASH":
            # ✅ 新逻辑：CASH 是主动避险信号，绝不跨日继承。
            # 如果今天没有明确要求持有 CASH，则直接释放流动性，将现金共识清零
            adjusted_consensus["CASH"] = 0.0
        else:
            # Part A.2: Missing/zero signal but we have holdings
            holdings = previous_holdings.get(ticker, 0.0)
            
            # Translate raw shares into portfolio weight to apply Epsilon Deadband
            price = current_prices.get(ticker, 0.0)
            holding_weight = (holdings * price) / initial_capital if initial_capital > 0 else 0.0
            
            if holding_weight > WEIGHT_EPSILON or holding_weight < -WEIGHT_EPSILON:
                prev_cons = previous_consensus.get(ticker, 0.0)
                
                # Part B: Four-Tier Kinematic Filter
                decay_factor = 0.0
                
                if decay_mode == "none":
                    decay_factor = 0.0
                else:
                    df = prices_history.get(ticker)
                    if df is not None and len(df) >= 6 and "close" in df.columns:
                        closes = df["close"].dropna().values
                        if len(closes) >= 6:
                            latest_close = closes[-1]
                            prev_close_1 = closes[-2]
                            prev_close_2 = closes[-3]
                            ma5 = np.mean(closes[-5:])
                            price_delta_1d = (latest_close - prev_close_1) / prev_close_1 if prev_close_1 > 0 else 0.0
                            price_delta_3d = latest_close - closes[-4]
                            R_t_minus_1 = (prev_close_1 - prev_close_2) / prev_close_2 if prev_close_2 > 0 else 0.0
                            prev_close_3 = closes[-4]
                            R_t_minus_2 = (prev_close_2 - prev_close_3) / prev_close_3 if prev_close_3 > 0 else 0.0
                            
                            if decay_mode == "soft":
                                # Soft Mode: 20% hard cutoff, no MA5, slower decay across the board
                                cond_A_long = holding_weight > WEIGHT_EPSILON and price_delta_1d <= -0.20
                                cond_A_short = holding_weight < -WEIGHT_EPSILON and price_delta_1d >= 0.20
                                
                                if cond_A_long or cond_A_short:
                                    decay_factor = 0.0
                                else:
                                    if holding_weight * price_delta_3d > WEIGHT_EPSILON:
                                        decay_factor = 0.98  # Tailwind Soft Landing
                                    elif holding_weight * price_delta_3d < -WEIGHT_EPSILON:
                                        is_long = holding_weight > WEIGHT_EPSILON
                                        if is_long:
                                            accel = (R_t_minus_1 < R_t_minus_2) and (R_t_minus_2 < 0)
                                        else:
                                            accel = (R_t_minus_1 > R_t_minus_2) and (R_t_minus_2 > 0)
                                            
                                        if accel:
                                            decay_factor = 0.70  # Headwind Acceleration
                                        else:
                                            decay_factor = 0.90  # Headwind Deceleration
                                    else:
                                        decay_factor = 0.90
                                        
                            elif decay_mode == "harsh":
                                # Original Harsh Mode
                                cond_A_long = holding_weight > WEIGHT_EPSILON and price_delta_1d <= -0.08
                                cond_A_short = holding_weight < -WEIGHT_EPSILON and price_delta_1d >= 0.08
                                
                                prev_ma5 = np.mean(closes[-6:-1])
                                cond_B_long = holding_weight > WEIGHT_EPSILON and (latest_close < ma5) and (prev_close_1 < prev_ma5)
                                cond_B_short = holding_weight < -WEIGHT_EPSILON and (latest_close > ma5) and (prev_close_1 > prev_ma5)
                                
                                if cond_A_long or cond_A_short or cond_B_long or cond_B_short:
                                    decay_factor = 0.0
                                else:
                                    if holding_weight * price_delta_3d > WEIGHT_EPSILON:
                                        decay_factor = 0.95  # Tailwind Soft Landing
                                    elif holding_weight * price_delta_3d < -WEIGHT_EPSILON:
                                        is_long = holding_weight > WEIGHT_EPSILON
                                        if is_long:
                                            accel = (R_t_minus_1 < R_t_minus_2) and (R_t_minus_2 < 0)
                                        else:
                                            accel = (R_t_minus_1 > R_t_minus_2) and (R_t_minus_2 > 0)
                                            
                                        if accel:
                                            decay_factor = 0.50  # Headwind Acceleration
                                        else:
                                            decay_factor = 0.85  # Headwind Deceleration
                                    else:
                                        decay_factor = 0.85
                        else:
                            decay_factor = 0.0
                    else:
                        decay_factor = 0.0
                    
                adjusted_consensus[ticker] = prev_cons * decay_factor
            else:
                # Deadband intercept: Explicitly force tiny noisy fractions to exactly 0.0
                adjusted_consensus[ticker] = 0.0

    # Clean up residuals and apply Top-K Truncation (Max 50 positions)
    MAX_POSITIONS = 50
    active_tickers = []
    zero_tickers = []
    
    # Sort tickers by absolute consensus score descending, breaking ties alphabetically
    sorted_tickers = sorted(adjusted_consensus.keys(), key=lambda t: (abs(adjusted_consensus[t]), t), reverse=True)
    
    for i, t in enumerate(sorted_tickers):
        score = adjusted_consensus[t]
        # Keep if within Top 50 AND above the minimum residual threshold
        if i < MAX_POSITIONS and abs(score) >= 1e-6:
            active_tickers.append(t)
        else:
            adjusted_consensus[t] = 0.0
            zero_tickers.append(t)

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
        today_consensus=today_consensus,
        lambda_penalty=turnover_penalty,
        use_risk_manager=use_risk_manager,
        segregate_capital=segregate_capital
    )

    # Merge results: Zero-consensus tickers are implicitly liquidated (target=0.0)
    optimal_shares = {t: 0.0 for t in zero_tickers}
    optimal_shares.update(optimal_shares_active)

    return optimal_shares, adjusted_consensus
