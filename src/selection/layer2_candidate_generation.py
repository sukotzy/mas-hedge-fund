import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from typing import List, Dict, Tuple

class CandidateGenerator:
    """
    Layer 2 Component: Generates final candidates from the pool.
    Uses Hierarchical Clustering (Ward's method) to ensure diversity.
    Maps clusters to investment styles (Value, Momentum, etc.) for agent routing.
    """
    
    def cluster_candidates(self, 
                           dist_matrix: np.ndarray, 
                           tickers: List[str], 
                           features_df: pd.DataFrame = None,
                           sectors: pd.Series = None,
                           k: int = 5) -> Dict[int, List[str]]:
        """
        Perform Hierarchical Clustering on the composite distance matrix.
        Combines Topology (Returns), Risk (Volatility), Liquidity (Volume), and Structure (Sector).
        Returns a mapping of Cluster ID -> List of Tickers.
        """
        if len(tickers) < k:
             # Fallback if specific pool is too small
             return {0: tickers}
             
        from scipy.spatial.distance import squareform, pdist
        
        # 1. Base Topology Distance (Correlation-based)
        dist_topo = (dist_matrix + dist_matrix.T) / 2
        np.fill_diagonal(dist_topo, 0)
        dist_topo = np.clip(dist_topo, 0, None)
        
        # Max scaling for topo to ensure comparability
        max_topo = np.max(dist_topo) if np.max(dist_topo) > 0 else 1.0
        dist_topo = dist_topo / max_topo

        final_dist = dist_topo.copy() * 0.4 # Base weight 40%
        N = len(tickers)

        # 2. Feature Distance (Volatility, Volume Ratio)
        if features_df is not None and not features_df.empty:
            from sklearn.preprocessing import MinMaxScaler
            
            # Ensure order matches tickers
            features_aligned = features_df.reindex(tickers).fillna(0)
            
            # Normalize features to [0, 1]
            scaler = MinMaxScaler()
            feats_scaled = scaler.fit_transform(features_aligned)
            
            # Euclidean distance in feature space
            dist_feat = squareform(pdist(feats_scaled, metric='euclidean'))
            max_feat = np.max(dist_feat) if np.max(dist_feat) > 0 else 1.0
            dist_feat = dist_feat / max_feat
            
            final_dist += dist_feat * 0.3 # Weight 30%

        # 3. Sector Penalty (Hard structural isolation)
        if sectors is not None and not sectors.empty:
            sectors_aligned = sectors.reindex(tickers).fillna(-1).values
            dist_sector = np.zeros((N, N))
            
            for i in range(N):
                for j in range(i+1, N):
                    s1 = sectors_aligned[i]
                    s2 = sectors_aligned[j]
                    
                    if s1 == -1 or s2 == -1:
                        penalty = 0.5 # Unknown sector, mild penalty
                    elif s1 != s2:
                        penalty = 1.0 # Different sector, full penalty
                    else:
                        penalty = 0.0 # Same sector
                    
                    dist_sector[i, j] = penalty
                    dist_sector[j, i] = penalty
                    
            final_dist += dist_sector * 0.3 # Weight 30%

        # 4. Final Formatting for Scipy Linkage
        final_dist = (final_dist + final_dist.T) / 2
        np.fill_diagonal(final_dist, 0)
        final_dist = np.clip(final_dist, 0, None)
        
        condensed_dist = squareform(final_dist)
        
        # Ward's linkage
        Z = linkage(condensed_dist, method='ward')
        
        # Cut tree to get k clusters
        labels = fcluster(Z, t=k, criterion='maxclust')
        
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(tickers[i])
            
        return clusters

    def calculate_style_factors(self, prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
        """
        Compute simple style factors for selection.
        1. Momentum (12m - 1m): Return(t-21 to t-252)
        2. Value (Book-to-Market): Needs Book Value.
        """
        styles = pd.DataFrame(index=prices.columns)
        
        # 1. Momentum (Classic 12-1)
        # Approx 252 days - 21 days
        start_idx = 0
        end_idx = -21 
        
        if len(prices) > 252:
            p_end = prices.iloc[end_idx]
            p_start = prices.iloc[-252]
            mom = (p_end - p_start) / p_start
            styles['momentum'] = mom
        else:
            # Fallback to simple momentum
            styles['momentum'] = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
            
        # 2. Value (Book to Price)
        # Using Fundamentals Snapshot
        if not fundamentals.empty and 'total_equity' in fundamentals.columns:
            # Map fundamentals back to index
            # Index is Ticker
            styles['book_value'] = np.nan
            
            # Create a map
            bv_map = fundamentals.set_index('ticker')['total_equity']
            styles['book_value'] = styles.index.map(bv_map)
            
            # Current Price
            current_price = prices.iloc[-1]
            
            # Shares stats not always available in snapshot properly, 
            # Assuming 'total_equity' is total $, and Market Cap is Price * Shares.
            # B/P = Total Equity / Market Cap.
            # We don't have MarketCap readily here without Shares count.
            # Approximation: If we don't have shares, we can't do B/P accurately.
            # Let's rely on Momentum primarily for now, or assume we can pull Market Cap from Loader later.
            # For this prototype: Score = Momentum for "Trend", and Inverse Momentum for "Reversal"?
            # Or use Fundamentals net_income if available (proxy for profitability).
            
            # Let's use 'net_income' as a 'Quality' proxy if available.
            ni_map = fundamentals.set_index('ticker')['net_income']
            styles['quality'] = styles.index.map(ni_map)
        
        styles = styles.fillna(0)
        return styles

    def calculate_panic_score(self, prices: pd.DataFrame, volume: pd.DataFrame) -> pd.Series:
        """
        Calculate Panic Score based on Price Crash and Volume Spike.
        Formula:
          VolRatio = Volume_t / Mean(Volume_{t-1...t-20})
          Ret = (Close_t - Close_{t-1}) / Close_{t-1}
          If Ret < 0: PanicScore = VolRatio * |Ret| * 100
          Else: 0
        """
        # Ensure we have enough data (21 days: 20 for mean, 1 for current)
        if len(prices) < 21:
            return pd.Series(0, index=prices.columns)
            
        # Current Day (t)
        p_t = prices.iloc[-1]
        p_prev = prices.iloc[-2]
        v_t = volume.iloc[-1]
        
        # Moving Average Volume (t-1 to t-20)
        # rolling mean excluding t?
        # "Mean(Volume_{t-1...t-20})"
        # slice last 21 days: [t-20, ..., t]
        # we want mean of [t-20 ... t-1]
        recent_vol = volume.iloc[-21:-1]
        v_avg = recent_vol.mean()
        
        # Avoid division by zero
        v_avg = v_avg.replace(0, np.nan).fillna(1.0) # If avg vol is 0, VolRatio becomes v_t/1
        
        vol_ratio = v_t / v_avg
        
        # Returns
        ret = (p_t - p_prev) / p_prev
        
        # Calculate Score
        # Vectorized
        panic_scores = pd.Series(0.0, index=prices.columns)
        
        mask_crash = ret < 0
        
        # Score = VolRatio * |Ret| * 100
        scores = vol_ratio * ret.abs() * 100
        
        panic_scores[mask_crash] = scores[mask_crash]
        
        # Sanity check: If Volume data was missing (0), v_t might be 0 -> score 0. Correct.
        return panic_scores.fillna(0)

    def select_candidates(self, 
                         clusters: Dict[int, List[str]], 
                         styles: pd.DataFrame, 
                         anomaly_scores: pd.Series,
                         panic_scores: pd.Series,
                         degrees: pd.Series,
                         market_regime: str,
                         include_hint: bool = True) -> List[Dict]:
        """
        Select 1 representative from each cluster using Dual-Track Scoring.
        
        Track 1 (Long): Score = 0.6 * Mom + 0.4 * (1 - Anom)
        Track 2 (Short): Score = 0.3 * |Mom| + 0.3 * Anom + 0.2 * Centrality + 0.2 * Panic
        
        Selection:
        Compare Max(Short) vs Max(Long).
        If Short > Long * 1.1 -> Short.
        Else -> Long.
        """
        candidates = []
        
        for cluster_id, local_tickers in clusters.items():
            if not local_tickers: continue
            
            # Prepare data for this cluster
            c_styles = styles.loc[local_tickers]
            c_anoms = anomaly_scores.reindex(local_tickers).fillna(0)
            c_panic = panic_scores.reindex(local_tickers).fillna(0)
            c_degrees = degrees.reindex(local_tickers).fillna(0) # Centrality
            
            # Optimization: Clip Anomaly Scores to [0, 1]
            # If Score < 0 (Normal), treat as 0 penalty. If > 1, cap at 1.
            c_anoms_clipped = c_anoms.clip(0, 1)

            # Normalize inputs roughly to [0, 1] or comparable scales if possible
            # Momentum is usually [-0.5, 0.5] or similar.
            # Anomaly Score is usually positive? (We negated decision_function, so usually [-0.5, 0.5]?)
            # Let's assume raw values are used as per formula, but we might needs scaling.
            # Requirement gave explicit weights, implying raw values.
            
            # Track 1: Long Score
            # Condition: Momentum > 0
            mom = c_styles['momentum']
            
            # Anomaly Score normalization? 
            # If Anomaly score is very high (e.g. 0.2), (1 - Anom) is 0.8.
            # If Anomaly score is negative (normal), (1 - Anom) > 1.
            # Let's assume Anomaly Score is somewhat verified.
            
            score_long = (0.6 * mom) + (0.4 * (1 - c_anoms_clipped))
            # Mask: Momentum must be > 0
            score_long[mom <= 0] = -999 # Disqualify
            
            best_long_ticker = score_long.idxmax()
            best_long_val = score_long.max()
            
            # Track 2: Short Score
            # Entry Condition: (Mom < 0) OR (Panic > 2.0)
            
            # FIX: Force strict alignment to silence FutureWarning on | operator
            mom = mom.reindex(c_panic.index)
            
            score_short = (0.3 * mom.abs()) + (0.3 * c_anoms_clipped) + (0.2 * c_degrees) + (0.2 * c_panic)
            
            # Mask
            mask_short = (mom < 0) | (c_panic > 2.0)
            score_short[~mask_short] = -999
            
            best_short_ticker = score_short.idxmax()
            best_short_val = score_short.max()
            
             # Final Selection
            # Default to Long if both invalid? Or skip?
            if best_long_val == -999 and best_short_val == -999:
                continue # No suitable candidate
                
            action = "long"
            selected_ticker = best_long_ticker
            # Formatting reason string
            specific_reason = f"Score Long: {best_long_val:.2f}"
            
            # Comparison Logic
            # If Short valid AND (Short > Long * 1.1 OR Long invalid)
            threshold = best_long_val * 1.1
            
            if best_short_val > -999:
                if (best_long_val == -999) or (best_short_val > threshold):
                    action = "short"
                    selected_ticker = best_short_ticker
                    specific_reason = f"Score Short: {best_short_val:.2f} (Panic: {c_panic[best_short_ticker]:.2f})"
            
            # Application of Hint
            if include_hint:
                final_action = action
                final_reason = f"Cluster {cluster_id}: {specific_reason}"
            else:
                final_action = "analyze"
                final_reason = f"Cluster {cluster_id} Representative (Hidden)"

            candidate = {
                "ticker": selected_ticker,
                "action": final_action,
                "reason": final_reason,
                "regime": market_regime,
                "confidence": round(float(best_short_val if action == "short" else best_long_val), 4)
            }
            candidates.append(candidate)
            
        return candidates
