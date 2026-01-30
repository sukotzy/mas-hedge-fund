import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from typing import List, Dict, Tuple

class CandidateSelector:
    """
    Layer 2 Component: Selects final candidates from the pool.
    Uses Hierarchical Clustering (Ward's method) to ensure diversity.
    Maps clusters to investment styles (Value, Momentum, etc.) for agent routing.
    """
    
    def cluster_candidates(self, dist_matrix: np.ndarray, tickers: List[str], k: int = 5) -> Dict[int, List[str]]:
        """
        Perform Hierarchical Clustering on the distance matrix.
        Returns a mapping of Cluster ID -> List of Tickers.
        """
        if len(tickers) < k:
             # Fallback if specific pool is too small
             return {0: tickers}
             
        # Compact distance matrix form for linkage
        # distance matrix must be condensed (upper triangular)
        # scipy squareform needed? 
        # dist_matrix from Layer 1 is square.
        from scipy.spatial.distance import squareform
        
        # Check if symmetric
        if not np.allclose(dist_matrix, dist_matrix.T):
             # Force symmetry
             dist_matrix = (dist_matrix + dist_matrix.T) / 2
             np.fill_diagonal(dist_matrix, 0)
        
        condensed_dist = squareform(dist_matrix)
        
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

    def select_candidates(self, 
                         clusters: Dict[int, List[str]], 
                         styles: pd.DataFrame, 
                         anomaly_scores: pd.Series,
                         market_regime: str,
                         include_hint: bool = True) -> List[Dict]:
        """
        Select 1 representative from each cluster.
        Logic:
        - If Cluster has High Anomaly Score -> Short Candidate.
        - Else -> Long Candidate (Momentum/Quality).
        
        Args:
            include_hint: If True, includes 'action' (long/short) and specific reason.
                          If False, action is 'analyze' and reason is generic.
        
        Returns a list of candidates to be analyzed by ALL agents.
        """
        candidates = []
        
        for cluster_id, local_tickers in clusters.items():
            if not local_tickers: continue
            
            # Get data for this cluster
            cluster_styles = styles.loc[local_tickers]
            cluster_anoms = anomaly_scores.loc[local_tickers] if not anomaly_scores.empty else pd.Series(0, index=local_tickers)
            
            # 1. Check for Short Candidates first (Priority in Crisis)
            # Find max anomaly score in cluster
            best_anom_ticker = cluster_anoms.idxmax()
            best_anom_score = cluster_anoms.max()
            
            # Threshold for "Short"
            # If score is high (relative) and momentum is negative
            is_short = False
            selected_ticker = None
            reason = ""
            
            if best_anom_score > 0.0: # Threshold: >0 means determined as 'Outlier' by iForest
                 mom = cluster_styles.loc[best_anom_ticker, 'momentum']
                 if mom < 0:
                     is_short = True
                     selected_ticker = best_anom_ticker
                     reason = f"High Anomaly ({best_anom_score:.2f}) & Neg Mom"
            
            if not is_short:
                # Select based on Momentum (Trend Following)
                # In Expansion: Pick High Momentum
                # In Contraction: Pick Low Volatility? (Not computed here yet)
                
                # Simple selection: Highest Momentum
                best_mom_ticker = cluster_styles['momentum'].idxmax()
                selected_ticker = best_mom_ticker
                reason = f"Cluster Leader Momentum: {cluster_styles.loc[selected_ticker, 'momentum']:.2f}"
            
            # Create Candidate Dict
            # Note: No specific 'agent' assigned. This candidate is for the whole team.
            
            if include_hint:
                action = "short" if is_short else "long"
                final_reason = f"Cluster {cluster_id}: {reason}"
            else:
                action = "analyze"
                final_reason = f"Cluster {cluster_id} Representative"

            candidate = {
                "ticker": selected_ticker,
                "action": action,
                "reason": final_reason,
                "regime": market_regime
            }
            candidates.append(candidate)
            
        return candidates
