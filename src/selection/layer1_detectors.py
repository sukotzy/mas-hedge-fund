import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Tuple

class MarketRegimeDetector:
    """
    Market Regime Detection using MST (Minimum Spanning Tree).
    
    Logic:
    1. Calculate Correlation Matrix (Robust Numpy).
    2. Convert to Distance Matrix: d = sqrt(2(1-rho)).
    3. Build MST.
    4. Calculate NTL (Normalized Tree Length).
    5. Compare current NTL vs History (Z-Score) to detect Crisis.
    """
    
    def compute_distance_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Calculate distance metric d_ij = sqrt(2(1 - rho_ij)).
        Uses Numpy for correlation to handle sparse/infinite data without Pandas alignment crashes.
        """
        # Pearson correlation using Numpy
        # returns is (T x N), np.corrcoef expects (N x T) - variables as rows
        if returns.empty:
            return np.zeros((0, 0))
            
        # Robustness: Force float type and handle any residual NaNs/Infs
        clean_values = returns.fillna(0).astype(float).values.T
        clean_values = np.nan_to_num(clean_values)
        
        corr_matrix = np.corrcoef(clean_values)
        
        # Clip and zero diagonal
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
        corr_matrix = np.nan_to_num(corr_matrix)
        
        dist_matrix = np.sqrt(2 * (1 - corr_matrix))
        np.fill_diagonal(dist_matrix, 0)
        
        return dist_matrix

    def build_mst(self, dist_matrix: np.ndarray) -> np.ndarray:
        """
        Construct MST from distance matrix using Prim's or Kruskal's algorithm.
        """
        mst_csr = minimum_spanning_tree(dist_matrix)
        return mst_csr.toarray()

    def calculate_ntl(self, mst_matrix: np.ndarray) -> float:
        """
        Calculate Normalized Tree Length (NTL).
        NTL = Sum(weights) / (N - 1)
        """
        total_weight = np.sum(mst_matrix)
        num_nodes = mst_matrix.shape[0]
        if num_nodes <= 1:
            return 0.0
        return total_weight / (num_nodes - 1)

    def detect_regime(self, ntl_current: float, ntl_history: List[float]) -> str:
        """
        Determine regime based on Z-score of NTL.
        """
        if not ntl_history or len(ntl_history) < 10:
            return "Normal_Expansion"
            
        mu = np.mean(ntl_history)
        sigma = np.std(ntl_history)
        
        if sigma == 0:
            return "Normal_Expansion"
            
        z_score = (ntl_current - mu) / sigma
        
        if z_score < -1.5:
            return "Crisis_Contraction"
        else:
            return "Normal_Expansion"

class AnomalyDetector:
    """
    Anomaly Detection using Isolation Forest + Financial Features.
    
    Features:
    1. Idiosyncratic Volatility
    2. Volume-Price Divergence
    3. Momentum
    """
    
    def compute_features(self, prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for anomaly detection.
        Robust to empty inputs and infinite values.
        """
        returns = np.log(prices / prices.shift(1)).dropna()
        vol_changes = np.log(volume / volume.shift(1)).dropna()
        
        # Handle infinite values (log of 0)
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        vol_changes = vol_changes.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Align dates
        common_index = returns.index.intersection(vol_changes.index)
        returns = returns.loc[common_index]
        vol_changes = vol_changes.loc[common_index]
        
        features = pd.DataFrame(index=prices.columns)
        
        # 1. Volatility
        features['volatility'] = returns.std() * np.sqrt(252)
        
        # 2. Volume-Price Correlation (Explicit Loop for robustness)
        vp_dict = {}
        for ticker in returns.columns:
            try:
                if ticker in vol_changes.columns:
                    r = returns[ticker]
                    v = vol_changes[ticker]
                    valid_idx = r.dropna().index.intersection(v.dropna().index)
                    if len(valid_idx) > 2:
                        vp_dict[ticker] = r.loc[valid_idx].corr(v.loc[valid_idx])
                    else:
                        vp_dict[ticker] = 0.0
                else:
                    vp_dict[ticker] = 0.0
            except Exception:
                vp_dict[ticker] = 0.0
                
        features['vp_divergence'] = pd.Series(vp_dict).reindex(features.index).fillna(0)
        
        # 3. Momentum
        if len(prices) >= 20:
            p_end = prices.iloc[-1]
            p_start = prices.iloc[-20]
            momentum = (p_end - p_start) / p_start
        else:
            momentum = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
            
        features['momentum'] = momentum
        features = features.fillna(0)
        
        return features

    def detect_anomalies(self, features: pd.DataFrame, contamination: float = 0.1) -> pd.Series:
        """
        Run Isolation Forest.
        Returns: Anomaly Scores (higher is more anomalous).
        """
        X = features[['volatility', 'vp_divergence', 'momentum']].values
        
        clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        clf.fit(X)
        
        scores = -clf.decision_function(X) 
        
        return pd.Series(scores, index=features.index, name='anomaly_score')

class TopologyFilter:
    """
    Selects candidates based on Network Topology (MST).
    Goal: Select 'Hubs' (Central info flow) and 'Leaves' (Isolated/Distinct).
    """
    
    def compute_degree_centrality(self, mst_matrix: np.ndarray) -> np.ndarray:
        """
        Compute degree centrality for each node in the MST.
        """
        if not np.allclose(mst_matrix, mst_matrix.T):
            mst_matrix = np.maximum(mst_matrix, mst_matrix.T)
        degrees = np.count_nonzero(mst_matrix, axis=1)
        return degrees

    def get_topology_candidates(self, degrees: np.ndarray, tickers: List[str], n_hubs: int = 15, n_leaves: int = 15) -> List[str]:
        """
        Select Top N Hubs (High Degree) and Top N Leaves (Low Degree).
        """
        if len(tickers) != len(degrees):
             return []
        deg_series = pd.Series(degrees, index=tickers)
        hubs = deg_series.nlargest(n_hubs).index.tolist()
        leaves = deg_series.nsmallest(n_leaves).index.tolist()
        return list(set(hubs + leaves))

def get_combined_candidate_pool(
    topology_candidates: List[str], 
    anomaly_scores: pd.Series, 
    top_n_anomalies: int = 30
) -> List[str]:
    """
    The "Physical Screen": Combine Topology (Hubs+Leaves) with Anomalies.
    """
    anomalies = anomaly_scores.sort_values(ascending=False).head(top_n_anomalies).index.tolist()
    pool = list(set(topology_candidates + anomalies))
    return pool
