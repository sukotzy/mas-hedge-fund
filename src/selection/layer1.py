import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Tuple

class MarketRegimeDetector:
    """
    Layer 1 Component: Detects Market Regime using Minimum Spanning Tree (MST).
    Calculates Normalized Tree Length (NTL) to distinguish between 'Crisis/Contraction' and 'Normal/Expansion'.
    """
    
    def compute_distance_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Calculate distance metric d_ij = sqrt(2(1 - rho_ij)).
        Input: Log Returns DataFrame (Date x Ticker).
        """
        # Pearson correlation
        corr_matrix = returns.corr().values
        
        # Clip to handle floating point errors slightly outside [-1, 1]
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
        
        # Distance transformation
        dist_matrix = np.sqrt(2 * (1 - corr_matrix))
        
        # Ensure diagonal is 0
        np.fill_diagonal(dist_matrix, 0)
        
        return dist_matrix

    def build_mst(self, dist_matrix: np.ndarray) -> np.ndarray:
        """
        Construct MST from distance matrix using Prim's or Kruskal's algorithm (scipy uses Prim/Kruskal).
        Returns the adjacency matrix of the MST.
        """
        mst_csr = minimum_spanning_tree(dist_matrix)
        return mst_csr.toarray()

    def calculate_ntl(self, mst_matrix: np.ndarray) -> float:
        """
        Calculate Normalized Tree Length (NTL).
        NTL = Sum(weights) / (N - 1)
        """
        # MST matrix is symmetric or upper triangular? Scipy returns upper triangular.
        # Sum of all weights
        total_weight = np.sum(mst_matrix)
        
        num_nodes = mst_matrix.shape[0]
        if num_nodes <= 1:
            return 0.0
            
        ntl = total_weight / (num_nodes - 1)
        return ntl

    def detect_regime(self, ntl_current: float, ntl_history: List[float]) -> str:
        """
        Determine regime based on Z-score of NTL.
        If Z < -1.5 (Tree shrinking rapidly) -> 'Crisis_Contraction'
        Else -> 'Normal_Expansion'
        """
        if not ntl_history or len(ntl_history) < 10:
            return "Normal_Expansion" # Default if insufficient history
            
        mu = np.mean(ntl_history)
        sigma = np.std(ntl_history)
        
        if sigma == 0:
            return "Normal_Expansion"
            
        z_score = (ntl_current - mu) / sigma
        
        # Threshold from paper: -1.5 indicates significant contraction (Crisis)
        if z_score < -1.5:
            return "Crisis_Contraction"
        else:
            return "Normal_Expansion"

class AnomalyDetector:
    """
    Layer 1 Component: Detects Anomalies for Short Candidates using Isolation Forest.
    Focuses on 'Structural Breaks' or 'Manipulations'.
    """
    
    def compute_features(self, prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for anomaly detection:
        1. Idiosyncratic Volatility (proxy: std of returns)
        2. Volume-Price Divergence (corr between price change and vol change)
        3. Momentum (Return over window)
        """
        returns = np.log(prices / prices.shift(1)).dropna()
        vol_changes = np.log(volume / volume.shift(1)).dropna()
        
        # Align dates
        common_index = returns.index.intersection(vol_changes.index)
        returns = returns.loc[common_index]
        vol_changes = vol_changes.loc[common_index]
        
        features = pd.DataFrame(index=prices.columns)
        
        # 1. Volatility (Annualized)
        features['volatility'] = returns.std() * np.sqrt(252)
        
        # 2. Volume-Price Correlation
        # Vectorized correlation per column
        vp_corr = returns.corrwith(vol_changes)
        features['vp_divergence'] = vp_corr
        
        # 3. Momentum (Last 20 days)
        # Using simple return: (P_end - P_start) / P_start
        # Or log return sum
        if len(prices) >= 20:
            p_end = prices.iloc[-1]
            p_start = prices.iloc[-20]
            momentum = (p_end - p_start) / p_start
        else:
            momentum = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
            
        features['momentum'] = momentum
        
        # Fill NaNs
        features = features.fillna(0)
        
        return features

    def detect_anomalies(self, features: pd.DataFrame, contamination: float = 0.1) -> pd.Series:
        """
        Run Isolation Forest.
        Returns: Anomaly Scores (higher is more anomalous).
        sklearn IForest returns -1 for outlier, 1 for inlier. 
        decision_function returns lower scores for anomalies.
        We want a normalized 'Anomaly Score' where High = Anomalous.
        """
        # Prepare X
        X = features[['volatility', 'vp_divergence', 'momentum']].values
        
        clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        clf.fit(X)
        
        # decision_function: Average anomaly score of X of the base classifiers.
        # The anomaly score of an input sample is computed as the mean anomaly score of the trees in the forest.
        # The measure of normality of an observation given a tree is the depth of the leaf containing this observation, 
        # which is equivalent to the number of splittings required to isolate this point. 
        # In case of several trees, the ensemble anomaly score is the mean of the scores of each tree.
        # For sklearn: The lower, the more abnormal. 
        # We negate it to make "Higher = More Anomalous"
        
        scores = -clf.decision_function(X) 
        
        return pd.Series(scores, index=features.index, name='anomaly_score')

        return candidates.head(top_n).index.tolist()

class TopologyFilter:
    """
    Layer 1 Component: Selects candidates based on Network Topology (MST).
    Goal: Select 'Hubs' (Central info flow) and 'Leaves' (Isolated/Distinct).
    """
    
    def compute_degree_centrality(self, mst_matrix: np.ndarray) -> np.ndarray:
        """
        Compute degree centrality for each node in the MST.
        MST matrix is expected to be the adjacency matrix.
        """
        # Ensure MST is symmetric (undirected graph)
        # If input is upper triangular, make it symmetric
        if not np.allclose(mst_matrix, mst_matrix.T):
            mst_matrix = np.maximum(mst_matrix, mst_matrix.T)
            
        # Degree = Number of non-zero connections per node
        # Since it's a weighted matrix, we count non-zeros
        degrees = np.count_nonzero(mst_matrix, axis=1)
        return degrees

    def get_topology_candidates(self, degrees: np.ndarray, tickers: List[str], n_hubs: int = 15, n_leaves: int = 15) -> List[str]:
        """
        Select Top N Hubs (High Degree) and Top N Leaves (Low Degree).
        """
        if len(tickers) != len(degrees):
             return []
             
        # Create Series
        deg_series = pd.Series(degrees, index=tickers)
        
        # Hubs: Largest degrees
        hubs = deg_series.nlargest(n_hubs).index.tolist()
        
        # Leaves: Smallest degrees (typically degree 1 in MST)
        # We take the bottom N.
        leaves = deg_series.nsmallest(n_leaves).index.tolist()
        
        return list(set(hubs + leaves))

def get_combined_candidate_pool(
    topology_candidates: List[str], 
    anomaly_scores: pd.Series, 
    top_n_anomalies: int = 30
) -> List[str]:
    """
    The "Physical Screen": Combine Topology (Hubs+Leaves) with Anomalies.
    Rule: Group A (30 Topology) + Group B (Top 30 Anomalies).
    """
    # Group B: Top Anomalies
    anomalies = anomaly_scores.sort_values(ascending=False).head(top_n_anomalies).index.tolist()
    
    # Union
    pool = list(set(topology_candidates + anomalies))
    return pool
