import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Tuple

class BaseMarketRegimeDetector:
    """
    Base class for Market Regime Detection using MST.
    Shared logic for MST construction, NTL calculation, and Regime Classification.
    Subclasses must implement compute_distance_matrix.
    """
    
    def compute_distance_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement compute_distance_matrix")

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

class BaseAnomalyDetector:
    """
    Base class for Anomaly Detection.
    Shared logic for Isolation Forest.
    Subclasses must implement compute_features.
    """
    
    def compute_features(self, prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement compute_features")

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
