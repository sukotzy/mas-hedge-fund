import numpy as np
import pandas as pd
from src.selection.layer1_shared import BaseMarketRegimeDetector, BaseAnomalyDetector

class MarketRegimeDetector(BaseMarketRegimeDetector):
    """
    Real-Time implementation of Market Regime Detection.
    Uses Standard Pandas Correlation (optimized for dense, aligned daily data).
    """
    
    def compute_distance_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Calculate distance metric d_ij = sqrt(2(1 - rho_ij)).
        Input: Log Returns DataFrame (Date x Ticker).
        """
        # Pearson correlation (Pandas is fast for dense single-day history)
        corr_matrix = returns.corr().values
        
        # Clip to handle floating point errors slightly outside [-1, 1]
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
        
        # Distance transformation
        dist_matrix = np.sqrt(2 * (1 - corr_matrix))
        
        # Ensure diagonal is 0
        np.fill_diagonal(dist_matrix, 0)
        
        return dist_matrix

class AnomalyDetector(BaseAnomalyDetector):
    """
    Real-Time implementation of Anomaly Detection.
    Uses Standard Pandas corrwith for vectorized correlation.
    """
    
    def compute_features(self, prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for anomaly detection:
        1. Idiosyncratic Volatility
        2. Volume-Price Divergence
        3. Momentum
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
        # Vectorized correlation per column using Pandas (Optimized for Realtime)
        features['vp_divergence'] = returns.corrwith(vol_changes)
        
        # 3. Momentum (Last 20 days)
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
