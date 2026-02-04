import numpy as np
import pandas as pd
from src.selection.layer1_shared import BaseMarketRegimeDetector, BaseAnomalyDetector

class RobustMarketRegimeDetector(BaseMarketRegimeDetector):
    """
    Robust version of MarketRegimeDetector for Batch Processing.
    Uses Numpy for correlation to handle sparse/infinite data without Pandas alignment crashes.
    """
    def compute_distance_matrix(self, returns: pd.DataFrame) -> np.ndarray:
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

class RobustAnomalyDetector(BaseAnomalyDetector):
    """
    Robust version of AnomalyDetector for Batch Processing.
    """
    def compute_features(self, prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
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
