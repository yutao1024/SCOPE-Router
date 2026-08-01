#!/usr/bin/env python3
"""
VLM Router Benchmark - Router base class and utility functions
"""

from typing import Dict
import numpy as np
import pandas as pd


class RouterBase:
    """Base class for all router implementations."""
    
    def fit(self, Y: np.ndarray, C: np.ndarray, meta: pd.DataFrame, **kwargs):
        """
        Train the router.
        
        Args:
            Y: Quality matrix (N x K), is_correct indicator
            C: Cost matrix (N x K), $/sample
            meta: Metadata DataFrame
        """
        raise NotImplementedError
    
    def predict(self, X=None, meta: pd.DataFrame = None) -> np.ndarray:
        """
        Predict routing decisions.
        
        Args:
            X: Features (optional; some baselines don't need it)
            meta: Metadata
        
        Returns:
            Selected model indices (N-dimensional integer array)
        """
        raise NotImplementedError
    
    def name(self) -> str:
        """Return the router name."""
        return self.__class__.__name__


def quality(Y: np.ndarray, choose: np.ndarray) -> float:
    """
    Compute the average quality actually achieved by the router.
    
    Args:
        Y: Quality matrix (N x K)
        choose: Selected model indices (N,)
    
    Returns:
        Mean is_correct (0-1)
    """
    return float(Y[np.arange(Y.shape[0]), choose].mean())


def cost(C: np.ndarray, choose: np.ndarray) -> float:
    """
    Compute the average cost actually incurred by the router.
    
    Args:
        C: Cost matrix (N x K)
        choose: Selected model indices (N,)
    
    Returns:
        Mean cost ($/sample)
    """
    return float(C[np.arange(C.shape[0]), choose].mean())


