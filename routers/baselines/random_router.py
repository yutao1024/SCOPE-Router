#!/usr/bin/env python3
"""
Random router (random baseline)
Randomly selects a model.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from routers.common import RouterBase


class RandomRouter(RouterBase):
    """Randomly select a model."""
    
    def __init__(self, seed=42):
        self.K = None
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def fit(self, Y, C, meta, **kw):
        """Record the number of models."""
        self.K = Y.shape[1]
        print(f"  Random router: randomly selecting from {self.K} models")
    
    def predict(self, X=None, meta=None):
        """Randomly select a model."""
        N = len(meta) if meta is not None else 1
        return self.rng.randint(0, self.K, size=N)


