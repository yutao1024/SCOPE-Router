#!/usr/bin/env python3
"""
Cheapest single-model (global) router
Always selects the lowest-cost model.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from routers.common import RouterBase


class CheapestGlobal(RouterBase):
    """Always select the single lowest-cost model."""
    
    def __init__(self):
        self.cheapest_j = None
    
    def fit(self, Y, C, meta, **kw):
        """Select the model with the lowest average cost."""
        mean_cost = C.mean(axis=0)
        self.cheapest_j = int(np.argmin(mean_cost))
        print(f"  Cheapest model index: {self.cheapest_j}, cost: ${mean_cost[self.cheapest_j]:.6f}/sample")
    
    def predict(self, X=None, meta=None):
        """Select the cheapest model for all samples."""
        N = len(meta) if meta is not None else 1
        return np.full((N,), self.cheapest_j, dtype=int)


