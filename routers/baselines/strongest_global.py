#!/usr/bin/env python3
"""
Strongest single-model (global) router
Always selects the globally highest-accuracy model.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from routers.common import RouterBase


class StrongestGlobal(RouterBase):
    """Always select the single globally highest-accuracy model."""
    
    def __init__(self):
        self.best_j = None
    
    def fit(self, Y, C, meta, **kw):
        """Select the globally highest-accuracy model."""
        acc = Y.mean(axis=0)  # mean accuracy per model (column)
        self.best_j = int(np.argmax(acc))
        print(f"  Best model index: {self.best_j}, accuracy: {acc[self.best_j]:.4f}")
    
    def predict(self, X=None, meta=None):
        """Select the strongest model for all samples."""
        N = len(meta) if meta is not None else 1
        return np.full((N,), self.best_j, dtype=int)


