#!/usr/bin/env python3
"""
Strongest single-model (per-dataset) router
Selects the highest-accuracy model for each dataset.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from routers.common import RouterBase


class StrongestPerDataset(RouterBase):
    """Select the strongest model for each dataset."""
    
    def __init__(self):
        self.map_ds2j = {}
    
    def fit(self, Y, C, meta: pd.DataFrame, **kw):
        """Compute the strongest model for each dataset."""
        for ds in meta['dataset'].unique():
            mask = (meta['dataset'] == ds).values
            idx = np.where(mask)[0]
            if len(idx) > 0:
                acc = Y[idx].mean(axis=0)
                self.map_ds2j[ds] = int(np.argmax(acc))
                print(f"  Dataset {ds}: best model index={self.map_ds2j[ds]}, accuracy={acc[self.map_ds2j[ds]]:.4f}")
    
    def predict(self, X=None, meta: pd.DataFrame = None):
        """Select the corresponding strongest model based on dataset."""
        fallback = list(self.map_ds2j.values())[0] if self.map_ds2j else 0
        return np.array([
            self.map_ds2j.get(ds, fallback) 
            for ds in meta['dataset']
        ], dtype=int)


