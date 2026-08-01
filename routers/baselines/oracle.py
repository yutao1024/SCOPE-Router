#!/usr/bin/env python3
"""
Oracle router (theoretical upper bound)
Always selects a correct model (if any); otherwise selects the cheapest model.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from routers.common import RouterBase


class Oracle(RouterBase):
    """Oracle router - theoretical performance upper bound."""
    
    def __init__(self):
        self.Y_train = None
        self.C_train = None
    
    def fit(self, Y, C, meta, **kw):
        """Store training data for later prediction."""
        self.Y_train = Y
        self.C_train = C
        print("  Oracle router: always selects the cheapest correct model")
    
    def predict(self, X=None, meta=None, Y_test=None, C_test=None):
        """
        Choose the cheapest correct model for each sample.
        
        Args:
            Y_test: Quality matrix for the test set (Oracle can \"see\" the answers)
            C_test: Cost matrix for the test set
        """
        if Y_test is None or C_test is None:
            raise ValueError("Oracle router requires Y_test and C_test")
        
        N, K = Y_test.shape
        choices = np.zeros(N, dtype=int)
        
        for i in range(N):
            # Find all correct models
            correct_models = np.where(Y_test[i] == 1)[0]
            
            if len(correct_models) > 0:
                # Select the cheapest correct model
                costs = C_test[i, correct_models]
                cheapest_idx = correct_models[np.argmin(costs)]
                choices[i] = cheapest_idx
            else:
                # If no correct model, select the cheapest one
                choices[i] = np.argmin(C_test[i])
        
        return choices


