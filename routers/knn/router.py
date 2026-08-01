#!/usr/bin/env python3
"""
KNN（）
KNN
"""

import numpy as np
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import pickle

from routers.common import RouterBase

try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class KNNRouter(RouterBase):
    """
    KNN - KNN
    
    textvision embedding：
    - concat: 
    - average: 
    - weighted_average: 
    - normalize_concat: L2
    """
    
    def __init__(
        self, 
        n_neighbors: int = 5, 
        weights: str = 'distance', 
        metric: str = 'cosine',
        fusion_method: str = 'concat',
        text_weight: float = 0.5,
        text_encoder: str = "BAAI/bge-m3",
        vision_encoder: str = "facebook/dinov2-base"
    ):
        """
        Args:
            n_neighbors: KNNK
            weights:  ('uniform' or 'distance')
            metric:  ('cosine', 'euclidean', etc.)
            fusion_method:  ('concat', 'average', 'weighted_average', 'normalize_concat')
            text_weight: （weighted_average），[0, 1]
            text_encoder: （: "BAAI/bge-m3"）
            vision_encoder: （: "facebook/dinov2-base"）
        """
        if not HAS_SKLEARN:
            raise ImportError("Please install: pip install scikit-learn")
        
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.fusion_method = fusion_method
        self.text_weight = text_weight
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        
        self.knn = None
        self.scaler = None
        self.model_mapping = None  # {model_index: model_id}
        self.reverse_mapping = None  # {model_id: model_index}
        self.feature_extracted = False
    
    def fit(self, Y: np.ndarray, C: np.ndarray, meta, 
            X: Optional[np.ndarray] = None, 
            X_text: Optional[np.ndarray] = None,
            X_vision: Optional[np.ndarray] = None,
            model_mapping: Optional[Dict[int, str]] = None, **kwargs):
        """
        KNN
        
        Args:
            Y:  (N x K)
            C:  (N x K)
            meta: DataFrame
            X:  (N x D)，（）
            X_text:  (N x D_text)，X_vision
            X_vision:  (N x D_vision)，X_text
            model_mapping: ID {0: 'model1', 1: 'model2', ...}
        """
        import pandas as pd
        from routers.utils.fusion import fuse_embeddings
        
        # NOTE: (translated from Chinese)
        if X is not None:
            # NOTE: (translated from Chinese)
            pass
        elif X_text is not None and X_vision is not None:
            # NOTE: (translated from Chinese)
            X = fuse_embeddings(X_text, X_vision, 
                               method=self.fusion_method, 
                               text_weight=self.text_weight)
        elif 'embedding' in meta.columns:
            # meta
            X = np.vstack(meta['embedding'].values)
        elif 'text_embedding' in meta.columns and 'vision_embedding' in meta.columns:
            # meta
            X_text = np.vstack(meta['text_embedding'].values)
            X_vision = np.vstack(meta['vision_embedding'].values)
            X = fuse_embeddings(X_text, X_vision,
                               method=self.fusion_method,
                               text_weight=self.text_weight)
        else:
            raise ValueError(
                "KNN。：\n"
                "  1. X（）\n"
                "  2. X_textX_vision（）\n"
                "  3. meta'embedding'\n"
                "  4. meta'text_embedding''vision_embedding'"
            )
        
        # NOTE: (translated from Chinese)
        if model_mapping is None:
            # （label）
            if 'label' in meta.columns:
                unique_labels = sorted(meta['label'].unique())
                model_mapping = {i: str(label) for i, label in enumerate(unique_labels)}
            else:
                # NOTE: (translated from Chinese)
                K = Y.shape[1]
                model_mapping = {i: f'model_{i}' for i in range(K)}
        
        self.model_mapping = model_mapping
        self.reverse_mapping = {v: k for k, v in model_mapping.items()}
        
        # （Y）
        # **：**
        labels = []
        valid_indices = []  # training
        
        for i in range(len(Y)):
            # NOTE: (translated from Chinese)
            correct_models = np.where(Y[i] == 1)[0]
            if len(correct_models) > 0:
                # ，
                costs = C[i, correct_models]
                best_idx = correct_models[np.argmin(costs)]
                
                # （model_mapping）
                model_id = model_mapping.get(best_idx, f'model_{best_idx}')
                label = self.reverse_mapping.get(model_id, best_idx)
                labels.append(label)
                valid_indices.append(i)
            # else: ，
        
        labels = np.array(labels)
        valid_indices = np.array(valid_indices)
        
        # ，
        if len(valid_indices) < len(Y):
            print(f"  Filtering training samples: {len(Y)} -> {len(valid_indices)} "
                  f"(removed {len(Y) - len(valid_indices)} no correct model)")
            X = X[valid_indices]
            Y = Y[valid_indices]
            C = C[valid_indices]
            # ：meta，KNNXlabels
        
        # NOTE: (translated from Chinese)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # KNN
        self.knn = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric
        )
        self.knn.fit(X_scaled, labels)
        
        print(f"  KNNTraining complete: K={self.n_neighbors}, "
              f"Fusion method={self.fusion_method}, "
              f"={X.shape[1]}, Num models={len(model_mapping)}")
    
    def predict(self, X: Optional[np.ndarray] = None, 
                X_text: Optional[np.ndarray] = None,
                X_vision: Optional[np.ndarray] = None,
                meta=None) -> np.ndarray:
        """
        
        
        Args:
            X:  (N x D)，（）
            X_text:  (N x D_text)
            X_vision:  (N x D_vision)
            meta: DataFrame
        
        Returns:
             (N,)
        """
        import pandas as pd
        from routers.utils.fusion import fuse_embeddings
        
        # （fit）
        if X is not None:
            pass
        elif X_text is not None and X_vision is not None:
            X = fuse_embeddings(X_text, X_vision,
                               method=self.fusion_method,
                               text_weight=self.text_weight)
        elif meta is not None and 'embedding' in meta.columns:
            X = np.vstack(meta['embedding'].values)
        elif meta is not None and 'text_embedding' in meta.columns and 'vision_embedding' in meta.columns:
            X_text = np.vstack(meta['text_embedding'].values)
            X_vision = np.vstack(meta['vision_embedding'].values)
            X = fuse_embeddings(X_text, X_vision,
                               method=self.fusion_method,
                               text_weight=self.text_weight)
        else:
            raise ValueError("Features required：X、X_text+X_visionmetaembedding")
        
        # NOTE: (translated from Chinese)
        X_scaled = self.scaler.transform(X)
        
        # NOTE: (translated from Chinese)
        predictions = self.knn.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, X: Optional[np.ndarray] = None,
                     X_text: Optional[np.ndarray] = None,
                     X_vision: Optional[np.ndarray] = None,
                     meta=None) -> np.ndarray:
        """"""
        import pandas as pd
        from routers.utils.fusion import fuse_embeddings
        
        # （predict）
        if X is not None:
            pass
        elif X_text is not None and X_vision is not None:
            X = fuse_embeddings(X_text, X_vision,
                               method=self.fusion_method,
                               text_weight=self.text_weight)
        elif meta is not None and 'embedding' in meta.columns:
            X = np.vstack(meta['embedding'].values)
        elif meta is not None and 'text_embedding' in meta.columns and 'vision_embedding' in meta.columns:
            X_text = np.vstack(meta['text_embedding'].values)
            X_vision = np.vstack(meta['vision_embedding'].values)
            X = fuse_embeddings(X_text, X_vision,
                               method=self.fusion_method,
                               text_weight=self.text_weight)
        else:
            raise ValueError("Features required")
        
        X_scaled = self.scaler.transform(X)
        return self.knn.predict_proba(X_scaled)
    
    def save(self, path: str):
        """save"""
        with open(path, 'wb') as f:
            pickle.dump({
                'knn': self.knn,
                'scaler': self.scaler,
                'model_mapping': self.model_mapping,
                'reverse_mapping': self.reverse_mapping,
                'n_neighbors': self.n_neighbors,
                'weights': self.weights,
                'metric': self.metric,
                'fusion_method': self.fusion_method,
                'text_weight': self.text_weight,
                'text_encoder': self.text_encoder,
                'vision_encoder': self.vision_encoder
            }, f)
    
    @classmethod
    def load(cls, path: str):
        """load"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        router = cls(
            n_neighbors=data['n_neighbors'],
            weights=data.get('weights', 'distance'),
            metric=data.get('metric', 'cosine'),
            fusion_method=data.get('fusion_method', 'concat'),
            text_weight=data.get('text_weight', 0.5),
            text_encoder=data.get('text_encoder', 'BAAI/bge-m3'),
            vision_encoder=data.get('vision_encoder', 'facebook/dinov2-base')
        )
        router.knn = data['knn']
        router.scaler = data['scaler']
        router.model_mapping = data['model_mapping']
        router.reverse_mapping = data.get('reverse_mapping', {v: k for k, v in data['model_mapping'].items()})
        
        return router

