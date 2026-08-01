#!/usr/bin/env python3
"""
PR-KNN: KNN

（Copeland），。
RouterEvalPRKnn-knn。
"""

import numpy as np
import pickle
import sys
from pathlib import Path
from typing import Optional, Dict
import warnings

from routers.common import RouterBase

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class PRKNNRouter(RouterBase):
    """
    PR-KNN: KNN
    
    : (Y, C)
    : 
      1. k
      2. (a,b)，i: π_i(a,b)
      3. Copeland: score_m = Σ_i w_i Σ_{b≠m} π_i(m,b)
      4. 
    """
    
    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = 'cosine',
        weights: str = 'distance',
        fusion_method: str = 'normalize_concat',
        text_weight: float = 0.5,
        verbose: int = 1,
        text_encoder: str = "BAAI/bge-m3",
        vision_encoder: str = "facebook/dinov2-base"
    ):
        """
        Args:
            n_neighbors: KNNK
            metric:  ('cosine', 'euclidean', etc.)
            weights:  ('uniform' or 'distance')
            fusion_method: 
            text_weight: （weighted_average）
            verbose:  (0=, 1=, >1=)
            text_encoder: （: "BAAI/bge-m3"）
            vision_encoder: （: "facebook/dinov2-base"）
        """
        if not HAS_SKLEARN:
            raise ImportError("Please install: pip install scikit-learn")
        
        self.k = n_neighbors
        self.metric = metric
        self.weights = weights
        self.fusion_method = fusion_method
        self.text_weight = text_weight
        self.verbose = verbose
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        
        self.nn = None
        self.scaler = None
        self.Xs_train = None  # training
        self.Y_train = None  # Train set
        self.C_train = None  # Train set
        self.model_mapping = None
        self.reverse_mapping = None
    
    def _fuse(
        self,
        meta,
        X: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_vision: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """（KNNRouter）"""
        from routers.utils.fusion import fuse_embeddings
        
        if X is not None:
            return X
        
        if X_text is not None and X_vision is not None:
            return fuse_embeddings(X_text, X_vision, self.fusion_method, self.text_weight)
        
        if meta is not None:
            if 'embedding' in meta.columns:
                return np.vstack(meta['embedding'].values)
            
            if 'text_embedding' in meta.columns and 'vision_embedding' in meta.columns:
                Xt = np.vstack(meta['text_embedding'].values)
                Xv = np.vstack(meta['vision_embedding'].values)
                return fuse_embeddings(Xt, Xv, self.fusion_method, self.text_weight)
        
        raise ValueError("Features required：X  (X_text, X_vision)  meta  embedding ")
    
    def _neighbor_weights(self, dists: np.ndarray) -> np.ndarray:
        """"""
        if self.weights == 'uniform':
            return np.ones_like(dists)
        # distance：1 / (ε + distance)
        return 1.0 / (1e-6 + dists)
    
    def fit(
        self,
        Y: np.ndarray,
        C: np.ndarray,
        meta,
        X: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_vision: Optional[np.ndarray] = None,
        model_mapping: Optional[Dict[int, str]] = None,
        **kwargs
    ):
        """
        PR-KNN
        
        Args:
            Y:  (N x K)
            C:  (N x K)
            meta: DataFrame
            X: ，
            X_text: ，
            X_vision: ，
            model_mapping: 
        """
        # NOTE: (translated from Chinese)
        Xf = self._fuse(meta, X, X_text, X_vision)
        
        # NOTE: (translated from Chinese)
        valid_mask = (Y.sum(axis=1) > 0)
        if valid_mask.sum() < len(Y):
            print(f"  Filtering training samples: {len(Y)} -> {valid_mask.sum()} "
                  f"(removed {len(Y) - valid_mask.sum()} no correct model)")
            Xf = Xf[valid_mask]
            Y = Y[valid_mask]
            C = C[valid_mask]
        
        # NOTE: (translated from Chinese)
        if self.verbose > 0:
            print("  ...")
        self.scaler = StandardScaler()
        self.Xs_train = self.scaler.fit_transform(Xf)
        
        # YC（）
        if self.verbose > 0:
            print(f"  Y/C (: {Y.shape})...")
        self.Y_train = Y.astype(np.int8)
        self.C_train = C.astype(np.float32)
        
        # KNN
        if self.verbose > 0:
            print(f"  KNN (K={self.k}, metric={self.metric})...")
            print(f"    samples: {len(self.Xs_train)}, : {self.Xs_train.shape[1]}")
            print("    ...", end="", flush=True)
        self.nn = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
        self.nn.fit(self.Xs_train)
        if self.verbose > 0:
            print(" ✓")
        
        # NOTE: (translated from Chinese)
        if model_mapping is None:
            K = Y.shape[1]
            model_mapping = {i: f'model_{i}' for i in range(K)}
        
        self.model_mapping = model_mapping
        self.reverse_mapping = {v: k for k, v in model_mapping.items()}
        
        print(f"  PR-KNNTraining complete: K={self.k}, Fusion method={self.fusion_method}, "
              f"={self.Xs_train.shape[1]}, Num models={len(model_mapping)}")
    
    def predict(
        self,
        X: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_vision: Optional[np.ndarray] = None,
        meta=None
    ) -> np.ndarray:
        """
        （）
        
        Returns:
             (N,)
        """
        Xf = self._fuse(meta, X, X_text, X_vision)
        Xs = self.scaler.transform(Xf)
        
        # k
        dists, indices = self.nn.kneighbors(Xs, n_neighbors=self.k, return_distance=True)
        # dists: (N_query, k), indices: (N_query, k)
        
        # NOTE: (translated from Chinese)
        weights = self._neighbor_weights(dists)  # (N_query, k)
        
        # YC
        Y_neighbors = self.Y_train[indices]  # (N_query, k, K)
        C_neighbors = self.C_train[indices]  # (N_query, k, K)
        
        N_query, k, K = Y_neighbors.shape
        
        # ，Copeland
        predictions = np.zeros(N_query, dtype=int)
        
        for qi in range(N_query):
            w = weights[qi].reshape(-1, 1)  # (k, 1)
            Y_k = Y_neighbors[qi]  # (k, K)
            C_k = C_neighbors[qi]  # (k, K)
            
            # Copeland：m，
            wins = np.zeros(K, dtype=float)
            
            for m in range(K):
                # b (b != m)
                for b in range(K):
                    if m == b:
                        continue
                    
                    # i(m, b)：π_i(m, b)
                    # π_i(m,b) = 1  Y_i,m > Y_i,b  (Y_i,m == Y_i,b  C_i,m < C_i,b)
                    better = (Y_k[:, m] > Y_k[:, b]) | (
                        (Y_k[:, m] == Y_k[:, b]) & (C_k[:, m] < C_k[:, b])
                    )
                    
                    # NOTE: (translated from Chinese)
                    wins[m] += (w[better, 0]).sum()
            
            # Copeland
            predictions[qi] = int(np.argmax(wins))
        
        return predictions
    
    def predict_proba(
        self,
        X: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_vision: Optional[np.ndarray] = None,
        meta=None
    ) -> np.ndarray:
        """
        （Copelandsoftmax）
        
        ：PR-KNN，Copeland
        """
        Xf = self._fuse(meta, X, X_text, X_vision)
        Xs = self.scaler.transform(Xf)
        
        dists, indices = self.nn.kneighbors(Xs, n_neighbors=self.k, return_distance=True)
        weights = self._neighbor_weights(dists)
        
        Y_neighbors = self.Y_train[indices]
        C_neighbors = self.C_train[indices]
        
        N_query, k, K = Y_neighbors.shape
        scores = np.zeros((N_query, K))
        
        for qi in range(N_query):
            w = weights[qi].reshape(-1, 1)
            Y_k = Y_neighbors[qi]
            C_k = C_neighbors[qi]
            
            for m in range(K):
                for b in range(K):
                    if m == b:
                        continue
                    better = (Y_k[:, m] > Y_k[:, b]) | (
                        (Y_k[:, m] == Y_k[:, b]) & (C_k[:, m] < C_k[:, b])
                    )
                    scores[qi, m] += (w[better, 0]).sum()
        
        # （softmax）
        # NOTE: (translated from Chinese)
        scores = scores / max(scores.max(), 1.0)  # 
        
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        
        return probs
    
    def save(self, path: str):
        """save"""
        with open(path, 'wb') as f:
            pickle.dump({
                'k': self.k,
                'metric': self.metric,
                'weights': self.weights,
                'fusion_method': self.fusion_method,
                'text_weight': self.text_weight,
                'verbose': self.verbose,
                'text_encoder': self.text_encoder,
                'vision_encoder': self.vision_encoder,
                'scaler': self.scaler,
                'Xs_train': self.Xs_train,
                'Y_train': self.Y_train,
                'C_train': self.C_train,
                'model_mapping': self.model_mapping,
                'reverse_mapping': self.reverse_mapping
            }, f)
    
    @classmethod
    def load(cls, path: str):
        """load"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        router = cls(
            n_neighbors=data['k'],
            metric=data['metric'],
            weights=data['weights'],
            fusion_method=data['fusion_method'],
            text_weight=data['text_weight'],
            verbose=data.get('verbose', 1),
            text_encoder=data.get('text_encoder', 'BAAI/bge-m3'),
            vision_encoder=data.get('vision_encoder', 'facebook/dinov2-base')
        )
        
        router.scaler = data['scaler']
        router.Xs_train = data['Xs_train']
        router.Y_train = data['Y_train']
        router.C_train = data['C_train']
        router.model_mapping = data['model_mapping']
        router.reverse_mapping = data.get('reverse_mapping')
        if router.reverse_mapping is None and router.model_mapping:
            router.reverse_mapping = {v: k for k, v in router.model_mapping.items()}
        
        # KNN
        router.nn = NearestNeighbors(n_neighbors=router.k, metric=router.metric)
        router.nn.fit(router.Xs_train)
        
        return router
    
    def __repr__(self):
        return (f"PRKNNRouter(K={self.k}, metric={self.metric}, "
                f"weights={self.weights}, fusion={self.fusion_method})")

