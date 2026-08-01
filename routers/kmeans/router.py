#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KMeans (Prototype) Router

“/prototype()”： prototype 。

 linear/mlp ：
- Router : router.py
- : concat/average/weighted_average/normalize_concat/only_text/only_image

 “soft label / train_lambda=0”：
-  soft_targets + train_lambda ， soft target  argmax 。
- ， λ=0 ：**uniform over correct models**。
- ：prototype_k  Y[:,k]=1 （ λ≠0， C ）。
"""

from __future__ import annotations

import pickle
from typing import Dict, Optional

import numpy as np

from routers.common import RouterBase

try:
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


class KMeansRouter(RouterBase):
    """
    Prototype-based router:
    - fit:  k  prototype（ Y[:,k]=1 ）
    - predict: argmax cosine similarity to prototypes
    """

    def __init__(
        self,
        *,
        fusion_method: str = "normalize_concat",
        text_weight: float = 0.5,
        normalize: bool = True,
        lambda_cost: float = 0.0,
        text_encoder: str = "BAAI/bge-m3",
        vision_encoder: str = "facebook/dinov2-base",
        verbose: int = 1,
    ):
        if not HAS_SKLEARN:
            raise ImportError("Please install: pip install scikit-learn")

        self.fusion_method = fusion_method
        self.text_weight = float(text_weight)
        self.normalize = bool(normalize)
        self.lambda_cost = float(lambda_cost)  # （predict ）
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.verbose = int(verbose)

        self.scaler: Optional[StandardScaler] = None
        self.prototypes: Optional[np.ndarray] = None  # (K, D)
        self.K: Optional[int] = None

        self.model_mapping: Optional[Dict[int, str]] = None
        self.reverse_mapping: Optional[Dict[str, int]] = None

        # （predict ）
        self.cost_min = None
        self.cost_max = None

    def _fuse(
        self,
        meta,
        X: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_vision: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        from routers.utils.fusion import fuse_embeddings

        if X is not None:
            return X
        if X_text is not None and X_vision is not None:
            return fuse_embeddings(X_text, X_vision, self.fusion_method, self.text_weight)
        if meta is not None:
            if "embedding" in meta.columns:
                return np.vstack(meta["embedding"].values)
            if "text_embedding" in meta.columns and "vision_embedding" in meta.columns:
                Xt = np.vstack(meta["text_embedding"].values)
                Xv = np.vstack(meta["vision_embedding"].values)
                return fuse_embeddings(Xt, Xv, self.fusion_method, self.text_weight)
        raise ValueError("Features required：X  (X_text, X_vision)  meta  embedding ")

    def fit(
        self,
        Y: np.ndarray,
        C: np.ndarray,
        meta,
        *,
        X: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_vision: Optional[np.ndarray] = None,
        model_mapping: Optional[Dict[int, str]] = None,
        **kwargs,
    ):
        feats = self._fuse(meta, X=X, X_text=X_text, X_vision=X_vision)

        # NOTE: (translated from Chinese)
        has_correct = (Y.sum(axis=1) > 0)
        valid = np.where(has_correct)[0]
        if len(valid) < len(Y) and self.verbose:
            print(
                f"  Filtering training samples: {len(Y)} -> {len(valid)} "
                f"(removed {len(Y) - len(valid)} no correct model)"
            )

        Yv = (Y[valid] > 0).astype(np.int32)
        Cv = C[valid]
        feats = feats[valid]

        if model_mapping is None:
            model_mapping = {i: f"model_{i}" for i in range(Y.shape[1])}
        self.model_mapping = model_mapping
        self.reverse_mapping = {v: k for k, v in model_mapping.items()}

        self.scaler = StandardScaler()
        Z = self.scaler.fit_transform(feats)
        if self.normalize:
            Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)

        K = Y.shape[1]
        self.K = K
        self.prototypes = np.zeros((K, Z.shape[1]), dtype=np.float32)

        # λ=0 (uniform over correct): prototype_k  Y[:,k]=1 
        for k in range(K):
            idx = np.where(Yv[:, k] > 0)[0]
            if len(idx) == 0:
                self.prototypes[k] = Z.mean(axis=0)
            else:
                self.prototypes[k] = Z[idx].mean(axis=0)

        if self.normalize:
            self.prototypes = self.prototypes / np.maximum(np.linalg.norm(self.prototypes, axis=1, keepdims=True), 1e-12)

        if self.verbose:
            print(
                f"  KMeans Training complete: Fusion method={self.fusion_method}, "
                f"={Z.shape[1]}, Num models={len(model_mapping)} (label=softλ0/uniform-over-correct)"
            )

        self.cost_min = Cv.min(axis=0)
        self.cost_max = Cv.max(axis=0)
        return self

    def predict(
        self,
        *,
        X: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_vision: Optional[np.ndarray] = None,
        meta=None,
    ) -> np.ndarray:
        scores = self.predict_proba(X=X, X_text=X_text, X_vision=X_vision, meta=meta)
        return scores.argmax(axis=1)

    def predict_proba(
        self,
        *,
        X: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_vision: Optional[np.ndarray] = None,
        meta=None,
    ) -> np.ndarray:
        if self.prototypes is None or self.scaler is None:
            raise RuntimeError("KMeansRouter training： fit()")

        feats = self._fuse(meta, X=X, X_text=X_text, X_vision=X_vision)
        Z = self.scaler.transform(feats)
        if self.normalize:
            Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)

        logits = Z @ self.prototypes.T
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "fusion_method": self.fusion_method,
                    "text_weight": self.text_weight,
                    "normalize": self.normalize,
                    "lambda_cost": self.lambda_cost,
                    "text_encoder": self.text_encoder,
                    "vision_encoder": self.vision_encoder,
                    "scaler": self.scaler,
                    "prototypes": self.prototypes,
                    "K": self.K,
                    "model_mapping": self.model_mapping,
                    "reverse_mapping": self.reverse_mapping,
                    "cost_min": self.cost_min,
                    "cost_max": self.cost_max,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "KMeansRouter":
        with open(path, "rb") as f:
            d = pickle.load(f)

        obj = cls(
            fusion_method=d["fusion_method"],
            text_weight=d["text_weight"],
            normalize=d["normalize"],
            lambda_cost=d.get("lambda_cost", 0.0),
            text_encoder=d.get("text_encoder", "BAAI/bge-m3"),
            vision_encoder=d.get("vision_encoder", "facebook/dinov2-base"),
            verbose=1,
        )
        obj.scaler = d["scaler"]
        obj.prototypes = d["prototypes"]
        obj.K = d["K"]
        obj.model_mapping = d["model_mapping"]
        obj.reverse_mapping = d["reverse_mapping"]
        obj.cost_min = d.get("cost_min")
        obj.cost_max = d.get("cost_max")
        return obj


# Backward-compatible alias (old name used in scripts/older code)
KmeansClassifierRouter = KMeansRouter


