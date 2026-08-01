#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OVR (One-vs-Rest) Router

，“/”。

 linear/mlp ：
- Router : router.py
- : concat/average/weighted_average/normalize_concat/only_text/only_image

:
- OVR  predict_proba  K  P(y_k=1|x)，。
"""

from __future__ import annotations

import pickle
from typing import Dict, Optional

import numpy as np

from routers.common import RouterBase

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


class OVRRouter(RouterBase):
    """
    OVR Router：K  LogisticRegression heads。
    """

    def __init__(
        self,
        *,
        fusion_method: str = "normalize_concat",
        text_weight: float = 0.5,
        text_encoder: str = "BAAI/bge-m3",
        vision_encoder: str = "facebook/dinov2-base",
        verbose: int = 1,
    ):
        if not HAS_SKLEARN:
            raise ImportError("Please install: pip install scikit-learn")

        self.fusion_method = fusion_method
        self.text_weight = float(text_weight)
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.verbose = int(verbose)

        self.scaler: Optional[StandardScaler] = None
        self.K: Optional[int] = None
        self.clfs = []  # list[Optional[LogisticRegression]]

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
        """（KNNRouter）"""
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

        # Hard-label only:
        #  Y（）， K  P(y_k=1|x)。
        # ：“”（ 0 ）。
        has_correct = (Y.sum(axis=1) > 0)
        valid = np.where(has_correct)[0]
        Y_target = Y[valid].astype(int) if len(valid) > 0 else np.zeros((0, Y.shape[1]), dtype=int)

        if self.verbose:
            print("  Label mode: hard label（ Y ）")
            if len(valid) < len(Y):
                print(
                    f"  Filtering training samples: {len(Y)} -> {len(valid)} "
                    f"(removed {len(Y) - len(valid)} no correct model)"
                )

        self.scaler = StandardScaler()
        Z = self.scaler.fit_transform(feats)
        Z = Z[valid]

        if model_mapping is None:
            model_mapping = {i: f"model_{i}" for i in range(Y.shape[1])}
        self.model_mapping = model_mapping
        self.reverse_mapping = {v: k for k, v in model_mapping.items()}

        self.K = Y.shape[1]
        self.clfs = []

        #  K 
        for k in range(self.K):
            yk = Y_target[:, k]
            if yk.max() == 0:
                if self.verbose:
                    print(f"  Warning:  {k} ({self.model_mapping[k]}) 。")
                clf = None
            else:
                clf = LogisticRegression(max_iter=1000, class_weight="balanced")
                clf.fit(Z, yk)
            self.clfs.append(clf)

        if self.verbose:
            print(
                f"  OVR Training complete: Fusion method={self.fusion_method}, "
                f"={Z.shape[1]}, Num models={len(model_mapping)}"
            )

        self.cost_min = C.min(axis=0)
        self.cost_max = C.max(axis=0)
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
        if self.scaler is None or self.K is None:
            raise RuntimeError("OVRRouter training： fit()")

        Z = self._fuse(meta, X=X, X_text=X_text, X_vision=X_vision)
        Z = self.scaler.transform(Z)

        N = Z.shape[0]
        K = self.K
        scores = np.zeros((N, K), dtype=float)
        for k in range(K):
            clf = self.clfs[k]
            if clf is None:
                scores[:, k] = 0.0
            else:
                scores[:, k] = clf.predict_proba(Z)[:, 1]
        return scores

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "fusion_method": self.fusion_method,
                    "text_weight": self.text_weight,
                    "scaler": self.scaler,
                    "K": self.K,
                    "clfs": self.clfs,
                    "model_mapping": self.model_mapping,
                    "reverse_mapping": self.reverse_mapping,
                    "cost_min": self.cost_min,
                    "cost_max": self.cost_max,
                    "text_encoder": self.text_encoder,
                    "vision_encoder": self.vision_encoder,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "OVRRouter":
        with open(path, "rb") as f:
            d = pickle.load(f)

        obj = cls(
            fusion_method=d["fusion_method"],
            text_weight=d["text_weight"],
            text_encoder=d.get("text_encoder", "BAAI/bge-m3"),
            vision_encoder=d.get("vision_encoder", "facebook/dinov2-base"),
            verbose=1,
        )
        obj.scaler = d["scaler"]
        obj.K = d["K"]
        obj.clfs = d["clfs"]
        obj.model_mapping = d["model_mapping"]
        obj.reverse_mapping = d["reverse_mapping"]
        obj.cost_min = d.get("cost_min")
        obj.cost_max = d.get("cost_max")
        return obj


# （； routers.ovr）
BinaryOVRRouter = OVRRouter


