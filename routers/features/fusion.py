#!/usr/bin/env python3
"""
Feature fusion module - fuse text and vision embeddings

Supported fusion methods:
- concat: simple concatenation [t; v]
- concat_with_interaction: concat + Hadamard product + absolute difference [t; v; t⊙v; |t-v|]
- gated: gated fusion (requires learnable parameters; for trainable routers)
- weighted: weighted concatenation
"""

import numpy as np
from typing import Optional


class FeatureFusion:
    """
    Feature fuser for text and vision features.

    Supports multiple fusion methods.
    """
    
    def __init__(
        self,
        method: str = "concat",
        include_interaction: bool = True,
        text_weight: float = 0.5,
        vision_weight: float = 0.5,
        normalize: bool = True
    ):
        """
        Args:
            method: Fusion method ("concat", "concat_with_interaction", "weighted")
            include_interaction: Whether to include interaction features (Hadamard, abs diff)
            text_weight: Text weight (for weighted)
            vision_weight: Vision weight (for weighted)
            normalize: Whether to normalize the fused features
        """
        self.method = method
        self.include_interaction = include_interaction
        self.text_weight = text_weight
        self.vision_weight = vision_weight
        self.normalize = normalize
        
        if method not in AVAILABLE_FUSION_METHODS:
            raise ValueError(
                f"Unknown fusion method: {method}\n"
                f"Available methods: {list(AVAILABLE_FUSION_METHODS.keys())}"
            )
    
    def fuse(
        self,
        text_embeddings: np.ndarray,
        vision_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Fuse text and vision features.
        
        Args:
            text_embeddings: (N, D_t) text features
            vision_embeddings: (N, D_v) vision features
        
        Returns:
            fused_embeddings: (N, D_fused) fused features
        """
        N = text_embeddings.shape[0]
        
        if self.method == "concat":
            # Simple concatenation [t; v]
            if self.include_interaction:
                # Extended: [t; v; t⊙v; |t-v|]
                min_dim = min(text_embeddings.shape[1], vision_embeddings.shape[1])
                text_trunc = text_embeddings[:, :min_dim]
                vision_trunc = vision_embeddings[:, :min_dim]
                
                hadamard = text_trunc * vision_trunc
                diff = np.abs(text_trunc - vision_trunc)
                
                fused = np.concatenate([
                    text_embeddings,
                    vision_embeddings,
                    hadamard,
                    diff
                ], axis=1)
            else:
                fused = np.concatenate([text_embeddings, vision_embeddings], axis=1)
        
        elif self.method == "concat_with_interaction":
            # Extended: [t; v; t⊙v; |t-v|]
            min_dim = min(text_embeddings.shape[1], vision_embeddings.shape[1])
            text_trunc = text_embeddings[:, :min_dim]
            vision_trunc = vision_embeddings[:, :min_dim]
            
            hadamard = text_trunc * vision_trunc
            diff = np.abs(text_trunc - vision_trunc)
            
            fused = np.concatenate([
                text_embeddings,
                vision_embeddings,
                hadamard,
                diff
            ], axis=1)
        
        elif self.method == "weighted":
            # Weighted concatenation
            weighted_text = text_embeddings * self.text_weight
            weighted_vision = vision_embeddings * self.vision_weight
            fused = np.concatenate([weighted_text, weighted_vision], axis=1)
        
        elif self.method == "gated":
            # Gated fusion (simplified; should be learnable in practice)
            # z = [α·v; (1-α)·t]
            # Simplification: fixed weight (should be learned via an MLP)
            alpha = 0.5  # should be learned
            fused = np.concatenate([
                alpha * vision_embeddings,
                (1 - alpha) * text_embeddings
            ], axis=1)
        
        elif self.method == "hadamard_only":
            # Hadamard product only (dims must match)
            min_dim = min(text_embeddings.shape[1], vision_embeddings.shape[1])
            text_trunc = text_embeddings[:, :min_dim]
            vision_trunc = vision_embeddings[:, :min_dim]
            fused = text_trunc * vision_trunc
        
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")
        
        # Normalize
        if self.normalize:
            # Z-score normalization
            mean = fused.mean(axis=0, keepdims=True)
            std = fused.std(axis=0, keepdims=True) + 1e-8
            fused = (fused - mean) / std
        
        return fused
    
    @property
    def output_dimension(
        self,
        text_dim: int,
        vision_dim: int
    ) -> int:
        """Compute fused output dimension."""
        if self.method == "concat":
            if self.include_interaction:
                min_dim = min(text_dim, vision_dim)
                return text_dim + vision_dim + min_dim + min_dim  # + hadamard + diff
            else:
                return text_dim + vision_dim
        
        elif self.method == "concat_with_interaction":
            min_dim = min(text_dim, vision_dim)
            return text_dim + vision_dim + min_dim + min_dim
        
        elif self.method == "weighted":
            return text_dim + vision_dim
        
        elif self.method == "gated":
            return text_dim + vision_dim  # or max(text_dim, vision_dim) * 2
        
        elif self.method == "hadamard_only":
            return min(text_dim, vision_dim)
        
        else:
            return text_dim + vision_dim  # default


# Available fusion methods
AVAILABLE_FUSION_METHODS = {
    "concat": {
        "description": "Simple concatenation [t; v]",
        "output_dim": "text_dim + vision_dim",
    },
    "concat_with_interaction": {
        "description": "Concat + interaction features [t; v; t⊙v; |t-v|]",
        "output_dim": "text_dim + vision_dim + 2*min(text_dim, vision_dim)",
    },
    "weighted": {
        "description": "Weighted concatenation",
        "output_dim": "text_dim + vision_dim",
    },
    "gated": {
        "description": "Gated fusion (simplified)",
        "output_dim": "text_dim + vision_dim",
    },
    "hadamard_only": {
        "description": "Hadamard product only",
        "output_dim": "min(text_dim, vision_dim)",
    },
}


def list_available_fusion_methods():
    """List all available fusion methods."""
    print("=" * 80)
    print("Available feature fusion methods:")
    print("=" * 80)
    for method, info in AVAILABLE_FUSION_METHODS.items():
        print(f"  {method:30s} {info['description']}")
        print(f"   Output dim: {info['output_dim']}")


if __name__ == "__main__":
    list_available_fusion_methods()

