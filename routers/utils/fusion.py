#!/usr/bin/env python3
"""
Feature fusion utilities - non-trainable fusion methods.
Supports multiple strategies for fusing text and vision embeddings.
"""

import numpy as np
from typing import Literal


def fuse_embeddings(
    text_emb: np.ndarray,
    vision_emb: np.ndarray,
    method: Literal[
        'concat',
        'average',
        'weighted_average',
        'normalize_concat',
        'concat_interaction',
        'normalize_concat_interaction',
        'only_text',
        'only_image',
    ] = 'concat',
    text_weight: float = 0.5
) -> np.ndarray:
    """
    Fuse text and vision features.
    
    Args:
        text_emb: Text features (N, D_text)
        vision_emb: Vision features (N, D_vision)
        method: Fusion method
            - 'concat': concatenate [text, vision]
            - 'average': average (if dims differ, project to min dim first)
            - 'weighted_average': weighted average (requires same dim; otherwise uses min dim)
            - 'normalize_concat': L2-normalize then concatenate
            - 'concat_interaction': concatenate plus t*v and |t-v| on the shared prefix
            - 'normalize_concat_interaction': L2-normalize each modality, then add interaction features
            - 'only_text': use text features only (single-modality)
            - 'only_image': use vision features only (single-modality)
        text_weight: Text weight (for weighted_average), in [0, 1]
    
    Returns:
        Fused features (N, D_fused)
    """
    assert text_emb.shape[0] == vision_emb.shape[0], "Number of samples must match"
    
    if method == 'concat':
        # Concatenate
        fused = np.hstack([text_emb, vision_emb])
        
    elif method == 'average':
        # Average: if dims differ, project to smaller dim
        min_dim = min(text_emb.shape[1], vision_emb.shape[1])
        if text_emb.shape[1] != vision_emb.shape[1]:
            # Simple truncation to the same dim
            text_proj = text_emb[:, :min_dim]
            vision_proj = vision_emb[:, :min_dim]
            fused = (text_proj + vision_proj) / 2.0
        else:
            fused = (text_emb + vision_emb) / 2.0
        
    elif method == 'weighted_average':
        # Weighted average: requires matching dims (or use min dim)
        if text_emb.shape[1] != vision_emb.shape[1]:
            min_dim = min(text_emb.shape[1], vision_emb.shape[1])
            text_proj = text_emb[:, :min_dim]
            vision_proj = vision_emb[:, :min_dim]
        else:
            text_proj = text_emb
            vision_proj = vision_emb
        
        fused = text_weight * text_proj + (1 - text_weight) * vision_proj
        
    elif method == 'normalize_concat':
        # L2-normalize then concatenate
        def normalize(vec):
            norm = np.linalg.norm(vec, axis=1, keepdims=True)
            norm = np.where(norm > 0, norm, 1.0)
            return vec / norm
        
        text_norm = normalize(text_emb)
        vision_norm = normalize(vision_emb)
        fused = np.hstack([text_norm, vision_norm])

    elif method == 'concat_interaction':
        min_dim = min(text_emb.shape[1], vision_emb.shape[1])
        text_shared = text_emb[:, :min_dim]
        vision_shared = vision_emb[:, :min_dim]
        fused = np.hstack([
            text_emb,
            vision_emb,
            text_shared * vision_shared,
            np.abs(text_shared - vision_shared),
        ])

    elif method == 'normalize_concat_interaction':
        def normalize(vec):
            norm = np.linalg.norm(vec, axis=1, keepdims=True)
            norm = np.where(norm > 0, norm, 1.0)
            return vec / norm

        text_norm = normalize(text_emb)
        vision_norm = normalize(vision_emb)
        min_dim = min(text_norm.shape[1], vision_norm.shape[1])
        text_shared = text_norm[:, :min_dim]
        vision_shared = vision_norm[:, :min_dim]
        fused = np.hstack([
            text_norm,
            vision_norm,
            text_shared * vision_shared,
            np.abs(text_shared - vision_shared),
        ])

    elif method == 'only_text':
        assert text_emb is not None, "only_text requires text_emb"
        fused = text_emb

    elif method == 'only_image':
        assert vision_emb is not None, "only_image requires vision_emb"
        fused = vision_emb

    else:
        raise ValueError(f"Unknown fusion method: {method}")
    
    return fused


def get_fusion_dimension(
    text_dim: int,
    vision_dim: int,
    method: Literal[
        'concat',
        'average',
        'weighted_average',
        'normalize_concat',
        'concat_interaction',
        'normalize_concat_interaction',
        'only_text',
        'only_image',
    ] = 'concat'
) -> int:
    """Get fused feature dimension."""
    if method in ['concat', 'normalize_concat']:
        return text_dim + vision_dim
    elif method in ['concat_interaction', 'normalize_concat_interaction']:
        return text_dim + vision_dim + 2 * min(text_dim, vision_dim)
    elif method in ['average', 'weighted_average']:
        return min(text_dim, vision_dim)
    elif method == 'only_text':
        return text_dim
    elif method == 'only_image':
        return vision_dim
    else:
        raise ValueError(f"Unknown fusion method: {method}")
