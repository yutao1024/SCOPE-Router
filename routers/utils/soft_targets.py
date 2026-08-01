#!/usr/bin/env python3
"""
Soft Target Construction for Cost-Aware Router Training

Based on multi-objective Lagrangian formulation:
- Only assigns probability mass to correct models
- Weights by cost using exp(-λ*c) or (c+ε)^(-λ)
- λ controls the spectrum from uniform (λ=0) to cheapest-only (λ→∞)
"""

import numpy as np
from typing import Tuple, Literal
from scipy.special import logsumexp


def build_soft_targets(
    Y: np.ndarray,
    C: np.ndarray,
    lam: float = 0.0,
    scheme: Literal['exp', 'power'] = 'exp',
    eps: float = 1e-12,
    fallback: Literal['cheapest', 'cost_soft'] = 'cheapest',
    cost_scale: float = 1.0
) -> np.ndarray:
    r"""
    Construct cost-weighted soft targets \(t_i^{(\lambda)}\), normalized over the correct set only.
    
    Mathematical formulation:
        t_i^{(λ)}(m) = [𝟙{Y_{i,m}=1} · g_λ(C_{i,m})] / Σ_{j: Y_{i,j}=1} g_λ(C_{i,j})
    
    where g_λ(·) is a monotonically decreasing function of cost:
        - scheme='exp': g_λ(c) = exp(-λ·c)
        - scheme='power': g_λ(c) = (c+ε)^(-λ)
    
    **Numerical Stability**: Uses log-space computation to avoid underflow when
    λ is large. All exponentials are computed via log-sum-exp trick
    
    Args:
        Y: (N, K) int/bool, is_correct ∈ {0,1}
        C: (N, K) float, cost ≥ 0
        lam: float ≥ 0, cost trade-off coefficient λ
            - λ=0: uniform distribution over correct models
            - λ→∞: one-hot to cheapest correct model
        scheme: 'exp' uses exp(-λc), 'power' uses (c+eps)^(-λ)
        eps: numerical stability constant
        fallback: when sample has no correct model:
            - 'cheapest': one-hot to argmin_m C_{i,m}
            - 'cost_soft': soft distribution over all models by exp(-λc)
        cost_scale: float > 0, scale factor for costs (default 1.0)
            - Useful when cost differences are small
            - Larger scale → more sensitive to cost differences
            - e.g., cost_scale=100 amplifies 17x difference to 1700x
    
    Returns:
        T: (N, K) float32, each row sums to 1 (soft target distribution)
    
    Examples:
        >>> Y = np.array([[1, 0, 1], [0, 1, 1]])
        >>> C = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
        >>> T = build_soft_targets(Y, C, lam=0.0)  # uniform over correct
        >>> print(T[0])  # [0.5, 0, 0.5] - uniform over models 0 and 2
        >>> T = build_soft_targets(Y, C, lam=10.0)  # cheapest correct
        >>> print(T[0])  # [1.0, 0, 0] - only cheapest correct (model 0)
    """
    Y = (Y > 0).astype(np.float64)
    C = C.astype(np.float64)
    N, K = Y.shape
    T = np.zeros((N, K), dtype=np.float64)
    
    # Apply cost scaling
    C_scaled = C * cost_scale
    
    # Mask for correct models
    M = Y  # (N, K)
    
    # Compute log weights for numerical stability (log-space)
    # This avoids underflow when exp(-lambda * C_scaled) becomes very small
    if scheme == 'exp':
        log_W = -lam * C_scaled  # (N, K), log of weights
    elif scheme == 'power':
        log_W = -lam * np.log(C_scaled + eps)
    else:
        raise ValueError(f"scheme must be 'exp' or 'power', got {scheme}")
    
    # Mask incorrect models by setting their log weights to -inf
    log_W_masked = np.where(M > 0, log_W, -np.inf)  # (N, K)
    
    # Log-sum-exp for stable normalization
    log_S = logsumexp(log_W_masked, axis=1, keepdims=True)  # (N, 1)
    
    # Samples with at least one correct model: normalize using log-space
    has_correct = ~np.isinf(log_S[:, 0])
    if np.any(has_correct):
        # T[i] = exp(log_W_masked[i] - log_S[i])
        T[has_correct] = np.exp(log_W_masked[has_correct] - log_S[has_correct])
    
    # Samples with no correct model: fallback strategy
    if np.any(~has_correct):
        idx = np.where(~has_correct)[0]
        if fallback == 'cheapest':
            # One-hot to globally cheapest model (use original C for selection)
            j_min = np.argmin(C[idx], axis=1)  # (n_bad,)
            T[idx, :] = 0.0
            T[idx, j_min] = 1.0
        elif fallback == 'cost_soft':
            # Soft distribution over all models by cost (use log-space for stability)
            if scheme == 'exp':
                log_Wb = -lam * C_scaled[idx]
            else:
                log_Wb = -lam * np.log(C_scaled[idx] + eps)
            log_Sb = logsumexp(log_Wb, axis=1, keepdims=True)
            T[idx] = np.exp(log_Wb - log_Sb)
        else:
            raise ValueError(f"fallback must be 'cheapest' or 'cost_soft', got {fallback}")
    
    return T.astype(np.float32)


def build_relevance_targets(
    Y: np.ndarray,
    C: np.ndarray,
    lam: float = 0.0,
    cost_scale: float = 1.0,
    fallback: Literal['zeros', 'cheapest'] = 'zeros',
) -> np.ndarray:
    """
    Construct unnormalized pairwise relevance targets for BCE/SigLIP training.

    For samples with at least one correct model:
        R[i, m] = 1{Y[i, m] = 1} * exp(-lambda * cost_scale * (C[i, m] - C_min_correct[i]))

    Thus the cheapest correct model has relevance 1, other correct models are
    in (0, 1], and wrong models are 0. Rows are intentionally not normalized.

    Args:
        Y: (N, K) int/bool, is_correct in {0,1}
        C: (N, K) float, cost >= 0
        lam: cost trade-off coefficient
        cost_scale: scale factor applied to cost differences
        fallback: for samples with no correct model:
            - 'zeros': all-zero relevance row
            - 'cheapest': set globally cheapest model to 1

    Returns:
        R: (N, K) float32, pairwise relevance in [0, 1]
    """
    Y = (Y > 0).astype(bool)
    C = C.astype(np.float64)
    R = np.zeros_like(C, dtype=np.float64)

    has_correct = Y.any(axis=1)
    if np.any(has_correct):
        correct_costs = np.where(Y[has_correct], C[has_correct], np.inf)
        min_correct = np.min(correct_costs, axis=1, keepdims=True)
        if np.isinf(float(lam)):
            rel = (C[has_correct] == min_correct).astype(np.float64)
        else:
            rel = np.exp(-float(lam) * float(cost_scale) * (C[has_correct] - min_correct))
        R[has_correct] = np.where(Y[has_correct], rel, 0.0)

    if np.any(~has_correct):
        idx = np.where(~has_correct)[0]
        if fallback == 'cheapest':
            j_min = np.argmin(C[idx], axis=1)
            R[idx, j_min] = 1.0
        elif fallback != 'zeros':
            raise ValueError(f"fallback must be 'zeros' or 'cheapest', got {fallback}")

    return np.clip(R, 0.0, 1.0).astype(np.float32)


def build_hard_targets_cheapest_correct(Y: np.ndarray, C: np.ndarray) -> np.ndarray:
    r"""
    Construct hard targets: choose the cheapest correct model per sample (one-hot).
    
    Equivalent to the limiting case of build_soft_targets as \(\lambda \to \infty\).
    
    Args:
        Y: (N, K) int/bool, is_correct ∈ {0,1}
        C: (N, K) float, cost ≥ 0
    
    Returns:
        labels: (N,) int, model indices ∈ {0, ..., K-1}
    """
    Y = (Y > 0).astype(bool)
    C = C.astype(np.float64)
    N, K = Y.shape
    
    labels = np.zeros(N, dtype=np.int32)
    
    for i in range(N):
        correct_mask = Y[i]
        if correct_mask.any():
            # Among correct models, choose the cheapest
            costs_correct = np.where(correct_mask, C[i], np.inf)
            labels[i] = np.argmin(costs_correct)
        else:
            # No correct model: choose globally cheapest
            labels[i] = np.argmin(C[i])
    
    return labels


def analyze_soft_targets(T: np.ndarray, Y: np.ndarray, C: np.ndarray) -> dict:
    """
    Analyze statistical properties of soft targets.
    
    Args:
        T: (N, K) soft targets
        Y: (N, K) correctness matrix
        C: (N, K) cost matrix
    
    Returns:
        stats: dictionary with analysis metrics
    """
    N, K = T.shape
    Y_bool = (Y > 0).astype(bool)
    
    stats = {}
    
    # Entropy (higher = more uniform)
    entropy = -np.sum(T * np.log(T + 1e-12), axis=1)
    stats['mean_entropy'] = float(entropy.mean())
    stats['std_entropy'] = float(entropy.std())
    stats['max_entropy'] = float(np.log(K))  # theoretical max
    
    # Effective number of models (exp(entropy))
    eff_K = np.exp(entropy)
    stats['mean_effective_K'] = float(eff_K.mean())
    
    # Concentration on cheapest correct
    cheapest_correct_indices = build_hard_targets_cheapest_correct(Y, C)
    prob_on_cheapest = T[np.arange(N), cheapest_correct_indices]
    stats['mean_prob_cheapest_correct'] = float(prob_on_cheapest.mean())
    stats['median_prob_cheapest_correct'] = float(np.median(prob_on_cheapest))
    
    # Average number of models with >1% probability
    active_models = (T > 0.01).sum(axis=1)
    stats['mean_active_models'] = float(active_models.mean())
    
    # Samples with no correct model
    has_no_correct = (~Y_bool.any(axis=1))
    stats['pct_no_correct'] = float(100.0 * has_no_correct.mean())
    
    return stats


def print_soft_target_analysis(lam: float, stats: dict):
    """Print soft-target analysis results."""
    print(f"\n📊 Soft Target Analysis (λ={lam:.2f}):")
    print(f"  • Mean entropy: {stats['mean_entropy']:.3f} (max={stats['max_entropy']:.3f})")
    print(f"  • Effective #models: {stats['mean_effective_K']:.2f}")
    print(f"  • Mean prob on cheapest correct: {stats['mean_prob_cheapest_correct']:.3f}")
    print(f"  • Mean active models (>1%): {stats['mean_active_models']:.1f}")
    print(f"  • Samples w/ no correct: {stats['pct_no_correct']:.1f}%")


# ============================================================================
# Example usage and validation
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("Soft Target Construction - Example & Validation")
    print("="*80)
    
    # Create synthetic data
    np.random.seed(42)
    N, K = 100, 5
    Y = np.random.randint(0, 2, (N, K))
    C = np.random.uniform(0.001, 0.01, (N, K))
    
    # Test different λ values
    lambdas = [0.0, 1.0, 5.0, 10.0, 50.0]
    
    for lam in lambdas:
        T = build_soft_targets(Y, C, lam=lam, scheme='exp')
        stats = analyze_soft_targets(T, Y, C)
        print_soft_target_analysis(lam, stats)
        
        # Show first 3 samples
        print(f"\n  Sample distributions (first 3 rows):")
        for i in range(min(3, N)):
            correct_str = "".join([str(int(y)) for y in Y[i]])
            dist_str = " ".join([f"{t:.3f}" for t in T[i]])
            print(f"    Sample {i} (correct={correct_str}): [{dist_str}]")
    
    print("\n" + "="*80)
    print("✅ Validation complete!")
    print("="*80)
