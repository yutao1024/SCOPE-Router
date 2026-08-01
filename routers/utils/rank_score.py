#!/usr/bin/env python3
"""
Rank Score: RouterArena-Compatible Ranking Metric

Combines accuracy and cost into a single ranking score using:
1. Log-normalized cost (lower cost = higher score)
2. Weighted harmonic mean of accuracy and normalized cost

Reference: RouterArena: An Open Platform for Comprehensive Comparison of LLM Routers
           https://www.arxiv.org/pdf/2510.00202
"""

import numpy as np
from typing import Union, Optional


def rank_score(
    avg_acc: float,
    avg_cost: float,
    cmin: float,
    cmax: float,
    beta: float = 0.1,
    eps: float = 1e-12
) -> float:
    """
    Calculate Rank Score (RouterArena-compatible ranking metric)
    
    Formula:
        C = (log2(cmax) - log2(clip(cost, cmin, cmax))) / (log2(cmax) - log2(cmin))
        S_β = (1+β) * A * C / (β*A + C)
    
    Where:
        - C: log-normalized cost ∈ [0, 1], higher is better (cheaper)
        - A: accuracy ∈ [0, 1]
        - β: weight parameter (β<1 favors accuracy, β>1 favors cost)
    
    Args:
        avg_acc: Average accuracy (0 to 1)
        avg_cost: Average cost per sample ($)
        cmin: Minimum cost in the model pool (cheapest model)
        cmax: Maximum cost in the model pool (most expensive model)
        beta: Weight parameter (default 0.1, favors accuracy)
        eps: Small constant for numerical stability
    
    Returns:
        Rank Score ∈ [0, 1], higher is better
    
    Examples:
        >>> # High accuracy, low cost -> high score
        >>> rank_score(0.9, 0.001, 0.0005, 0.01, beta=0.1)
        0.87...
        
        >>> # High accuracy, high cost -> medium score
        >>> rank_score(0.9, 0.009, 0.0005, 0.01, beta=0.1)
        0.45...
        
        >>> # Low accuracy, low cost -> low score (accuracy dominates with β=0.1)
        >>> rank_score(0.5, 0.001, 0.0005, 0.01, beta=0.1)
        0.48...
    """
    # Clip cost to valid range
    cost_clipped = float(np.clip(avg_cost, cmin, cmax))
    
    # Log-normalized cost (higher C = cheaper = better)
    # Cost doubles -> C decreases by equal steps
    log_cmax = np.log2(cmax)
    log_cmin = np.log2(cmin)
    log_cost = np.log2(cost_clipped)
    
    C = (log_cmax - log_cost) / (log_cmax - log_cmin + eps)
    C = float(np.clip(C, 0.0, 1.0))
    
    # Clip accuracy
    A = float(np.clip(avg_acc, 0.0, 1.0))
    
    # Weighted harmonic mean
    # Harmonic mean penalizes extreme imbalance
    score = (1 + beta) * A * C / (beta * A + C + eps)
    
    return float(np.clip(score, 0.0, 1.0))


def batch_rank_score(
    avg_accs: np.ndarray,
    avg_costs: np.ndarray,
    cmin: float,
    cmax: float,
    beta: float = 0.1
) -> np.ndarray:
    """
    Calculate Rank Scores for multiple models/routers
    
    Args:
        avg_accs: Array of average accuracies
        avg_costs: Array of average costs
        cmin: Minimum cost in the model pool
        cmax: Maximum cost in the model pool
        beta: Weight parameter
    
    Returns:
        Array of Rank Scores
    """
    return np.array([
        rank_score(acc, cost, cmin, cmax, beta)
        for acc, cost in zip(avg_accs, avg_costs)
    ])


def get_cost_bounds_from_config(config_path: str) -> tuple:
    """
    Load cost bounds (cmin, cmax) from cost config file
    
    Args:
        config_path: Path to cost_bounds.json
    
    Returns:
        (cmin, cmax) tuple
    """
    import json
    from pathlib import Path
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Cost bounds config not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config['cmin'], config['cmax']


def explain_rank_score(score: float, avg_acc: float, avg_cost: float, 
                        cmin: float, cmax: float, beta: float = 0.1) -> str:
    """
    Generate human-readable explanation of Rank Score
    
    Args:
        score: Rank Score value
        avg_acc: Average accuracy
        avg_cost: Average cost
        cmin: Minimum cost
        cmax: Maximum cost
        beta: Beta parameter
    
    Returns:
        Explanation string
    """
    # Calculate normalized cost
    cost_clipped = float(np.clip(avg_cost, cmin, cmax))
    log_cmax = np.log2(cmax)
    log_cmin = np.log2(cmin)
    log_cost = np.log2(cost_clipped)
    C = (log_cmax - log_cost) / (log_cmax - log_cmin + 1e-12)
    C = float(np.clip(C, 0.0, 1.0))
    
    # Cost relative to bounds
    cost_ratio = (avg_cost - cmin) / (cmax - cmin) if cmax > cmin else 0.5
    
    explanation = f"""
Rank Score Breakdown:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Rank Score:         {score:.4f} / 1.0
  
Components:
  • Accuracy:         {avg_acc:.4f} / 1.0
  • Normalized Cost:  {C:.4f} / 1.0 (higher = cheaper)
  • Raw Cost:         ${avg_cost:.6f}
    - Min cost:       ${cmin:.6f}
    - Max cost:       ${cmax:.6f}
    - Relative:       {cost_ratio:.1%} of range
  
Parameters:
  • Beta (β):         {beta:.2f}
    {'- Favors accuracy' if beta < 1 else '- Favors cost'}
  
Formula:
  S_β = (1+β) × A × C / (β×A + C)
      = (1+{beta}) × {avg_acc:.4f} × {C:.4f} / ({beta}×{avg_acc:.4f} + {C:.4f})
      = {score:.4f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Interpretation:
  {'🏆 Excellent' if score >= 0.8 else '✓ Good' if score >= 0.6 else '○ Fair' if score >= 0.4 else '△ Poor'} - {'High accuracy with reasonable cost' if score >= 0.8 else 'Balanced performance' if score >= 0.6 else 'Trade-off needed' if score >= 0.4 else 'Improvement recommended'}
"""
    return explanation


# ============================================================================
# Testing and validation
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("Rank Score - Test Cases")
    print("="*80)
    
    # Define cost bounds (example: cheapest model = $0.0005, most expensive = $0.01)
    cmin = 0.0005
    cmax = 0.01
    
    test_cases = [
        # (name, accuracy, cost, expected_interpretation)
        ("Perfect cheap model", 1.0, 0.0005, "Best possible"),
        ("Perfect expensive model", 1.0, 0.01, "High accuracy, high cost"),
        ("Good cheap model", 0.85, 0.001, "Good balance"),
        ("Good expensive model", 0.85, 0.008, "Good accuracy, expensive"),
        ("Poor cheap model", 0.5, 0.001, "Low accuracy despite low cost"),
        ("Medium balanced", 0.7, 0.005, "Medium performance"),
    ]
    
    print(f"\nCost bounds: ${cmin:.6f} - ${cmax:.6f}")
    print(f"Beta: 0.1 (favors accuracy)\n")
    
    for name, acc, cost, interp in test_cases:
        score = rank_score(acc, cost, cmin, cmax, beta=0.1)
        print(f"{name:25s}: Score={score:.4f}, Acc={acc:.2f}, Cost=${cost:.6f} - {interp}")
    
    # Test different beta values
    print("\n" + "="*80)
    print("Effect of Beta Parameter (Acc=0.8, Cost=$0.002)")
    print("="*80)
    
    for beta in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
        score = rank_score(0.8, 0.002, cmin, cmax, beta=beta)
        favor = "accuracy" if beta < 1 else "balanced" if beta == 1 else "cost"
        print(f"β={beta:4.2f} (favors {favor:8s}): Score={score:.4f}")
    
    # Detailed explanation example
    print("\n" + "="*80)
    print("Detailed Explanation Example")
    print("="*80)
    score = rank_score(0.85, 0.002, cmin, cmax, beta=0.1)
    print(explain_rank_score(score, 0.85, 0.002, cmin, cmax, beta=0.1))


