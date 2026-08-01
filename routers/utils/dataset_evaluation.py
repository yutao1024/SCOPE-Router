#!/usr/bin/env python3
"""
Per-dataset evaluation utilities.
Used by all router types for dataset-grouped evaluation.
"""

import numpy as np
import pandas as pd
from typing import Optional
from routers.utils.rank_score import rank_score


def evaluate_by_dataset(
    router,
    Y_split: np.ndarray,
    C_split: np.ndarray,
    meta_split: pd.DataFrame,
    models: list,
    text_embeddings: Optional[np.ndarray],
    vision_embeddings: Optional[np.ndarray],
    split_embedding_indices: np.ndarray,
    cmin: float,
    cmax: float,
    beta: float = 0.1,
    query_embeddings: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Evaluate router performance grouped by dataset.
    
    Args:
        router: Trained router
        Y_split: Label matrix [num_samples, num_models]
        C_split: Cost matrix [num_samples, num_models]
        meta_split: Metadata DataFrame (must include 'dataset' column)
        models: Model list
        text_embeddings: Text embeddings [num_total_samples, dim] (optional)
        vision_embeddings: Vision embeddings [num_total_samples, dim] (optional)
        split_embedding_indices: Indices of split samples in the embeddings arrays
        cmin: Minimum cost (for rank_score)
        cmax: Maximum cost (for rank_score)
        beta: Beta parameter for rank_score (default 0.1)
    
    Returns:
        DataFrame with columns: dataset, accuracy, avg_cost, rank_score, num_correct, num_samples
    """
    print("\n📊 Evaluating performance by dataset...")
    
    # Check for dataset column
    if 'dataset' not in meta_split.columns:
        print(f"  ⚠️  No 'dataset' column found in metadata")
        return pd.DataFrame()
    
    # Prepare feature arrays (if needed)
    X = None
    X_text = None
    X_vision = None
    if query_embeddings is not None and len(split_embedding_indices) > 0:
        X = query_embeddings[split_embedding_indices]
    elif text_embeddings is not None and len(split_embedding_indices) > 0:
        X_text = text_embeddings[split_embedding_indices]
    if query_embeddings is None and vision_embeddings is not None and len(split_embedding_indices) > 0:
        X_vision = vision_embeddings[split_embedding_indices]
    
    # Predict for the whole split
    try:
        if hasattr(router, 'predict'):
            # Routers with a predict() method
            if X is not None:
                predictions = router.predict(X=X, meta=meta_split)
            elif X_text is not None and X_vision is not None:
                predictions = router.predict(X_text=X_text, X_vision=X_vision, meta=meta_split)
            elif X_text is not None:
                predictions = router.predict(X_text=X_text, meta=meta_split)
            elif X_vision is not None:
                predictions = router.predict(X_vision=X_vision, meta=meta_split)
            else:
                predictions = router.predict(meta=meta_split)
        else:
            print(f"  ⚠️  Router does not have predict method")
            return pd.DataFrame()
    except Exception as e:
        print(f"  ❌ Prediction failed: {e}")
        return pd.DataFrame()
    
    # Ensure predictions is a numpy array
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    
    # Get all datasets
    datasets = meta_split['dataset'].unique()
    print(f"  ✓ Found {len(datasets)} datasets")
    
    results = []
    
    for dataset_name in sorted(datasets):
        # Sample indices for this dataset (relative to meta_split)
        dataset_mask = (meta_split['dataset'] == dataset_name).values
        dataset_indices = np.where(dataset_mask)[0]
        
        if len(dataset_indices) == 0:
            continue
        
        # Predictions/labels/costs for this dataset
        dataset_preds = predictions[dataset_indices]
        dataset_Y = Y_split[dataset_indices]
        dataset_C = C_split[dataset_indices]
        
        # Compute accuracy and cost
        correct = 0
        total_cost = 0.0
        
        for i, pred_idx in enumerate(dataset_preds):
            if 0 <= pred_idx < dataset_Y.shape[1]:
                # Check correctness
                if dataset_Y[i, pred_idx] == 1:
                    correct += 1
                # Accumulate cost
                total_cost += dataset_C[i, pred_idx]
        
        num_samples = len(dataset_indices)
        accuracy = correct / num_samples if num_samples > 0 else 0.0
        avg_cost = total_cost / num_samples if num_samples > 0 else 0.0
        
        # Compute rank_score
        rs = rank_score(accuracy, avg_cost, cmin, cmax, beta)
        
        results.append({
            'dataset': dataset_name,
            'accuracy': accuracy,
            'avg_cost': avg_cost,
            'rank_score': rs,
            'num_correct': correct,
            'num_samples': num_samples
        })
    
    results_df = pd.DataFrame(results).sort_values('rank_score', ascending=False)
    
    print(f"  ✓ Evaluated {len(results_df)} datasets")
    
    return results_df


def save_dataset_results(
    dataset_results_df: pd.DataFrame,
    output_dir,
    model_name: str,
    split: str = 'test',
    print_summary: bool = True
):
    """
    Save per-dataset evaluation results.
    
    Args:
        dataset_results_df: Per-dataset results DataFrame
        output_dir: Output directory (Path or str)
        model_name: Model name
        split: Split name ('train', 'dev', 'test')
        print_summary: Whether to print a summary
    
    Returns:
        Saved CSV file path
    """
    from pathlib import Path
    
    output_dir = Path(output_dir)
    
    if dataset_results_df.empty:
        print(f"  ⚠️  No dataset results to save")
        return None
    
    # Save CSV file
    csv_path = output_dir / f"{model_name}_{split}_by_dataset.csv"
    dataset_results_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"✓ Per-dataset results saved: {csv_path}")
    
    # Print summary
    if print_summary:
        print(f"\n  📈 Top 10 datasets by rank_score:")
        for i, row in dataset_results_df.head(10).iterrows():
            print(f"    {row['dataset']:<30} Acc: {row['accuracy']:.4f}, "
                  f"Cost: ${row['avg_cost']:.6f}, RS: {row['rank_score']:.4f}, "
                  f"Samples: {row['num_samples']}")
        
        if len(dataset_results_df) > 10:
            print(f"\n  📉 Bottom 10 datasets by rank_score:")
            for i, row in dataset_results_df.tail(10).iterrows():
                print(f"    {row['dataset']:<30} Acc: {row['accuracy']:.4f}, "
                      f"Cost: ${row['avg_cost']:.6f}, RS: {row['rank_score']:.4f}, "
                      f"Samples: {row['num_samples']}")
    
    return csv_path
