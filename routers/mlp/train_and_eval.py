#!/usr/bin/env python3
"""
MLPR router training and evaluation script

Usage:
    python routers/mlp/train_and_eval.py \
        --dataset_dir . \
        --fusion_method normalize_concat \
        --use_soft_labels \
        --train_lambda 10.0 \
        --output_dir outputs/mlp_router
"""

import argparse
import json
from pathlib import Path
import sys
import random
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from routers.mlp.router import MLPRouter
from routers.utils.train_utils import (
    load_data_for_training,
    align_train_data,
    evaluate_router_with_indices,
    profile_router_latency
)
from routers.utils.rank_score import get_cost_bounds_from_config
from routers.utils.dataset_evaluation import evaluate_by_dataset, save_dataset_results


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate MLPR router")
    parser.add_argument("--dataset_dir", default=".", help="Dataset root directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--fusion_method", default="normalize_concat",
                       choices=['concat', 'average', 'weighted_average', 'normalize_concat', 'only_text', 'only_image'],
                       help="Feature fusion method")
    parser.add_argument("--text_weight", type=float, default=0.5,
                       help="Text weight (for weighted_average)")
    parser.add_argument("--hidden_sizes", type=int, nargs='+', default=[2048, 1024],
                       help="Hidden layer sizes, e.g. --hidden_sizes 2048 1024")
    parser.add_argument("--activation", default="relu", choices=['relu', 'tanh', 'logistic'],
                       help="Activation function")
    parser.add_argument("--alpha", type=float, default=0.0001,
                       help="L2 regularization coefficient")
    parser.add_argument("--use_soft_labels", action="store_true",
                       help="Train with soft labels")
    parser.add_argument("--train_lambda", type=str, default="0.0",
                       help="Lambda for soft-label training (number or 'inf')")
    parser.add_argument("--text_encoder", default="BAAI/bge-m3",
                       help="Text encoder model name (default: BAAI/bge-m3)")
    parser.add_argument("--vision_encoder", default="facebook/dinov2-base",
                       help="Vision encoder model name (default: facebook/dinov2-base)")
    parser.add_argument("--enable_monitoring", action="store_true",
                       help="Enable training monitoring (early stopping, dev eval, etc.)")
    parser.add_argument("--patience", type=int, default=100,
                       help="Early stopping patience (default: 100)")
    parser.add_argument("--monitor_metric", default="rank_score",
                       choices=["rank_score", "accuracy", "avg_cost"],
                       help="Metric for selecting the best model (default: rank_score, based on dev set)")
    parser.add_argument("--output_dir", default="outputs/mlp_router", help="Output directory")
    
    args = parser.parse_args()

    # Reproducibility (best-effort)
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        pass
    
    # Parse lambda (handle \"inf\")
    if args.train_lambda.lower() == 'inf':
        train_lambda = float('inf')
        is_hard_label = True
        use_soft_labels = False
    else:
        train_lambda = float(args.train_lambda)
        is_hard_label = (train_lambda == float('inf'))
        use_soft_labels = args.use_soft_labels and not is_hard_label
    
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("🚀 MLPR router training and evaluation")
    print("="*80)
    print(f"Fusion method: {args.fusion_method}")
    print(f"Hidden layers: {tuple(args.hidden_sizes)}, activation: {args.activation}")
    if is_hard_label:
        print("Hard-label training: λ=∞ (hard label to cheapest correct model)")
    elif use_soft_labels:
        print(f"Soft-label training: λ={train_lambda}")
    
    # 1. Load data
    data = load_data_for_training(dataset_dir,
                                  text_encoder=args.text_encoder,
                                  vision_encoder=args.vision_encoder)
    
    # 2. Prepare training data (alignment)
    print("\n🔧 Preparing training data...")
    train_ids = set(data['splits'].get('train', []))
    
    (X_text_train, X_vision_train, Y_train, C_train, meta_train,
     aligned_meta_indices, aligned_embedding_indices) = align_train_data(data, train_ids)
    
    print(f"  Train set: {len(aligned_meta_indices)} samples (aligned)")
    
    # Prepare dev data (if monitoring enabled)
    X_text_dev, X_vision_dev, Y_dev, C_dev, meta_dev = None, None, None, None, None
    if args.enable_monitoring:
        print("\n🔧 Preparing dev data (for training monitoring)...")
        dev_ids = set(data['splits'].get('dev', []))
        (X_text_dev, X_vision_dev, Y_dev, C_dev, meta_dev,
         _, _) = align_train_data(data, dev_ids)
        print(f"  Dev set: {len(meta_dev)} samples (aligned)")
    
    # Compute average model costs (for cost-aware training/inference)
    model_costs = C_train.mean(axis=0)

    # Rank Score cost bounds (for dev monitoring + reporting consistency)
    cost_bounds_file = dataset_dir / "data/matrices/cost_bounds.json"
    if cost_bounds_file.exists():
        try:
            cmin, cmax = get_cost_bounds_from_config(str(cost_bounds_file))
        except Exception:
            cmin = float(data['C'].min())
            cmax = float(data['C'].max())
    else:
        cmin = float(data['C'].min())
        cmax = float(data['C'].max())
    
    # 3. Train MLPRouter
    print("\n🎓 Training MLPR router...")
    if args.enable_monitoring:
        print(f"  Training monitoring: enabled (patience={args.patience})")
    
    router = MLPRouter(
        hidden_sizes=tuple(args.hidden_sizes),
        activation=args.activation,
        alpha=args.alpha,
        fusion_method=args.fusion_method,
        text_weight=args.text_weight,
        text_encoder=args.text_encoder,
        vision_encoder=args.vision_encoder,
        use_soft_labels=use_soft_labels,
        train_lambda=0.0 if is_hard_label else train_lambda,
        enable_monitoring=args.enable_monitoring,
        patience=args.patience,
        monitor_metric=args.monitor_metric,
        random_state=args.seed
    )
    
    # Build model mapping
    K = Y_train.shape[1]
    model_mapping = {i: data['models'][i] for i in range(K)}
    
    # Pass dev data (if monitoring enabled)
    monitor_output_dir = output_dir / "training_monitor" if args.enable_monitoring else None
    
    router.fit(Y_train, C_train, meta_train,
               X_text=X_text_train, X_vision=X_vision_train,
               Y_dev=Y_dev, C_dev=C_dev, meta_dev=meta_dev,
               X_text_dev=X_text_dev, X_vision_dev=X_vision_dev,
               model_mapping=model_mapping,
               costs=model_costs,
               monitor_output_dir=monitor_output_dir,
               cmin=cmin, cmax=cmax, rank_score_beta=0.1)
    
    # Confirm best model is loaded (if monitoring was used)
    if args.enable_monitoring and monitor_output_dir:
        best_model_file = monitor_output_dir / "best_model.pkl"
        if best_model_file.exists():
            print("\n✓ Training monitoring: loaded best dev-set model automatically")
            # Router already loaded the best model in fit(); this is just a sanity check
        else:
            print("\n⚠️  Training monitoring did not save a best model; using last-epoch model")
    
    # 4. Save model (named by hyperparameters)
    model_name = f"mlp_{args.fusion_method}"
    if args.hidden_sizes != [256]:
        model_name += f"_h{'_'.join(map(str, args.hidden_sizes))}"
    if args.activation != 'relu':
        model_name += f"_{args.activation}"
    if is_hard_label:
        model_name += "_hardλinf"
    elif use_soft_labels:
        model_name += f"_softλ{train_lambda:.1f}"
    
    model_path = output_dir / f"{model_name}.pkl"
    router.save(str(model_path))
    print(f"\n✓ Model saved: {model_path}")
    
    # 5. Evaluate (train/dev/test)
    print("\n📊 Evaluating performance...")
    results = {}
    
    # Load cost bounds for Rank Score
    cost_bounds_file = dataset_dir / "data/matrices/cost_bounds.json"
    
    for split_name in ['train', 'dev', 'test']:
        if split_name not in data['splits']:
            continue
        
        split_ids = set(data['splits'][split_name])
        
        # Find indices of split samples in meta
        meta_sample_ids = data['meta']['sample_id'].values
        split_mask = pd.Series(meta_sample_ids).isin(split_ids)
        split_indices = np.where(split_mask.values)[0]
        
        # Find indices of split samples in embeddings
        embeddings_sample_ids = data['sample_ids']
        split_embedding_mask = pd.Series(embeddings_sample_ids).isin(split_ids)
        split_embedding_indices = np.where(split_embedding_mask.values)[0]
        
        if len(split_indices) == 0:
            continue
        
        Y_split = data['Y'][split_indices]
        C_split = data['C'][split_indices]
        meta_split = data['meta'].iloc[split_indices].copy()
        
        result = evaluate_router_with_indices(
            router, Y_split, C_split, meta_split, data['models'],
            data['text_embeddings'], data['vision_embeddings'], split_embedding_indices,
            cost_bounds_file=cost_bounds_file if cost_bounds_file.exists() else None
        )
        results[split_name] = result
        
        print(f"  {split_name}:")
        print(f"    Accuracy: {result['accuracy']:.4f} ({result['num_correct']}/{result['num_samples']})")
        print(f"    Avg cost: ${result['avg_cost']:.6f}")
        if 'rank_score' in result:
            print(f"    Rank Score: {result['rank_score']:.4f}")
    
    # 6. Evaluate test set by dataset
    dataset_results = {}
    if 'test' in data['splits']:
        # Load cost bounds (for rank_score)
        try:
            cmin, cmax = get_cost_bounds_from_config(str(cost_bounds_file))
            print(f"\n✓ Using cost bounds: cmin=${cmin:.6f}, cmax=${cmax:.6f}")
        except Exception as e:
            # Compute from data
            cmin = data['C'].min()
            cmax = data['C'].max()
            print(f"\n✓ Calculated cost bounds from data: cmin=${cmin:.6f}, cmax=${cmax:.6f}")
        
        # Per-dataset evaluation on the test set
        test_ids = set(data['splits']['test'])
        test_meta_mask = data['meta']['sample_id'].isin(test_ids)
        test_meta_indices = np.where(test_meta_mask.values)[0]
        test_embedding_mask = pd.Series(data['sample_ids']).isin(test_ids)
        test_embedding_indices = np.where(test_embedding_mask)[0]
        
        Y_test = data['Y'][test_meta_indices]
        C_test = data['C'][test_meta_indices]
        meta_test = data['meta'].iloc[test_meta_indices].copy()
        
        dataset_results_df = evaluate_by_dataset(
            router=router,
            Y_split=Y_test,
            C_split=C_test,
            meta_split=meta_test,
            models=data['models'],
            text_embeddings=data['text_embeddings'],
            vision_embeddings=data['vision_embeddings'],
            split_embedding_indices=test_embedding_indices,
            cmin=cmin,
            cmax=cmax,
            beta=0.1
        )
        
        if not dataset_results_df.empty:
            # Save per-dataset results
            dataset_results = dataset_results_df.to_dict('records')
            
            # Print top-10 and bottom-10 datasets
            print(f"\n  📈 Top 10 datasets by rank_score:")
            for i, row in dataset_results_df.head(10).iterrows():
                print(f"    {row['dataset']:<30} Acc: {row['accuracy']:.4f}, "
                      f"Cost: ${row['avg_cost']:.6f}, RS: {row['rank_score']:.4f}, "
                      f"Samples: {row['num_samples']}")
            
            print(f"\n  📉 Bottom 10 datasets by rank_score:")
            for i, row in dataset_results_df.tail(10).iterrows():
                print(f"    {row['dataset']:<30} Acc: {row['accuracy']:.4f}, "
                      f"Cost: ${row['avg_cost']:.6f}, RS: {row['rank_score']:.4f}, "
                      f"Samples: {row['num_samples']}")
    
    # 7. Profile latency on test set
    print("\n⏱️  Profiling latency...")
    # Reuse test_embedding_indices and meta_test computed above
    X_text_test = data['text_embeddings'][test_embedding_indices]
    X_vision_test = data['vision_embeddings'][test_embedding_indices]
    
    latency_metrics = profile_router_latency(
        router=router,
        X_text=X_text_test,
        X_vision=X_vision_test,
        meta=meta_test,
        batch_size=16,
        warmup_runs=5,
        test_runs=50
    )
    
    print(f"  Batch size: {latency_metrics['batch_size']}")
    if 'ms_per_token' in latency_metrics:
        print(f"  Per-token latency: {latency_metrics['ms_per_token']:.4f} ms/token")
        print(f"  Token throughput: {latency_metrics['throughput_tokens']:.1f} tokens/sec")
        print(f"  Token counting: {latency_metrics.get('token_counting_method', 'unknown')}")
    print(f"  Per-sample latency: {latency_metrics['ms_per_sample']:.3f} ms/sample")
    print(f"  Sample throughput: {latency_metrics['throughput']:.1f} samples/sec")
    
    # 8. Save outputs
    report = {
        'model': model_name,
        'hyperparameters': {
            'hidden_sizes': args.hidden_sizes,
            'activation': args.activation,
            'alpha': args.alpha,
            'seed': args.seed,
            'fusion_method': args.fusion_method,
            'text_weight': args.text_weight if args.fusion_method == 'weighted_average' else None,
            'use_soft_labels': args.use_soft_labels,
            'train_lambda': args.train_lambda
        },
        'results': results,
        'results_by_dataset': dataset_results,
        'latency': latency_metrics
    }
    
    report_path = output_dir / f"{model_name}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Report saved: {report_path}")
    
    # Save per-dataset results to a separate CSV file
    if dataset_results:
        dataset_csv_path = output_dir / f"{model_name}_test_by_dataset.csv"
        dataset_results_df.to_csv(dataset_csv_path, index=False, float_format='%.6f')
        print(f"✓ Per-dataset results saved: {dataset_csv_path}")
    
    # 9. Save CSV summary (for visualization with baselines)
    test_results = results.get('test', {})
    summary_row = {
        'router': model_name,
        'accuracy': test_results.get('accuracy', 0),
        'avg_quality': test_results.get('accuracy', 0),
        'avg_cost': test_results.get('avg_cost', 0),
        'num_samples': test_results.get('num_samples', 0),
        'num_correct': test_results.get('num_correct', 0),
        'rank_score': test_results.get('rank_score', -1),
        'latency_ms_per_sample': latency_metrics.get('ms_per_sample', -1),
        'throughput_samples_per_sec': latency_metrics.get('throughput', -1),
        'latency_ms_per_token': latency_metrics.get('ms_per_token', -1),
        'throughput_tokens_per_sec': latency_metrics.get('throughput_tokens', -1),
        'token_counting_method': latency_metrics.get('token_counting_method', 'N/A')
    }
    
    summary_path = output_dir / f"{model_name}_summary.csv"
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
    print(f"✓ Summary saved: {summary_path}")
    
    print("\n" + "="*80)
    print("✅ Training and evaluation complete!")
    print("="*80)
    
    return report


if __name__ == "__main__":
    main()
