#!/usr/bin/env python3
"""
RouterDC router training and evaluation script

Usage:
    python routers/routerdc/train_and_eval.py \
        --dataset_dir . \
        --batch_size 16 \
        --learning_rate 2e-5 \
        --max_epochs 5 \
        --use_soft_labels \
        --train_lambda 10.0 \
        --output_dir outputs/routerdc
"""

import argparse
import json
import os
from pathlib import Path
import sys
import random

# Avoid OpenBLAS oversubscribing high-core servers before NumPy/sklearn import.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("MKL_NUM_THREADS", "32")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "32")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from routers.routerdc.router import RouterDCRouter
from routers.utils.benchmarks_data import load_multimodal_data_for_end_to_end_router


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate RouterDC-LXMERT router")
    parser.add_argument("--dataset_dir", default=".", help="Dataset root directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    # RouterDC hyperparameters
    parser.add_argument("--K_plus", type=int, default=3, help="Number of positive samples")
    parser.add_argument("--K_minus", type=int, default=3, help="Number of negative samples")
    parser.add_argument("--lambda_sample_sample", type=float, default=1.0, 
                       help="Weight for sample-sample loss")
    parser.add_argument("--sample_sample_type", default="routerdc",
                       choices=["routerdc", "soft_target"],
                       help="Sample-sample loss: original RouterDC cluster loss or soft target similarity loss")
    parser.add_argument("--sample_sample_temperature", type=float, default=0.1,
                       help="Temperature for soft_target sample-sample loss")
    parser.add_argument("--num_clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("--H_out_group", type=int, default=3, help="Number of out-of-group samples")
    
    # LXMERT parameters
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    
    # Training parameters
    parser.add_argument("--max_epochs", type=int, default=5, help="Maximum training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1, 
                       help="Gradient accumulation steps")
    
    # Training monitoring
    parser.add_argument("--enable_monitoring", action="store_true", help="Enable training monitoring")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--monitor_metric", default="rank_score", 
                       choices=['rank_score', 'accuracy', 'avg_cost'], help="Monitoring metric")
    
    # Soft-label parameters
    parser.add_argument("--use_soft_labels", action="store_true", help="Train with soft labels")
    parser.add_argument("--train_lambda", type=str, default="0.0", 
                       help="Lambda for soft-label training (number or 'inf')")
    parser.add_argument("--cost_scale", type=float, default=100.0,
                       help="Scale applied to cost-aware soft/relevance targets")
    parser.add_argument("--loss_type", default="routerdc",
                       choices=["routerdc", "siglip_relevance"],
                       help="Sample-LLM loss: original RouterDC contrastive loss or dense relevance BCE")
    
    # Runtime parameters
    parser.add_argument("--device", default="cuda", choices=['cuda', 'cpu'], 
                       help="Compute device")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity")
    parser.add_argument("--output_dir", default="outputs/routerdc", 
                       help="Output directory")
    
    # Optional: limit number of training samples (for quick tests)
    parser.add_argument("--max_train_samples", type=int, default=None,
                       help="Limit number of training samples (for quick testing)")
    
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
    print("🚀 RouterDC-LXMERT router training and evaluation")
    print("="*80)
    if is_hard_label:
        print("Hard-label training: λ=∞ (hard label to cheapest correct model)")
    elif use_soft_labels:
        print(f"Soft-label training: λ={train_lambda}")
    print(f"Loss type: {args.loss_type}, cost_scale={args.cost_scale:g}")
    print(f"RouterDC hyperparameters: K+={args.K_plus}, K-={args.K_minus}, "
          f"λ_ss={args.lambda_sample_sample}, ss_type={args.sample_sample_type}, "
          f"ss_temp={args.sample_sample_temperature:g}, clusters={args.num_clusters}")
    print(f"Training: epochs={args.max_epochs}, lr={args.learning_rate}, "
          f"batch_size={args.batch_size}")
    
    # 1. Load data
    # NOTE: keep behavior consistent with historical per-router loader
    print("📥 Loading RouterDC-LXMERT training data...")
    data = load_multimodal_data_for_end_to_end_router(dataset_dir, mode="simple", verbose=1)
    
    # 2. Prepare training data
    print("\n🔧 Preparing training data...")
    train_ids = set(data['splits'].get('train', []))
    
    meta_sample_ids = data['meta']['sample_id'].values
    train_mask = pd.Series(meta_sample_ids).isin(train_ids)
    train_indices = np.where(train_mask.values)[0]
    
    # Limit training samples (if specified)
    if args.max_train_samples is not None and len(train_indices) > args.max_train_samples:
        print(f"  ⚠️  Limiting training samples: {len(train_indices)} -> {args.max_train_samples}")
        train_indices = np.random.choice(train_indices, args.max_train_samples, replace=False)
    
    Y_train = data['Y'][train_indices]
    C_train = data['C'][train_indices]
    meta_train = data['meta'].iloc[train_indices].copy()
    
    print(f"  Train set: {len(train_indices)} samples")
    
    # Prepare dev data (if monitoring enabled)
    Y_dev, C_dev, meta_dev = None, None, None
    if args.enable_monitoring:
        print("\n🔧 Preparing dev data (for training monitoring)...")
        dev_ids = set(data['splits'].get('dev', []))
        dev_mask = pd.Series(meta_sample_ids).isin(dev_ids)
        dev_indices = np.where(dev_mask.values)[0]
        
        Y_dev = data['Y'][dev_indices]
        C_dev = data['C'][dev_indices]
        meta_dev = data['meta'].iloc[dev_indices].copy()
        
        print(f"  Dev set: {len(dev_indices)} samples")
    
    # Compute average model costs
    model_costs = C_train.mean(axis=0)

    # Rank Score cost bounds (for dev monitoring consistency)
    cost_bounds_file_for_monitor = dataset_dir / "data/matrices/cost_bounds.json"
    if cost_bounds_file_for_monitor.exists():
        try:
            from routers.utils.rank_score import get_cost_bounds_from_config
            cmin, cmax = get_cost_bounds_from_config(str(cost_bounds_file_for_monitor))
        except Exception:
            cmin = float(C_train.min())
            cmax = float(C_train.max())
    else:
        cmin = float(C_train.min())
        cmax = float(C_train.max())
    
    # 3. Train RouterDC-LXMERT router
    print("\n🎓 Training RouterDC-LXMERT router...")
    if args.enable_monitoring:
        print(f"  Training monitoring: enabled (patience={args.patience})")
    
    router = RouterDCRouter(
        K_plus=args.K_plus,
        K_minus=args.K_minus,
        lambda_sample_sample=args.lambda_sample_sample,
        sample_sample_type=args.sample_sample_type,
        sample_sample_temperature=args.sample_sample_temperature,
        num_clusters=args.num_clusters,
        H_out_group=args.H_out_group,
        max_length=args.max_length,
        use_soft_labels=use_soft_labels,
        train_lambda=0.0 if is_hard_label else train_lambda,
        cost_scale=args.cost_scale,
        loss_type=args.loss_type,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        grad_accumulation_steps=args.grad_accumulation_steps,
        enable_monitoring=args.enable_monitoring,
        patience=args.patience,
        monitor_metric=args.monitor_metric,
        device=args.device,
        verbose=args.verbose
    )
    
    # Build model mapping
    K = Y_train.shape[1]
    model_mapping = {i: data['models'][i] for i in range(K)}
    
    # Pass dev data (if monitoring enabled)
    monitor_output_dir = output_dir / "training_monitor" if args.enable_monitoring else None
    
    router.fit(Y_train, C_train, meta_train,
               Y_dev=Y_dev, C_dev=C_dev, meta_dev=meta_dev,
               model_mapping=model_mapping,
               costs=model_costs,
               monitor_output_dir=monitor_output_dir,
               cmin=cmin, cmax=cmax, rank_score_beta=0.1)

    # Ensure evaluation uses the exact "best_model.pkl" selected during monitoring
    if args.enable_monitoring and monitor_output_dir is not None:
        best_model_file = monitor_output_dir / "best_model.pkl"
        summary_file = monitor_output_dir / "training_summary.json"
        if summary_file.exists():
            try:
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
                print("\n📌 Best monitored model info (from training_summary.json):")
                print(f"  best_epoch: {summary.get('best_epoch')}")
                print(f"  best_metric: {summary.get('best_metric')}={summary.get('best_metric_value')}")
                print(f"  best_model_path: {summary.get('best_model_path')}")
            except Exception as e:
                print(f"\n⚠️  Failed to read training_summary.json: {e}")

        if best_model_file.exists():
            try:
                router = RouterDCRouter.load(str(best_model_file), device=args.device)
                router.verbose = args.verbose
                print(f"\n✓ Loaded best monitored model explicitly for evaluation: {best_model_file}")
            except Exception as e:
                print(f"\n⚠️  Failed to load best_model.pkl explicitly; using current router: {e}")
        else:
            print(f"\n⚠️  Best monitored model file not found: {best_model_file}. Evaluation may not use the best model.")
    
    # 4. Save model
    model_name = f"routerdc_lr{args.learning_rate:.0e}_bs{args.batch_size}"
    if args.loss_type != "routerdc":
        model_name += f"_{args.loss_type}_cs{args.cost_scale:g}"
    if args.sample_sample_type != "routerdc":
        model_name += f"_ss{args.sample_sample_type}_t{args.sample_sample_temperature:g}"
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
    
    # Load cost bounds for Arena Score
    cost_bounds_file = dataset_dir / "data/matrices/cost_bounds.json"
    
    for split_name in ['train', 'dev', 'test']:
        if split_name not in data['splits']:
            continue
        
        split_ids = set(data['splits'][split_name])
        
        # Find indices of split samples in meta
        split_mask = pd.Series(meta_sample_ids).isin(split_ids)
        split_indices = np.where(split_mask.values)[0]
        
        if len(split_indices) == 0:
            continue
        
        Y_split = data['Y'][split_indices]
        C_split = data['C'][split_indices]
        meta_split = data['meta'].iloc[split_indices].copy()
        
        # Predict
        predictions = router.predict(meta=meta_split)
        
        # Compute metrics
        correct = Y_split[np.arange(len(predictions)), predictions]
        accuracy = correct.mean()
        
        costs = C_split[np.arange(len(predictions)), predictions]
        avg_cost = costs.mean()
        
        result = {
            'accuracy': float(accuracy),
            'avg_cost': float(avg_cost),
            'num_correct': int(correct.sum()),
            'num_samples': len(correct)
        }
        
        # Compute Rank Score
        if cost_bounds_file.exists():
            try:
                from routers.utils.rank_score import rank_score, get_cost_bounds_from_config
                from routers.utils.dataset_evaluation import evaluate_by_dataset, save_dataset_results
                cmin, cmax = get_cost_bounds_from_config(cost_bounds_file)
                score = rank_score(accuracy, avg_cost, cmin, cmax, beta=0.1)
                result['rank_score'] = float(score)
                result['rank_score_beta'] = 0.1
            except Exception as e:
                print(f"  Warning: failed to compute Rank Score: {e}")
        
        results[split_name] = result
        
        print(f"  {split_name}:")
        print(f"    Accuracy: {result['accuracy']:.4f} ({result['num_correct']}/{result['num_samples']})")
        print(f"    Avg cost: ${result['avg_cost']:.6f}")
        if 'rank_score' in result:
            print(f"    Rank Score: {result['rank_score']:.4f}")
    
    # 6. Evaluate test set by dataset
    dataset_results = {}
    dataset_results_df = pd.DataFrame()
    if 'test' in data['splits']:
        # Load cost bounds (for rank_score)
        try:
            from routers.utils.rank_score import get_cost_bounds_from_config
            cmin, cmax = get_cost_bounds_from_config(str(cost_bounds_file))
            print(f"\n✓ Using cost bounds: cmin=${cmin:.6f}, cmax=${cmax:.6f}")
        except Exception as e:
            # Compute from data
            cmin = data['C'].min()
            cmax = data['C'].max()
            print(f"\n✓ Calculated cost bounds from data: cmin=${cmin:.6f}, cmax=${cmax:.6f}")
        
        # Prepare test set data
        test_ids = set(data['splits']['test'])
        test_meta_mask = data['meta']['sample_id'].isin(test_ids)
        test_meta_indices = np.where(test_meta_mask.values)[0]
        
        Y_test = data['Y'][test_meta_indices]
        C_test = data['C'][test_meta_indices]
        meta_test = data['meta'].iloc[test_meta_indices].copy()
        
        # Evaluate
        from routers.utils.dataset_evaluation import evaluate_by_dataset, save_dataset_results
        dataset_results_df = evaluate_by_dataset(
            router=router,
            Y_split=Y_test,
            C_split=C_test,
            meta_split=meta_test,
            models=data['models'],
        text_embeddings=None,  # Router-DC-LXMERT does not use precomputed embeddings
            vision_embeddings=None,
            split_embedding_indices=np.array([]),
            cmin=cmin,
            cmax=cmax,
            beta=0.1
        )
        
        if not dataset_results_df.empty:
            dataset_results = dataset_results_df.to_dict('records')
            save_dataset_results(dataset_results_df, output_dir, model_name, split='test', print_summary=True)
    
    # 7. Save outputs
    report = {
        'model': model_name,
        'hyperparameters': {
            'K_plus': args.K_plus,
            'K_minus': args.K_minus,
            'lambda_sample_sample': args.lambda_sample_sample,
            'sample_sample_type': args.sample_sample_type,
            'sample_sample_temperature': args.sample_sample_temperature,
            'num_clusters': args.num_clusters,
            'H_out_group': args.H_out_group,
            'max_length': args.max_length,
            'loss_type': args.loss_type,
            'cost_scale': args.cost_scale,
            'use_soft_labels': args.use_soft_labels,
            'train_lambda': args.train_lambda,
            'max_epochs': args.max_epochs,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'warmup_ratio': args.warmup_ratio,
            'weight_decay': args.weight_decay
        },
        'results': results,
        'results_by_dataset': dataset_results
    }
    
    report_path = output_dir / f"{model_name}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Report saved: {report_path}")
    
    # 7. Save CSV summary (for visualization with baselines)
    test_results = results.get('test', {})
    summary_row = {
        'router': model_name,
        'accuracy': test_results.get('accuracy', 0),
        'avg_quality': test_results.get('accuracy', 0),
        'avg_cost': test_results.get('avg_cost', 0),
        'num_samples': test_results.get('num_samples', 0),
        'num_correct': test_results.get('num_correct', 0),
        'rank_score': test_results.get('rank_score', -1)
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
