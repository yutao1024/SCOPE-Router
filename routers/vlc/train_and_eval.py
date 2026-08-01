#!/usr/bin/env python3
"""
VLC router training and evaluation script

Supports 4 vision-language model variants:
- VisualBERT: single-stream architecture
- LXMERT: dual-stream architecture
- UNITER: single-stream architecture (with modal type embeddings)
- ViLBERT: dual-stream architecture (with co-attention)

Usage:
    python routers/vlc/train_and_eval.py \
        --model_type visualbert \
        --dataset_dir . \
        --use_soft_labels \
        --train_lambda 0.0 \
        --max_epochs 5 \
        --batch_size 16
"""

import argparse
import json
import shutil
import time
import random
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from routers.vlc.router import VLCRouter
from routers.utils.benchmarks_data import load_multimodal_data_for_end_to_end_router


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate VLC Router')
    
    # Model selection
    parser.add_argument('--model_type', type=str, default='visualbert',
                       choices=['visualbert', 'lxmert', 'uniter', 'vilbert'],
                       help='VL model type (visualbert/lxmert/uniter/vilbert)')
    
    # Dataset
    parser.add_argument('--dataset_dir', type=str, default='.',
                       help='Dataset root directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--image_base_dir', type=str, default='data',
                       help='Base directory for images')
    
    # Training
    parser.add_argument('--use_soft_labels', action='store_true',
                       help='Train with soft labels')
    parser.add_argument('--train_lambda', type=str, default='0.0',
                       help='Lambda for soft labels (number or inf)')
    parser.add_argument('--loss_type', type=str, default='softmax',
                       choices=['softmax', 'siglip_relevance'],
                       help='Training loss. siglip_relevance uses dense cost-aware relevance targets.')
    parser.add_argument('--sample_sample_weight', type=float, default=0.0,
                       help='Weight for SCOPE-Router-style soft-target sample-sample loss')
    parser.add_argument('--sample_sample_temperature', type=float, default=0.1,
                       help='Temperature for sample-sample contrastive loss')
    parser.add_argument('--cost_scale', type=float, default=100.0,
                       help='Scale applied when constructing cost-aware targets')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=5,
                       help='Maximum training epochs')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Warmup ratio')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze VL model backbone')
    
    # Monitoring
    parser.add_argument('--enable_monitoring', action='store_true', default=True,
                       help='Enable training monitoring')
    parser.add_argument('--patience', type=int, default=2,
                       help='Early stopping patience')
    parser.add_argument('--monitor_metric', type=str, default='rank_score',
                       choices=['rank_score', 'accuracy', 'avg_cost'],
                       help='Monitoring metric')
    
    # Training constraints
    parser.add_argument('--max_train_samples', type=int, default=None,
                       help='Max training samples (for quick tests; default: unlimited)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/vlc',
                       help='Output directory')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                       help='Compute device (cuda/cpu)')
    
    # Verbosity
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity (0=quiet, 1=normal, 2=debug)')
    
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
    
    # Parse lambda
    if args.train_lambda.lower() == 'inf':
        train_lambda = float('inf')
        is_hard_label = True
        use_soft_labels = False
    else:
        train_lambda = float(args.train_lambda)
        is_hard_label = (train_lambda == float('inf'))
        use_soft_labels = args.use_soft_labels and not is_hard_label
    
    # Setup paths
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    
    # Create output subdirectory based on configuration
    label_type = f"soft\u03bb{train_lambda}" if use_soft_labels else f"hard\u03bbinf"
    output_subdir = output_dir / args.model_type / f"{args.model_type}_{label_type}"
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print(f"VLC router training - {args.model_type.upper()}")
    print("=" * 80)
    print(f"  Model type: {args.model_type}")
    print(f"  Label type: {'soft labels (λ=' + str(train_lambda) + ')' if use_soft_labels else 'hard labels (λ=∞)'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Output dir: {output_subdir}")
    print()
    
    # Load data (shared multimodal loader)
    data = load_multimodal_data_for_end_to_end_router(dataset_dir, mode="vlc", verbose=1)
    
    Y = data['Y']
    C = data['C']
    meta = data['meta']
    splits = data['splits']
    model_mapping = data['model_mapping']
    
    # Split data
    train_ids = set(splits['train'])
    dev_ids = set(splits['dev'])
    test_ids = set(splits['test'])
    
    train_mask = meta['sample_id'].isin(train_ids)
    dev_mask = meta['sample_id'].isin(dev_ids)
    test_mask = meta['sample_id'].isin(test_ids)
    
    train_indices = np.where(train_mask.values)[0]
    dev_indices = np.where(dev_mask.values)[0]
    test_indices = np.where(test_mask.values)[0]
    
    # Limit training samples (if specified)
    if args.max_train_samples is not None and len(train_indices) > args.max_train_samples:
        print(f"  ⚠️  Limiting training samples: {len(train_indices)} -> {args.max_train_samples}")
        train_indices = np.random.choice(train_indices, args.max_train_samples, replace=False)
    
    Y_train, C_train = Y[train_indices], C[train_indices]
    Y_dev, C_dev = Y[dev_indices], C[dev_indices]
    Y_test, C_test = Y[test_indices], C[test_indices]
    
    meta_train = meta.iloc[train_indices].copy().reset_index(drop=True)
    meta_dev = meta.iloc[dev_indices].copy().reset_index(drop=True)
    meta_test = meta.iloc[test_indices].copy().reset_index(drop=True)
    
    print("📊 Data splits:")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Dev: {len(dev_indices)} samples")
    print(f"  Test: {len(test_indices)} samples")
    print()
    
    # Initialize router
    print("🔧 Initializing VLC router...")
    router = VLCRouter(
        model_type=args.model_type,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        device=args.device,
        use_soft_labels=use_soft_labels,
        train_lambda=train_lambda,
        loss_type=args.loss_type,
        sample_sample_weight=args.sample_sample_weight,
        sample_sample_temperature=args.sample_sample_temperature,
        cost_scale=args.cost_scale,
        enable_monitoring=args.enable_monitoring,
        patience=args.patience,
        monitor_metric=args.monitor_metric,
        verbose=args.verbose,
        image_base_dir=args.image_base_dir
    )
    
    # Train
    print("🎓 Starting training...")
    train_start = time.time()
    
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

    router.fit(
        Y=Y_train,
        C=C_train,
        meta=meta_train,
        model_mapping=model_mapping,
        Y_dev=Y_dev,
        C_dev=C_dev,
        meta_dev=meta_dev,
        monitor_output_dir=output_subdir / "training",
        cmin=cmin, cmax=cmax, rank_score_beta=0.1
    )
    
    train_time = time.time() - train_start
    print(f"\n⏱  Training time: {train_time / 60:.2f} minutes")
    
    # Ensure evaluation uses the exact "best_model.pkl" selected during monitoring
    monitor_output_dir = output_subdir / "training"
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
            router = VLCRouter.load(str(best_model_file))
            router.verbose = args.verbose
            print(f"\n✓ Loaded best monitored model explicitly for evaluation: {best_model_file}")
        except Exception as e:
            print(f"\n⚠️  Failed to load best_model.pkl explicitly; using current router: {e}")
    else:
        print(f"\n⚠️  Best monitored model file not found: {best_model_file}. Evaluation may not use the best model.")

    # Save model snapshot used for evaluation (easy to find at output_subdir root)
    model_file = output_subdir / "best_model.pkl"
    router.save(str(model_file))
    print(f"💾 Model saved: {model_file}")
    
    # Load cost bounds for Rank Score calculation
    cost_bounds_file = dataset_dir / "data/matrices/cost_bounds.json"
    
    # Evaluate on dev set (use FULL dev set without filtering)
    print("\n" + "=" * 80)
    print("📊 Dev set evaluation")
    print("=" * 80)
    
    dev_predictions = router.predict(meta=meta_dev)
    dev_acc = np.mean(Y_dev[range(len(dev_predictions)), dev_predictions])
    dev_cost = np.mean(C_dev[range(len(dev_predictions)), dev_predictions])
    
    # Calculate Rank Score using standard formula
    dev_rank_score = -1.0
    if cost_bounds_file.exists():
        try:
            from routers.utils.rank_score import rank_score, get_cost_bounds_from_config
            cmin, cmax = get_cost_bounds_from_config(cost_bounds_file)
            dev_rank_score = rank_score(dev_acc, dev_cost, cmin, cmax, beta=0.1)
        except Exception as e:
            print(f"  Warning: failed to compute Rank Score: {e}")
    
    print(f"  Accuracy: {dev_acc:.4f}")
    print(f"  Avg cost: {dev_cost:.6f}")
    if dev_rank_score >= 0:
        print(f"  Rank Score: {dev_rank_score:.4f}")
    
    # Evaluate on test set (use FULL test set without filtering)
    print("\n" + "=" * 80)
    print("📊 Test set evaluation")
    print("=" * 80)
    
    test_predictions = router.predict(meta=meta_test)
    test_acc = np.mean(Y_test[range(len(test_predictions)), test_predictions])
    test_cost = np.mean(C_test[range(len(test_predictions)), test_predictions])
    
    # Calculate Rank Score using standard formula
    test_rank_score = -1.0
    if cost_bounds_file.exists():
        try:
            from routers.utils.rank_score import rank_score, get_cost_bounds_from_config
            cmin, cmax = get_cost_bounds_from_config(cost_bounds_file)
            test_rank_score = rank_score(test_acc, test_cost, cmin, cmax, beta=0.1)
        except Exception as e:
            print(f"  Warning: failed to compute Rank Score: {e}")
    
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Avg cost: {test_cost:.6f}")
    if test_rank_score >= 0:
        print(f"  Rank Score: {test_rank_score:.4f}")
    
    # Evaluate test set by dataset
    print("\n" + "=" * 80)
    print("📊 Per-dataset evaluation")
    print("=" * 80)
    
    dataset_results = {}
    dataset_results_df = pd.DataFrame()
    
    # Load cost bounds (for rank_score)
    try:
        from routers.utils.rank_score import get_cost_bounds_from_config
        cmin, cmax = get_cost_bounds_from_config(str(cost_bounds_file))
        print(f"✓ Using cost bounds: cmin=${cmin:.6f}, cmax=${cmax:.6f}")
    except Exception as e:
        # Compute from data
        cmin = C.min()
        cmax = C.max()
        print(f"✓ Calculated cost bounds from data: cmin=${cmin:.6f}, cmax=${cmax:.6f}")
    
    # Evaluate
    # Convert model_mapping (dict) to models list (for evaluate_by_dataset compatibility)
    models_list = [model_mapping[i] for i in sorted(model_mapping.keys())]
    
    from routers.utils.dataset_evaluation import evaluate_by_dataset, save_dataset_results
    dataset_results_df = evaluate_by_dataset(
        router=router,
        Y_split=Y_test,
        C_split=C_test,
        meta_split=meta_test,
        models=models_list,
        text_embeddings=None,  # VLC does not use precomputed embeddings
        vision_embeddings=None,
        split_embedding_indices=np.array([]),
        cmin=cmin,
        cmax=cmax,
        beta=0.1
    )
    
    if not dataset_results_df.empty:
        dataset_results = dataset_results_df.to_dict('records')
        # Output file name is derived from configuration
        model_name = f"{args.model_type}_soft" if use_soft_labels else f"{args.model_type}_hard"
        model_name += f"λ{train_lambda if train_lambda != float('inf') else 'inf'}"
        save_dataset_results(dataset_results_df, output_dir, model_name, split='test', print_summary=True)
    
    # Latency profiling
    print("\n" + "=" * 80)
    print("⚡ Latency profiling")
    print("=" * 80)
    
    # Sample subset for profiling (use full test set)
    num_profile_samples = min(100, len(meta_test))
    profile_indices = np.random.choice(len(meta_test), num_profile_samples, replace=False)
    meta_profile = meta_test.iloc[profile_indices].reset_index(drop=True)
    
    # Warmup
    print("  Warmup runs...")
    for _ in range(3):
        _ = router.predict(meta=meta_profile)
    
    # Profile
    print(f"  Benchmark (n={num_profile_samples}, runs=5)...")
    latencies = []
    for _ in range(5):
        start = time.time()
        _ = router.predict(meta=meta_profile)
        latencies.append(time.time() - start)
    
    avg_latency_ms = np.mean(latencies) * 1000 / num_profile_samples
    std_latency_ms = np.std(latencies) * 1000 / num_profile_samples
    throughput = num_profile_samples / np.mean(latencies)
    
    print(f"  Per-sample latency: {avg_latency_ms:.2f} ± {std_latency_ms:.2f} ms")
    print(f"  Throughput: {throughput:.1f} samples/sec")
    
    # Calculate token-based metrics
    try:
        from routers.utils.token_stats import TokenCounter, compute_prompt_tokens
        counter = TokenCounter()
        total_tokens = sum(compute_prompt_tokens(row.to_dict(), counter) for _, row in meta_profile.iterrows())
        avg_tokens_per_sample = total_tokens / len(meta_profile)
        avg_latency_ms_per_token = avg_latency_ms / avg_tokens_per_sample
        throughput_tokens = throughput * avg_tokens_per_sample
        token_counting_method = 'tiktoken (exact)'
    except Exception as e:
        # Fallback to simple estimation
        avg_tokens_per_sample = 300  # Rough estimate
        avg_latency_ms_per_token = avg_latency_ms / avg_tokens_per_sample
        throughput_tokens = throughput * avg_tokens_per_sample
        token_counting_method = 'simple estimation'
    
    print(f"  Per-token latency: {avg_latency_ms_per_token:.4f} ms/token")
    print(f"  Token throughput: {throughput_tokens:.1f} tokens/sec")
    print(f"  Token counting: {token_counting_method}")
    
    # Save results
    print("\n" + "=" * 80)
    print("💾 Saving outputs")
    print("=" * 80)
    
    results = {
        'model_type': args.model_type,
        'config': {
            'use_soft_labels': use_soft_labels,
            'train_lambda': train_lambda if train_lambda != float('inf') else 'inf',
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'max_epochs': args.max_epochs,
            'dropout': args.dropout,
            'weight_decay': args.weight_decay,
            'freeze_backbone': args.freeze_backbone
        },
        'training': {
            'time_minutes': train_time / 60
        },
        'dev': {
            'accuracy': float(dev_acc),
            'avg_cost': float(dev_cost),
            'rank_score': float(dev_rank_score)
        },
        'test': {
            'accuracy': float(test_acc),
            'avg_cost': float(test_cost),
            'rank_score': float(test_rank_score)
        },
        'results_by_dataset': dataset_results,
        'latency': {
            'avg_ms_per_sample': float(avg_latency_ms),
            'std_ms_per_sample': float(std_latency_ms),
            'throughput_samples_per_sec': float(throughput),
            'avg_ms_per_token': float(avg_latency_ms_per_token),
            'throughput_tokens_per_sec': float(throughput_tokens),
            'token_counting_method': token_counting_method
        }
    }
    
    # Save JSON report
    report_file = output_subdir / f"{args.model_type}_{label_type}_report.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Report: {report_file}")

    # Also place a copy directly under --output_dir (for scripts that expect top-level outputs)
    public_report_file = output_dir / f"{args.model_type}_{label_type}_report.json"
    if public_report_file.resolve() != report_file.resolve():
        shutil.copyfile(report_file, public_report_file)
        print(f"  Report (top-level): {public_report_file}")
    
    # Save CSV summary
    summary_file = output_subdir / f"{args.model_type}_{label_type}_summary.csv"
    summary_row = {
        'model_type': args.model_type,
        'use_soft_labels': use_soft_labels,
        'train_lambda': train_lambda if train_lambda != float('inf') else 'inf',
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_epochs': args.max_epochs,
        'dropout': args.dropout,
        'freeze_backbone': args.freeze_backbone,
        'train_time_min': train_time / 60,
        'dev_accuracy': dev_acc,
        'dev_cost': dev_cost,
        'dev_rank_score': dev_rank_score,
        'test_accuracy': test_acc,
        'test_cost': test_cost,
        'test_rank_score': test_rank_score,
        'latency_ms_per_sample': avg_latency_ms,
        'throughput_samples_per_sec': throughput,
        'latency_ms_per_token': avg_latency_ms_per_token,
        'throughput_tokens_per_sec': throughput_tokens,
        'token_counting_method': token_counting_method
    }
    
    summary_df = pd.DataFrame([summary_row])
    summary_df.to_csv(summary_file, index=False)
    print(f"  Summary: {summary_file}")

    # Also place a copy directly under --output_dir (for scripts that expect top-level outputs)
    public_summary_file = output_dir / f"{args.model_type}_{label_type}_summary.csv"
    if public_summary_file.resolve() != summary_file.resolve():
        shutil.copyfile(summary_file, public_summary_file)
        print(f"  Summary (top-level): {public_summary_file}")
    
    print("\n" + "=" * 80)
    print("✅ Training and evaluation complete!")
    print("=" * 80)
    print("\n📊 Final results:")
    print(f"  Dev Rank Score:  {dev_rank_score:.4f}")
    print(f"  Test Rank Score: {test_rank_score:.4f}")
    print(f"  Latency: {avg_latency_ms:.2f} ms/sample ({avg_latency_ms_per_token:.4f} ms/token)")
    print()


if __name__ == "__main__":
    main()
