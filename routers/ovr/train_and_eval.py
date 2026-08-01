#!/usr/bin/env python3
"""
OVR router training and evaluation script

Structure aligned with `routers/linear/train_and_eval.py` and `routers/mlp/train_and_eval.py`.

Usage:
    python routers/ovr/train_and_eval.py \
        --dataset_dir . \
        --fusion_method normalize_concat \
        --output_dir outputs/ovr
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from routers.ovr.router import OVRRouter
from routers.utils.train_utils import (
    load_data_for_training,
    align_train_data,
    evaluate_router_with_indices,
    profile_router_latency,
)
from routers.utils.rank_score import get_cost_bounds_from_config
from routers.utils.dataset_evaluation import evaluate_by_dataset


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate OVR router")
    parser.add_argument("--dataset_dir", default=".", help="Dataset root directory")
    parser.add_argument(
        "--fusion_method",
        default="normalize_concat",
        choices=["concat", "average", "weighted_average", "normalize_concat", "only_text", "only_image"],
        help="Feature fusion method",
    )
    parser.add_argument("--text_weight", type=float, default=0.5, help="Text weight (for weighted_average)")

    parser.add_argument("--text_encoder", default="BAAI/bge-m3", help="Text encoder model name")
    parser.add_argument("--vision_encoder", default="facebook/dinov2-base", help="Vision encoder model name")
    parser.add_argument("--output_dir", default="outputs/ovr", help="Output directory")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity (0/1)")

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🚀 OVR router training and evaluation")
    print("=" * 80)
    print(f"Fusion method: {args.fusion_method}")
    if args.fusion_method == "weighted_average":
        print(f"Text weight: {args.text_weight}")
    print("Label mode: hard label (use raw Y matrix as multi-label targets)")

    # 1) Load data
    data = load_data_for_training(dataset_dir, text_encoder=args.text_encoder, vision_encoder=args.vision_encoder)

    # 2) Prepare training data (alignment)
    print("\n🔧 Preparing training data...")
    train_ids = set(data["splits"].get("train", []))
    (X_text_train, X_vision_train, Y_train, C_train, meta_train, _, _) = align_train_data(data, train_ids)
    print(f"  Train set: {len(meta_train)} samples (aligned)")

    # 3) Train OVR Router
    print("\n🎓 Training OVR router...")
    router = OVRRouter(
        fusion_method=args.fusion_method,
        text_weight=args.text_weight,
        text_encoder=args.text_encoder,
        vision_encoder=args.vision_encoder,
        verbose=args.verbose,
    )

    K = Y_train.shape[1]
    model_mapping = {i: data["models"][i] for i in range(K)}

    router.fit(
        Y_train,
        C_train,
        meta_train,
        X_text=X_text_train,
        X_vision=X_vision_train,
        model_mapping=model_mapping,
    )

    # 4) Save model
    model_name = f"ovr_{args.fusion_method}"
    if args.fusion_method == "weighted_average":
        model_name += f"_w{args.text_weight:.2f}"

    model_path = output_dir / f"{model_name}.pkl"
    router.save(str(model_path))
    print(f"\n✓ Model saved: {model_path}")

    # 5) Evaluate (train/dev/test)
    print("\n📊 Evaluating performance...")
    results = {}
    cost_bounds_file = dataset_dir / "data/matrices/cost_bounds.json"

    for split_name in ["train", "dev", "test"]:
        if split_name not in data["splits"]:
            continue

        split_ids = set(data["splits"][split_name])
        meta_sample_ids = data["meta"]["sample_id"].values
        split_mask = pd.Series(meta_sample_ids).isin(split_ids)
        split_indices = np.where(split_mask.values)[0]

        embeddings_sample_ids = data["sample_ids"]
        split_embedding_mask = pd.Series(embeddings_sample_ids).isin(split_ids)
        split_embedding_indices = np.where(split_embedding_mask.values)[0]

        if len(split_indices) == 0:
            print(f"  {split_name}: 0 samples, skipping")
            continue

        Y_split = data["Y"][split_indices]
        C_split = data["C"][split_indices]
        meta_split = data["meta"].iloc[split_indices].copy()

        result = evaluate_router_with_indices(
            router,
            Y_split,
            C_split,
            meta_split,
            data["models"],
            data["text_embeddings"],
            data["vision_embeddings"],
            split_embedding_indices,
            cost_bounds_file=cost_bounds_file if cost_bounds_file.exists() else None,
        )
        results[split_name] = result

        print(f"  {split_name}:")
        print(f"    Accuracy: {result['accuracy']:.4f} ({result['num_correct']}/{result['num_samples']})")
        print(f"    Avg cost: ${result['avg_cost']:.6f}")
        if "rank_score" in result:
            print(f"    Rank Score: {result['rank_score']:.4f}")

    # 6) Evaluate test set by dataset (aligned with linear/mlp outputs)
    dataset_results = {}
    dataset_results_df = pd.DataFrame()  # Initialize empty DataFrame to ensure scope
    if "test" in data["splits"]:
        try:
            cmin, cmax = get_cost_bounds_from_config(str(cost_bounds_file))
            print(f"\n✓ Using cost bounds: cmin=${cmin:.6f}, cmax=${cmax:.6f}")
        except Exception:
            cmin = data["C"].min()
            cmax = data["C"].max()
            print(f"\n✓ Calculated cost bounds from data: cmin=${cmin:.6f}, cmax=${cmax:.6f}")

        test_ids = set(data["splits"]["test"])
        test_meta_mask = data["meta"]["sample_id"].isin(test_ids)
        test_meta_indices = np.where(test_meta_mask.values)[0]
        test_embedding_mask = pd.Series(data["sample_ids"]).isin(test_ids)
        test_embedding_indices = np.where(test_embedding_mask)[0]

        if len(test_meta_indices) > 0 and len(test_embedding_indices) > 0:
            Y_test = data["Y"][test_meta_indices]
            C_test = data["C"][test_meta_indices]
            meta_test = data["meta"].iloc[test_meta_indices].copy()

            dataset_results_df = evaluate_by_dataset(
                router=router,
                Y_split=Y_test,
                C_split=C_test,
                meta_split=meta_test,
                models=data["models"],
                text_embeddings=data["text_embeddings"],
                vision_embeddings=data["vision_embeddings"],
                split_embedding_indices=test_embedding_indices,
                cmin=cmin,
                cmax=cmax,
                beta=0.1,
            )
            if not dataset_results_df.empty:
                dataset_results = dataset_results_df.to_dict("records")
                
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

    # 7) Latency profiling
    print("\n⏱️  Profiling latency...")
    latency_metrics = {"batch_size": 16, "ms_per_sample": -1, "throughput": -1}
    if "test" in data["splits"]:
        test_ids = set(data["splits"].get("test", []))
        test_embedding_mask = pd.Series(data["sample_ids"]).isin(test_ids)
        test_embedding_indices = np.where(test_embedding_mask.values)[0]
        if len(test_embedding_indices) > 0:
            X_text_test = data["text_embeddings"][test_embedding_indices]
            X_vision_test = data["vision_embeddings"][test_embedding_indices]
            latency_metrics = profile_router_latency(
                router=router,
                X_text=X_text_test,
                X_vision=X_vision_test,
                batch_size=16,
                warmup_runs=5,
                test_runs=50,
            )
            print(f"  Batch size: {latency_metrics['batch_size']}")
            print(f"  Per-sample latency: {latency_metrics['ms_per_sample']:.3f} ms/sample")
            print(f"  Sample throughput: {latency_metrics['throughput']:.1f} samples/sec")
        else:
            print("  ⚠️  Test split is empty; skipping latency analysis")

    # 8) Save report
    report = {
        "model": model_name,
        "hyperparameters": {
            "fusion_method": args.fusion_method,
            "text_weight": args.text_weight if args.fusion_method == "weighted_average" else None,
        },
        "results": results,
        "results_by_dataset": dataset_results,
        "latency": latency_metrics,
    }
    report_path = output_dir / f"{model_name}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Report saved: {report_path}")
    
    # Save per-dataset results to a separate CSV
    if not dataset_results_df.empty:
        dataset_csv_path = output_dir / f"{model_name}_test_by_dataset.csv"
        dataset_results_df.to_csv(dataset_csv_path, index=False, float_format='%.6f')
        print(f"✓ Per-dataset results saved: {dataset_csv_path}")

    # summary.csv (field set aligned with linear/mlp)
    test_results = results.get("test", {})
    summary_row = {
        "router": model_name,
        "accuracy": test_results.get("accuracy", 0),
        "avg_quality": test_results.get("accuracy", 0),
        "avg_cost": test_results.get("avg_cost", 0),
        "num_samples": test_results.get("num_samples", 0),
        "num_correct": test_results.get("num_correct", 0),
        "rank_score": test_results.get("rank_score", -1),
        "latency_ms_per_sample": latency_metrics.get("ms_per_sample", -1),
        "throughput_samples_per_sec": latency_metrics.get("throughput", -1),
    }
    summary_path = output_dir / f"{model_name}_summary.csv"
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
    print(f"✓ Summary saved: {summary_path}")

    print("\n" + "=" * 80)
    print("✅ Training and evaluation complete!")
    print("=" * 80)

    return report


if __name__ == "__main__":
    main()


