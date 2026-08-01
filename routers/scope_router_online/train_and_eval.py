#!/usr/bin/env python3
"""Train and evaluate the online SCOPE-Router variant."""

import argparse
from datetime import timedelta
import json
import os
import pickle
import random
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from routers.scope_router_online.router import OnlineScopeRouter
from routers.utils.benchmarks_data import attach_text_assets_from_benchmarks, load_splits_jsonl
from routers.utils.rank_score import get_cost_bounds_from_config, rank_score


def parse_float(value: str) -> float:
    if str(value).lower() == "inf":
        return float("inf")
    return float(value)


def format_float_for_name(value) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def setup_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = bool(args.ddp or args.multi_gpu or world_size > 1)
    if not use_ddp:
        return {
            "distributed": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "device": args.device,
        }

    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:
        raise ImportError("DDP training requires PyTorch") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("DDP mode requires CUDA GPUs")

    required_env = {"RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
    missing_env = sorted(required_env - set(os.environ))
    if missing_env:
        missing = ", ".join(missing_env)
        raise RuntimeError(f"DDP mode must be launched with torchrun; missing environment: {missing}")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
    return {
        "distributed": True,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": f"cuda:{local_rank}",
    }


def cleanup_distributed(enabled: bool):
    if not enabled:
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def load_raw_data_for_training(dataset_dir: Path, verbose: int = 1):
    if verbose > 0:
        print("📥 Loading raw multimodal data...")

    Y = np.load(dataset_dir / "data/matrices/Y.npz")["Y"]
    meta = pd.read_parquet(dataset_dir / "data/registry/meta.parquet")

    cost_npy = dataset_dir / "data/matrices/C.npy"
    C_full = np.load(cost_npy)
    sample_ids_file = dataset_dir / "data/matrices/sample_ids.pkl"
    if sample_ids_file.exists():
        with open(sample_ids_file, "rb") as f:
            c_sample_ids = pickle.load(f)
        c_id_to_idx = {sid: i for i, sid in enumerate(c_sample_ids)}
        C = np.zeros((len(meta), C_full.shape[1]), dtype=C_full.dtype)
        for i, sid in enumerate(meta["sample_id"].values):
            C[i] = C_full[c_id_to_idx[sid]] if sid in c_id_to_idx else C_full.mean(axis=0)
    else:
        C = C_full

    with open(dataset_dir / "data/registry/model_index.pkl", "rb") as f:
        models = pickle.load(f)

    meta = attach_text_assets_from_benchmarks(
        meta,
        dataset_dir / "BENCHMARKS",
        verbose=verbose,
        fallback_on_zero_match=False,
        allow_meta_text_fallback=False,
        allow_meta_image_fallback=False,
    )
    splits = load_splits_jsonl(dataset_dir, verbose=verbose)

    if verbose > 0:
        print(f"  Y: {Y.shape}, C: {C.shape}")
        print(f"  Models: {models}")
    return {"Y": Y, "C": C, "meta": meta, "models": models, "splits": splits}


def split_arrays(data, split_name: str):
    split_ids = set(data["splits"].get(split_name, []))
    mask = pd.Series(data["meta"]["sample_id"].values).isin(split_ids).values
    indices = np.where(mask)[0]
    return data["Y"][indices], data["C"][indices], data["meta"].iloc[indices].copy().reset_index(drop=True)


def evaluate_router(router, Y, C, meta, cmin=None, cmax=None, beta=0.1):
    preds = router.predict(meta=meta)
    correct = Y[np.arange(len(preds)), preds]
    costs = C[np.arange(len(preds)), preds]
    result = {
        "accuracy": float(correct.mean()),
        "avg_cost": float(costs.mean()),
        "num_correct": int(correct.sum()),
        "num_samples": int(len(correct)),
    }
    if cmin is not None and cmax is not None:
        result["rank_score"] = float(rank_score(result["accuracy"], result["avg_cost"], cmin, cmax, beta=beta))
    return result


def evaluate_by_dataset(router, Y, C, meta, cmin=None, cmax=None, beta=0.1):
    rows = []
    for dataset_name, group in meta.groupby("dataset", sort=True):
        idx = group.index.to_numpy()
        local_meta = group.copy()
        local_meta.index = range(len(local_meta))
        result = evaluate_router(router, Y[idx], C[idx], local_meta, cmin=cmin, cmax=cmax, beta=beta)
        result["dataset"] = dataset_name
        rows.append(result)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Train online SCOPE-Router")
    parser.add_argument("--dataset_dir", default=".", help="Dataset root directory")
    parser.add_argument("--profile_path", required=True, help="Calibration profile .npz")
    parser.add_argument("--output_dir", default="outputs/scope_router_online")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text_encoder", default="BAAI/bge-m3")
    parser.add_argument("--vision_encoder", default="facebook/dinov2-base")
    parser.add_argument("--fusion_method", default="normalize_concat",
                        choices=[
                            "concat",
                            "average",
                            "weighted_average",
                            "normalize_concat",
                            "concat_interaction",
                            "normalize_concat_interaction",
                            "only_text",
                            "only_image",
                        ])
    parser.add_argument("--text_weight", type=float, default=0.5)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--query_hidden_dim", type=int, default=128)
    parser.add_argument("--query_layers", type=int, default=2)
    parser.add_argument("--profile_hidden_dim", type=int, default=512)
    parser.add_argument("--profile_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--profile_learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Batch size for rank-0 evaluation. Defaults to 4x training batch size.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers per DDP process.")
    parser.add_argument("--pin_memory", action="store_true",
                        help="Use pinned host memory for CUDA DataLoaders.")
    parser.add_argument("--max_iter", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--loss_type", default="crm", choices=["softmax", "clip", "clip_relevance", "clip-relevance", "crm"])
    parser.add_argument("--crm_target", default="relevance", choices=["soft", "y", "relevance"])
    parser.add_argument("--train_lambda", type=str, default="10.0")
    parser.add_argument("--hard_labels", action="store_true", help="Use cheapest-correct hard labels instead of soft labels")
    parser.add_argument("--cost_scale", type=float, default=100.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--monitor_metric", default="rank_score", choices=["rank_score", "accuracy", "avg_cost"])
    parser.add_argument("--query_profile_update", default="static",
                        choices=["static", "epoch_refresh", "differentiable"],
                        help=(
                            "How to handle query-aware profile features while online encoders train. "
                            "static keeps the offline profile fixed; epoch_refresh recomputes calibration "
                            "query means after each epoch with stop-gradient; differentiable rebuilds them "
                            "inside each training batch so encoder gradients also flow through the profile."
                        ))
    parser.add_argument("--profile_refresh_batch_size", type=int, default=None,
                        help="Batch size for encoding calibration samples during query profile refresh.")
    parser.add_argument("--differentiable_calib_size", type=int, default=None,
                        help=(
                            "For --query_profile_update differentiable, randomly sample this many calibration "
                            "examples per training batch to build the differentiable query-aware profile. "
                            "Defaults to all calibration examples, which is usually too memory-heavy."
                        ))
    parser.add_argument("--device", default=None, help="Device, e.g. cuda or cpu. Defaults to auto.")
    parser.add_argument("--ddp", action="store_true",
                        help="Use torch DistributedDataParallel; also enabled automatically under torchrun")
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Deprecated alias for --ddp")
    parser.add_argument("--unsafe_select_on_test", action="store_true",
                        help="UNSAFE: use test split for best-state selection. Do not report resulting test as clean.")
    parser.add_argument("--skip_dataset_eval", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    dist_info = setup_distributed(args)
    is_main = dist_info["rank"] == 0

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = Path(args.profile_path)
    if not profile_path.is_absolute():
        profile_path = dataset_dir / profile_path
    if not profile_path.exists():
        raise FileNotFoundError(f"Missing profile file: {profile_path}")

    train_lambda = parse_float(args.train_lambda)
    selection_split = "test" if args.unsafe_select_on_test else "dev"
    use_soft_labels = not args.hard_labels

    if is_main:
        print("=" * 80)
        print("Online SCOPE-Router training")
        print("=" * 80)
        print(f"Profile: {profile_path}")
        print(f"Online encoders: {args.text_encoder} + {args.vision_encoder} ({args.fusion_method})")
        print(
            "Query path: train text/image encoders, then QueryMLP "
            f"(ed={args.embedding_dim}, qh={args.query_hidden_dim}, qlayers={args.query_layers})"
        )
        print(f"DDP: {dist_info['distributed']} world_size={dist_info['world_size']}")
        print(f"Loss: {args.loss_type}")
        if args.loss_type == "crm":
            print(f"CRM target: {args.crm_target}")
        print(f"Query-aware profile update: {args.query_profile_update}")
        if args.query_profile_update == "differentiable":
            print("WARNING: differentiable profile construction re-encodes calibration samples inside each training batch.")
            if args.differentiable_calib_size:
                print(f"Differentiable calibration subset per batch: {args.differentiable_calib_size}")
        print(f"Best-state selection split: {selection_split}")
        if args.unsafe_select_on_test:
            print("WARNING: using test for model selection. This is only for debugging/upper-bound probing.")

    data = load_raw_data_for_training(dataset_dir, verbose=1 if is_main else 0)
    Y_train, C_train, meta_train = split_arrays(data, "train")
    Y_sel, C_sel, meta_sel = split_arrays(data, selection_split)
    if is_main:
        print(f"\nTrain set: {len(meta_train)}")
        print(f"Selection set ({selection_split}): {len(meta_sel)}")

    cost_bounds_file = dataset_dir / "data/matrices/cost_bounds.json"
    if cost_bounds_file.exists():
        cmin, cmax = get_cost_bounds_from_config(str(cost_bounds_file))
    else:
        cmin = float(data["C"].min())
        cmax = float(data["C"].max())

    model_costs = C_train.mean(axis=0)
    model_mapping = {i: data["models"][i] for i in range(Y_train.shape[1])}

    router = OnlineScopeRouter(
        profile_path=str(profile_path),
        dataset_dir=str(dataset_dir),
        text_encoder=args.text_encoder,
        vision_encoder=args.vision_encoder,
        fusion_method=args.fusion_method,
        text_weight=args.text_weight,
        embedding_dim=args.embedding_dim,
        query_hidden_dim=args.query_hidden_dim,
        query_layers=args.query_layers,
        profile_hidden_dim=args.profile_hidden_dim,
        profile_layers=args.profile_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        profile_learning_rate=args.profile_learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        max_iter=args.max_iter,
        max_length=args.max_length,
        temperature=args.temperature,
        loss_type=args.loss_type,
        crm_target=args.crm_target,
        train_lambda=train_lambda,
        use_soft_labels=use_soft_labels,
        cost_scale=args.cost_scale,
        patience=args.patience,
        monitor_metric=args.monitor_metric,
        query_profile_update=args.query_profile_update,
        profile_refresh_batch_size=args.profile_refresh_batch_size,
        differentiable_calib_size=args.differentiable_calib_size,
        device=dist_info["device"],
        multi_gpu=False,
        distributed=dist_info["distributed"],
        local_rank=dist_info["local_rank"],
        rank=dist_info["rank"],
        world_size=dist_info["world_size"],
        random_state=args.seed,
        verbose=1 if is_main else 0,
    )

    if is_main:
        print("\nTraining...")
    router.fit(
        Y_train,
        C_train,
        meta_train,
        Y_dev=Y_sel,
        C_dev=C_sel,
        meta_dev=meta_sel,
        model_mapping=model_mapping,
        costs=model_costs,
        cmin=cmin,
        cmax=cmax,
        rank_score_beta=0.1,
        calibration_meta=data["meta"],
    )

    if dist_info["distributed"]:
        import torch.distributed as dist

        dist.barrier(device_ids=[dist_info["local_rank"]])
    if not is_main:
        cleanup_distributed(dist_info["distributed"])
        return None

    if dist_info["distributed"]:
        router.text_model = router._unwrap_model(router.text_model)
        router.vision_model = router._unwrap_model(router.vision_model)
        router.query_encoder = router._unwrap_model(router.query_encoder)
        router.profile_encoder = router._unwrap_model(router.profile_encoder)
        router.distributed = False

    label_tag = "hard" if args.hard_labels else f"softlambda{format_float_for_name(train_lambda)}"
    model_name = (
        f"scope_router_online_{Path(args.profile_path).stem}_{label_tag}"
        f"_loss{args.loss_type}_crmt{args.crm_target}"
        f"_ed{args.embedding_dim}_qh{args.query_hidden_dim}_ph{args.profile_hidden_dim}"
        f"_d{format_float_for_name(args.dropout)}"
        f"_wd{format_float_for_name(args.weight_decay)}_lr{format_float_for_name(args.learning_rate)}"
        f"_temp{format_float_for_name(args.temperature)}"
        f"_qprof{args.query_profile_update}"
    )
    if args.query_profile_update == "differentiable" and args.differentiable_calib_size:
        model_name += f"_dc{args.differentiable_calib_size}"
    if args.unsafe_select_on_test:
        model_name += "_unsafe_test_selected"
    model_path = output_dir / f"{model_name}.pkl"
    router.save(str(model_path))
    print(f"\nModel saved: {model_path}")

    print("\nEvaluating splits...")
    results = {}
    for split_name in ["train", "dev", "test"]:
        if split_name not in data["splits"]:
            continue
        Y_split, C_split, meta_split = split_arrays(data, split_name)
        if len(meta_split) == 0:
            continue
        result = evaluate_router(router, Y_split, C_split, meta_split, cmin=cmin, cmax=cmax, beta=0.1)
        results[split_name] = result
        print(f"  {split_name}: acc={result['accuracy']:.4f} cost=${result['avg_cost']:.6f} "
              f"rank_score={result.get('rank_score', -1):.4f}")

    dataset_results = []
    if not args.skip_dataset_eval and "test" in data["splits"]:
        Y_test, C_test, meta_test = split_arrays(data, "test")
        dataset_results_df = evaluate_by_dataset(router, Y_test, C_test, meta_test, cmin=cmin, cmax=cmax, beta=0.1)
        if not dataset_results_df.empty:
            dataset_results = dataset_results_df.to_dict("records")
            dataset_csv = output_dir / f"{model_name}_test_by_dataset.csv"
            dataset_results_df.to_csv(dataset_csv, index=False, float_format="%.6f")
            print(f"Per-dataset results saved: {dataset_csv}")

    report = {
        "model": model_name,
        "profile_path": str(profile_path),
        "selection_split": selection_split,
        "unsafe_select_on_test": bool(args.unsafe_select_on_test),
        "best_epoch": router.best_epoch,
        "best_selection_metrics": router.best_dev_metrics,
        "hyperparameters": vars(args),
        "results": results,
        "results_by_dataset": dataset_results,
    }
    report_path = output_dir / f"{model_name}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report saved: {report_path}")

    test_results = results.get("test", {})
    summary = {
        "router": model_name,
        "selection_split": selection_split,
        "unsafe_select_on_test": bool(args.unsafe_select_on_test),
        "accuracy": test_results.get("accuracy", 0.0),
        "avg_quality": test_results.get("accuracy", 0.0),
        "avg_cost": test_results.get("avg_cost", 0.0),
        "rank_score": test_results.get("rank_score", -1),
        "num_samples": test_results.get("num_samples", 0),
        "num_correct": test_results.get("num_correct", 0),
    }
    summary_path = output_dir / f"{model_name}_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path}")

    print("\nDone.")
    cleanup_distributed(dist_info["distributed"])
    return report


if __name__ == "__main__":
    main()
