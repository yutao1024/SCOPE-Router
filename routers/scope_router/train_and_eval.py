#!/usr/bin/env python3
"""
Train and evaluate SCOPE-Router.

This is the frozen-query-embedding variant: query embeddings are loaded from
precomputed text/vision parquet files, then only query/profile projection heads are
trained.
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from routers.scope_router.router import ScopeRouter
from routers.utils.dataset_evaluation import evaluate_by_dataset
from routers.utils.rank_score import get_cost_bounds_from_config
from routers.utils.train_utils import (
    align_train_data,
    evaluate_router_with_indices,
    load_data_for_training,
    profile_router_latency,
)


def parse_float(value: str) -> float:
    if str(value).lower() == "inf":
        return float("inf")
    return float(value)


def parse_csv_list(value: str):
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def aligned_split(data, split_name: str):
    split_ids = set(data["splits"].get(split_name, []))
    return align_train_data(data, split_ids)


def split_arrays_for_eval(data, split_name: str):
    split_ids = set(data["splits"].get(split_name, []))
    meta_sample_ids = data["meta"]["sample_id"].values
    embedding_sample_ids = data["sample_ids"]

    meta_mask = pd.Series(meta_sample_ids).isin(split_ids)
    embedding_mask = pd.Series(embedding_sample_ids).isin(split_ids)
    meta_indices = np.where(meta_mask.values)[0]
    embedding_indices = np.where(embedding_mask.values)[0]

    return (
        data["Y"][meta_indices],
        data["C"][meta_indices],
        data["meta"].iloc[meta_indices].copy(),
        embedding_indices,
    )


def format_float_for_name(value) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def short_tag(text: str, length: int = 8) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]


def compact_profile_tag(profile_path: Path) -> str:
    stem = profile_path.stem
    parts = []
    for size in ["128", "256", "512", "1024", "2048", "4096"]:
        if f"_{size}_" in f"_{stem}_":
            parts.append(size)
            break
    for strategy in ["diagnostic", "random", "kmeans", "hybrid"]:
        if strategy in stem:
            parts.append(strategy)
            break
    if "seen_only" in stem:
        parts.append("seenonly")
    split_match = re.search(r"split[A-Za-z0-9]+", stem)
    if split_match:
        parts.append(split_match.group(0))
    else:
        for split_tag in ["splitA", "splitB8", "splitB", "splitC"]:
            if split_tag in stem:
                parts.append(split_tag)
                break
    parts.append(short_tag(stem, 6))
    return "_".join(parts)


def compact_output_stem(stem: str, max_bytes: int = 180) -> str:
    """Keep artifact filenames below common filesystem limits while preserving uniqueness."""
    if len(stem.encode("utf-8")) <= max_bytes:
        return stem

    suffix = "_" + short_tag(stem, 10)
    budget = max_bytes - len(suffix.encode("utf-8"))
    kept = []
    used = 0
    for char in stem:
        char_bytes = len(char.encode("utf-8"))
        if used + char_bytes > budget:
            break
        kept.append(char)
        used += char_bytes
    return "".join(kept).rstrip("_-") + suffix


def write_subset_profile(full_profile_path: Path, output_dir: Path, seen_indices, tag: str) -> Path:
    profile_data = np.load(full_profile_path, allow_pickle=False)
    if "model_profile" not in profile_data:
        raise ValueError(f"{full_profile_path} must contain model_profile")
    full_profile = profile_data["model_profile"].astype(np.float32)
    if "model_names" in profile_data:
        full_names = profile_data["model_names"].astype(str)
    else:
        full_names = np.array([f"model_{i}" for i in range(full_profile.shape[0])], dtype=str)
    subset_path = output_dir / f"{full_profile_path.stem}_{tag}_seen_only.npz"
    arrays = {
        "model_profile": full_profile[seen_indices].astype(np.float32),
        "model_names": full_names[seen_indices].astype(str),
    }
    np.savez_compressed(subset_path, **arrays)
    return subset_path


def attach_full_candidate_profiles(router, full_profile_path: Path, expected_models):
    profile_data = np.load(full_profile_path, allow_pickle=False)
    if "model_profile" not in profile_data:
        raise ValueError(f"{full_profile_path} must contain model_profile")
    full_profile = profile_data["model_profile"].astype(np.float32)
    if "model_names" in profile_data:
        full_names = profile_data["model_names"].astype(str).tolist()
    else:
        full_names = [f"model_{i}" for i in range(full_profile.shape[0])]
    if list(full_names) != list(expected_models):
        raise ValueError(
            "Profile model_names do not match dataset model order; cannot attach full open-set candidates"
        )
    router.model_profile = router.profile_scaler.transform(full_profile)
    router.model_names = list(full_names)
    router.model_mapping = {i: name for i, name in enumerate(full_names)}
    router.reverse_mapping = {name: i for i, name in enumerate(full_names)}
    if hasattr(router, "_invalidate_profile_cache"):
        router._invalidate_profile_cache()


def attach_candidate_profile_subset(router, full_profile_path: Path, expected_models, candidate_indices):
    profile_data = np.load(full_profile_path, allow_pickle=False)
    if "model_profile" not in profile_data:
        raise ValueError(f"{full_profile_path} must contain model_profile")
    full_profile = profile_data["model_profile"].astype(np.float32)
    if "model_names" in profile_data:
        full_names = profile_data["model_names"].astype(str).tolist()
    else:
        full_names = [f"model_{i}" for i in range(full_profile.shape[0])]
    if list(full_names) != list(expected_models):
        raise ValueError(
            "Profile model_names do not match dataset model order; cannot attach candidate subset"
        )

    candidate_names = [full_names[i] for i in candidate_indices]
    router.model_profile = router.profile_scaler.transform(full_profile[candidate_indices])
    router.model_names = list(candidate_names)
    router.model_mapping = {i: name for i, name in enumerate(candidate_names)}
    router.reverse_mapping = {name: i for i, name in enumerate(full_names)}
    if hasattr(router, "_invalidate_profile_cache"):
        router._invalidate_profile_cache()


def main():
    parser = argparse.ArgumentParser(description="Train frozen-query SCOPE-Router")
    parser.add_argument("--dataset_dir", default=".", help="Dataset root directory")
    parser.add_argument("--profile_path", required=True, help="Calibration profile .npz")
    parser.add_argument("--output_dir", default="outputs/scope_router")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query_embedding_file", default=None,
                        help="Optional parquet with sample_id and embedding columns. "
                             "When provided, SCOPE-Router uses this joint query embedding "
                             "instead of text+vision encoder parquet files.")
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
                            "learnable_concat",
                            "learnable_gated_sum",
                            "only_text",
                            "only_image",
                        ])
    parser.add_argument("--text_weight", type=float, default=0.5)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--query_hidden_dim", type=int, default=512)
    parser.add_argument("--profile_hidden_dim", type=int, default=512)
    parser.add_argument("--query_layers", type=int, default=2)
    parser.add_argument("--profile_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer_type", default="adamw", choices=["adamw", "adam", "sgd"],
                        help="Optimizer for trainable projection/scorer parameters")
    parser.add_argument("--lr_scheduler", default="none", choices=["none", "cosine", "plateau", "step"],
                        help="Epoch-level learning-rate scheduler")
    parser.add_argument("--min_lr_factor", type=float, default=0.05,
                        help="Minimum LR as a factor of --learning_rate for cosine/plateau schedulers")
    parser.add_argument("--lr_step_size", type=int, default=20,
                        help="Epoch interval for --lr_scheduler step")
    parser.add_argument("--lr_gamma", type=float, default=0.5,
                        help="Decay factor for step/plateau schedulers")
    parser.add_argument("--lr_plateau_patience", type=int, default=5,
                        help="Bad epochs before reducing LR for --lr_scheduler plateau")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--score_type", default="dot", choices=["dot", "lowrank_bilinear", "set_aware", "cross_attention"],
                        help="Compatibility scorer after query/profile projection")
    parser.add_argument("--bilinear_rank", type=int, default=16,
                        help="Rank for --score_type lowrank_bilinear")
    parser.add_argument("--bilinear_residual_weight", type=float, default=1.0,
                        help="Fixed residual weight for the low-rank bilinear scorer")
    parser.add_argument("--set_encoder_layers", type=int, default=1,
                        help="Transformer layers for --score_type set_aware")
    parser.add_argument("--set_encoder_heads", type=int, default=4,
                        help="Attention heads for --score_type set_aware")
    parser.add_argument("--set_encoder_ff_dim", type=int, default=128,
                        help="Feed-forward hidden size for --score_type set_aware")
    parser.add_argument("--set_encoder_residual_weight", type=float, default=0.25,
                        help="Residual weight for contextualized profile features in --score_type set_aware")
    parser.add_argument("--set_encoder_normalize_input", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to L2-normalize profile-projection outputs before set-aware attention")
    parser.add_argument("--set_encoder_context_only", action="store_true",
                        help="Use only contextualized profile features instead of a residual update")
    parser.add_argument("--cross_attention_heads", type=int, default=4,
                        help="Attention heads for --score_type cross_attention")
    parser.add_argument("--cross_attention_residual_weight", type=float, default=0.25,
                        help="Residual weight for query update in --score_type cross_attention")
    parser.add_argument("--loss_type", default="softmax",
                        choices=["softmax", "clip", "clip_relevance", "clip-relevance", "crm", "blip"],
                        help="Training loss. Paper CRM uses --loss_type crm --crm_target relevance.")
    parser.add_argument("--crm_target", default="soft", choices=["soft", "y", "relevance"],
                        help="Target for --loss_type crm. Paper CRM uses relevance: unnormalized cost-aware relevance targets.")
    parser.add_argument("--crm_bias", default="none", choices=["none", "global", "profile"],
                        help="Bias for --loss_type crm: none, one global scalar, or a profile-conditioned scalar per model")
    parser.add_argument("--blip_alpha", type=float, default=0.4,
                        help="Momentum-teacher target mixing weight for --loss_type blip")
    parser.add_argument("--learn_blip_alpha", action="store_true",
                        help="Learn BLIP alpha as a sigmoid-constrained scalar initialized from --blip_alpha")
    parser.add_argument("--blip_momentum", type=float, default=0.995,
                        help="EMA momentum for BLIP-style momentum encoders")
    parser.add_argument("--blip_match_weight", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--rccr_weight", type=float, default=0.0,
                        help="Weight for RCCR: Routing-Consistency Contrastive Regularization")
    parser.add_argument("--rccr_temperature", type=float, default=0.1,
                        help="Temperature for RCCR")
    parser.add_argument("--learn_rccr_temperature", action="store_true",
                        help="Learn the RCCR temperature as a positive scalar initialized from --rccr_temperature")
    parser.add_argument("--train_lambda", type=str, default="10.0")
    parser.add_argument("--hard_labels", action="store_true", help="Use cheapest-correct hard labels instead of soft labels")
    parser.add_argument("--cost_scale", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--monitor_metric", default="rank_score", choices=["rank_score", "accuracy", "avg_cost"])
    parser.add_argument("--unsafe_select_on_test", action="store_true",
                        help="UNSAFE: use test split for best-state selection. Do not report resulting test as clean.")
    parser.add_argument("--holdout_models", default=None,
                        help="Comma-separated model names held out from training but restored for open-set evaluation")
    parser.add_argument("--eval_candidate_scope", default="all", choices=["seen", "holdout", "all"],
                        help="Candidate model pool for final evaluation when --holdout_models is set")
    parser.add_argument("--incremental_holdout_eval", action="store_true",
                        help="For --holdout_models, evaluate test metrics while adding held-out models back one by one")
    parser.add_argument("--incremental_holdout_order", default=None,
                        help="Optional comma-separated held-out model order for --incremental_holdout_eval")
    parser.add_argument("--skip_latency", action="store_true")
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

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = Path(args.profile_path)
    if not profile_path.is_absolute():
        profile_path = dataset_dir / profile_path
    if not profile_path.exists():
        calibration_guess = dataset_dir / "CALIBRATION" / "calib_1024_diagnostic_seed42.jsonl"
        raise FileNotFoundError(
            f"Missing profile file: {profile_path}\n"
            "Build it first, for example:\n"
            f"  PYTHONNOUSERSITE=1 python tools/build_calibration_profile.py "
            f"--dataset-dir {dataset_dir} "
            f"--calibration-file {calibration_guess if calibration_guess.exists() else 'CALIBRATION/calib_1024_diagnostic_seed42.jsonl'}"
        )

    train_lambda = parse_float(args.train_lambda)
    use_soft_labels = not args.hard_labels
    selection_split = "test" if args.unsafe_select_on_test else "dev"
    holdout_models = parse_csv_list(args.holdout_models)
    incremental_holdout_order = parse_csv_list(args.incremental_holdout_order) or list(holdout_models)
    if args.eval_candidate_scope == "holdout" and not holdout_models:
        raise ValueError("--eval_candidate_scope holdout requires --holdout_models")
    if args.incremental_holdout_eval and not holdout_models:
        raise ValueError("--incremental_holdout_eval requires --holdout_models")
    if args.incremental_holdout_eval:
        missing_from_order = [name for name in holdout_models if name not in set(incremental_holdout_order)]
        extra_in_order = [name for name in incremental_holdout_order if name not in set(holdout_models)]
        duplicate_in_order = sorted({name for name in incremental_holdout_order if incremental_holdout_order.count(name) > 1})
        if missing_from_order or extra_in_order or duplicate_in_order:
            raise ValueError(
                "--incremental_holdout_order must contain exactly the --holdout_models set; "
                f"missing={missing_from_order}, extra={extra_in_order}, duplicate={duplicate_in_order}"
            )

    print("=" * 80)
    print("SCOPE-Router training")
    print("=" * 80)
    print(f"Profile: {profile_path}")
    if args.query_embedding_file:
        print(f"Frozen query embedding: {args.query_embedding_file} (joint)")
    else:
        print(f"Frozen query embedding: {args.text_encoder} + {args.vision_encoder} ({args.fusion_method})")
    print(f"Loss: {args.loss_type}")
    print(
        f"Optimizer: {args.optimizer_type}, lr={args.learning_rate:g}, wd={args.weight_decay:g}, "
        f"scheduler={args.lr_scheduler}"
    )
    if args.loss_type in {"clip_relevance", "clip-relevance"}:
        print("CLIP target: relevance")
    if args.loss_type == "crm":
        print(f"CRM target: {args.crm_target}")
        print(f"CRM bias: {args.crm_bias}")
        if args.crm_target == "relevance":
            print("Paper objective: Cost-aware Relevance Matching (CRM)")
    if args.loss_type == "blip":
        print(f"BLIP alpha: {args.blip_alpha:g}")
        print(f"Learn BLIP alpha: {args.learn_blip_alpha}")
        print(f"BLIP momentum: {args.blip_momentum:g}")
    if args.rccr_weight > 0:
        print(f"RCCR weight: {args.rccr_weight:g}")
        print(f"RCCR temperature: {args.rccr_temperature:g}")
        print(f"Learn RCCR temperature: {args.learn_rccr_temperature}")
    if args.incremental_holdout_eval:
        print(f"Incremental holdout eval order: {incremental_holdout_order}")
    if holdout_models:
        print(f"Eval candidate scope: {args.eval_candidate_scope}")
    if args.score_type == "lowrank_bilinear":
        print(
            f"Score type: {args.score_type} "
            f"(rank={args.bilinear_rank}, weight={args.bilinear_residual_weight:g})"
        )
    elif args.score_type == "set_aware":
        print(
            f"Score type: {args.score_type} "
            f"(layers={args.set_encoder_layers}, heads={args.set_encoder_heads}, "
            f"ff={args.set_encoder_ff_dim}, weight={args.set_encoder_residual_weight:g}, "
            f"normalize_input={args.set_encoder_normalize_input}, context_only={args.set_encoder_context_only})"
        )
    elif args.score_type == "cross_attention":
        print(
            f"Score type: {args.score_type} "
            f"(heads={args.cross_attention_heads}, weight={args.cross_attention_residual_weight:g})"
        )
    print(f"Best-state selection split: {selection_split}")
    if args.unsafe_select_on_test:
        print("WARNING: using test for model selection. This is only for debugging/upper-bound probing.")

    data = load_data_for_training(
        dataset_dir,
        text_encoder=args.text_encoder,
        vision_encoder=args.vision_encoder,
        query_embedding_file=args.query_embedding_file,
    )
    use_joint_query_embedding = data.get("query_embeddings") is not None
    all_models = list(data["models"])
    model_to_idx = {name: i for i, name in enumerate(all_models)}
    missing_holdouts = [name for name in holdout_models if name not in model_to_idx]
    if missing_holdouts:
        raise ValueError(f"Unknown --holdout_models: {missing_holdouts}")
    holdout_indices = [model_to_idx[name] for name in holdout_models]
    seen_indices = [i for i in range(len(all_models)) if i not in set(holdout_indices)]
    if holdout_models:
        print(f"Open-set holdout models: {holdout_models}")
        print(f"Seen training models ({len(seen_indices)}): {[all_models[i] for i in seen_indices]}")

    print("\nPreparing train data...")
    X_text_train, X_vision_train, Y_train, C_train, meta_train, _, _ = aligned_split(data, "train")
    X_train = X_text_train if use_joint_query_embedding else None
    if use_joint_query_embedding:
        X_text_train = None
    if holdout_models:
        Y_train = Y_train[:, seen_indices]
        C_train = C_train[:, seen_indices]
    print(f"  Train set: {len(meta_train)}")

    print(f"\nPreparing {selection_split} data for best-state selection...")
    X_text_sel, X_vision_sel, Y_sel, C_sel, meta_sel, _, _ = aligned_split(data, selection_split)
    X_sel = X_text_sel if use_joint_query_embedding else None
    if use_joint_query_embedding:
        X_text_sel = None
    if holdout_models:
        Y_sel = Y_sel[:, seen_indices]
        C_sel = C_sel[:, seen_indices]
    print(f"  Selection set ({selection_split}): {len(meta_sel)}")

    if holdout_models:
        if args.eval_candidate_scope == "seen":
            eval_candidate_indices = seen_indices
        elif args.eval_candidate_scope == "holdout":
            eval_candidate_indices = holdout_indices
        else:
            eval_candidate_indices = list(range(len(all_models)))
    else:
        eval_candidate_indices = list(range(len(all_models)))
    eval_candidate_models = [all_models[i] for i in eval_candidate_indices]

    cost_bounds_file = dataset_dir / "data/matrices/cost_bounds.json"
    if cost_bounds_file.exists():
        cmin, cmax = get_cost_bounds_from_config(str(cost_bounds_file))
    else:
        cmin = float(data["C"].min())
        cmax = float(data["C"].max())

    model_costs = C_train.mean(axis=0)
    model_mapping = {new_i: all_models[old_i] for new_i, old_i in enumerate(seen_indices)}

    train_profile_path = profile_path
    holdout_tag = None
    if holdout_models:
        holdout_tag = "holdout_" + short_tag(",".join(holdout_models))
        train_profile_path = write_subset_profile(profile_path, output_dir, seen_indices, holdout_tag)
        print(f"Seen-only training profile: {train_profile_path}")
        print(f"Holdout tag: {holdout_tag}")

    router = ScopeRouter(
        profile_path=str(train_profile_path),
        embedding_dim=args.embedding_dim,
        query_hidden_dim=args.query_hidden_dim,
        profile_hidden_dim=args.profile_hidden_dim,
        query_layers=args.query_layers,
        profile_layers=args.profile_layers,
        dropout=args.dropout,
        fusion_method=args.fusion_method,
        text_weight=args.text_weight,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer_type,
        lr_scheduler=args.lr_scheduler,
        min_lr_factor=args.min_lr_factor,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        lr_plateau_patience=args.lr_plateau_patience,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        temperature=args.temperature,
        score_type=args.score_type,
        bilinear_rank=args.bilinear_rank,
        bilinear_residual_weight=args.bilinear_residual_weight,
        set_encoder_layers=args.set_encoder_layers,
        set_encoder_heads=args.set_encoder_heads,
        set_encoder_ff_dim=args.set_encoder_ff_dim,
        set_encoder_residual_weight=args.set_encoder_residual_weight,
        set_encoder_normalize_input=args.set_encoder_normalize_input,
        set_encoder_context_only=args.set_encoder_context_only,
        cross_attention_heads=args.cross_attention_heads,
        cross_attention_residual_weight=args.cross_attention_residual_weight,
        loss_type=args.loss_type,
        crm_target=args.crm_target,
        crm_bias=args.crm_bias,
        blip_alpha=args.blip_alpha,
        learn_blip_alpha=args.learn_blip_alpha,
        blip_momentum=args.blip_momentum,
        rccr_weight=args.rccr_weight,
        rccr_temperature=args.rccr_temperature,
        learn_rccr_temperature=args.learn_rccr_temperature,
        train_lambda=train_lambda,
        use_soft_labels=use_soft_labels,
        cost_scale=args.cost_scale,
        patience=args.patience,
        monitor_metric=args.monitor_metric,
        random_state=args.seed,
        verbose=1,
    )

    print("\nTraining...")
    router.fit(
        Y_train,
        C_train,
        meta_train,
        X=X_train,
        X_text=X_text_train,
        X_vision=X_vision_train,
        Y_dev=Y_sel,
        C_dev=C_sel,
        meta_dev=meta_sel,
        X_dev=X_sel,
        X_text_dev=X_text_sel,
        X_vision_dev=X_vision_sel,
        model_mapping=model_mapping,
        costs=model_costs,
        cmin=cmin,
        cmax=cmax,
        rank_score_beta=0.1,
    )

    final_blip_alpha = None
    if args.loss_type == "blip" and getattr(router, "blip_alpha_logit", None) is not None:
        try:
            import torch

            with torch.no_grad():
                final_blip_alpha = float(torch.sigmoid(router.blip_alpha_logit).detach().cpu())
            print(f"Final learned BLIP alpha: {final_blip_alpha:.6f}")
        except Exception:
            final_blip_alpha = None

    label_tag = "hard" if args.hard_labels else f"softlambda{format_float_for_name(train_lambda)}"
    score_tag = ""
    if args.score_type == "lowrank_bilinear":
        score_tag = (
            f"_score{args.score_type}"
            f"_br{args.bilinear_rank}"
            f"_bw{format_float_for_name(args.bilinear_residual_weight)}"
        )
    elif args.score_type == "set_aware":
        score_tag = (
            f"_score{args.score_type}"
            f"_sl{args.set_encoder_layers}"
            f"_sh{args.set_encoder_heads}"
            f"_sf{args.set_encoder_ff_dim}"
            f"_sw{format_float_for_name(args.set_encoder_residual_weight)}"
            f"_sn{int(args.set_encoder_normalize_input)}"
            f"_sc{int(args.set_encoder_context_only)}"
        )
    elif args.score_type == "cross_attention":
        score_tag = (
            f"_score{args.score_type}"
            f"_ch{args.cross_attention_heads}"
            f"_cw{format_float_for_name(args.cross_attention_residual_weight)}"
        )
    config_tag = (
        f"loss{args.loss_type}"
        f"_fuse{args.fusion_method}"
        f"_crmt{args.crm_target}"
        f"_crmb{args.crm_bias}"
        f"_blipa{format_float_for_name(args.blip_alpha)}"
        f"_blipalearn{int(args.learn_blip_alpha)}"
        f"_blipm{format_float_for_name(args.blip_momentum)}"
        f"_rccrw{format_float_for_name(args.rccr_weight)}"
        f"_rccrt{format_float_for_name(args.rccr_temperature)}"
        f"_rccrtlearn{int(args.learn_rccr_temperature)}"
        f"_"
        f"ed{args.embedding_dim}"
        f"_qh{args.query_hidden_dim}"
        f"_ph{args.profile_hidden_dim}"
        f"_d{format_float_for_name(args.dropout)}"
        f"_wd{format_float_for_name(args.weight_decay)}"
        f"_opt{args.optimizer_type}"
        f"_sched{args.lr_scheduler}"
        f"_lr{format_float_for_name(args.learning_rate)}"
        f"_minlr{format_float_for_name(args.min_lr_factor)}"
        f"_temp{format_float_for_name(args.temperature)}"
        f"{score_tag}"
    )
    unsafe_tag = "_unsafe_test_selected" if args.unsafe_select_on_test else ""
    if holdout_models:
        profile_tag = compact_profile_tag(profile_path)
        compact_config_tag = (
            f"loss{args.loss_type}_fuse{args.fusion_method}_crmt{args.crm_target}"
            f"_rccrw{format_float_for_name(args.rccr_weight)}"
            f"_rccrt{format_float_for_name(args.rccr_temperature)}"
            f"_rccrtlearn{int(args.learn_rccr_temperature)}"
            f"_ed{args.embedding_dim}_qh{args.query_hidden_dim}_ph{args.profile_hidden_dim}"
            f"_d{format_float_for_name(args.dropout)}_wd{format_float_for_name(args.weight_decay)}"
            f"_opt{args.optimizer_type}_sched{args.lr_scheduler}_lr{format_float_for_name(args.learning_rate)}"
            f"{score_tag}"
        )
        model_name = (
            f"scope_router_openset_{profile_tag}_{label_tag}_"
            f"{compact_config_tag}{unsafe_tag}_{holdout_tag}_scope{args.eval_candidate_scope}"
        )
    else:
        model_name = f"scope_router_{Path(args.profile_path).stem}_{label_tag}_{config_tag}{unsafe_tag}"
    original_model_name = model_name
    model_name = compact_output_stem(model_name)
    if model_name != original_model_name:
        print(f"Shortened artifact stem to avoid filename-length limits: {model_name}")
    model_path = output_dir / f"{model_name}.pkl"
    router.save(str(model_path))
    print(f"\nModel saved: {model_path}")

    if holdout_models:
        attach_candidate_profile_subset(router, profile_path, all_models, eval_candidate_indices)
        print(
            f"Attached {args.eval_candidate_scope} candidate profiles for open-set eval: "
            f"{len(eval_candidate_indices)} models"
        )

    print("\nEvaluating splits...")
    results = {}
    for split_name in ["train", "dev", "test"]:
        if split_name not in data["splits"]:
            continue
        Y_split, C_split, meta_split, embedding_indices = split_arrays_for_eval(data, split_name)
        if len(meta_split) == 0:
            continue
        result = evaluate_router_with_indices(
            router,
            Y_split,
            C_split,
            meta_split,
            data["models"],
            data["text_embeddings"],
            data["vision_embeddings"],
            embedding_indices,
            cost_bounds_file=cost_bounds_file if cost_bounds_file.exists() else None,
            query_embeddings=data.get("query_embeddings"),
        )
        results[split_name] = result
        print(f"  {split_name}: acc={result['accuracy']:.4f} cost=${result['avg_cost']:.6f} "
              f"rank_score={result.get('rank_score', -1):.4f}")

    dataset_results = []
    dataset_results_df = pd.DataFrame()
    if "test" in data["splits"]:
        Y_test, C_test, meta_test, test_embedding_indices = split_arrays_for_eval(data, "test")
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
            query_embeddings=data.get("query_embeddings"),
        )
        if not dataset_results_df.empty:
            dataset_results = dataset_results_df.to_dict("records")
            dataset_csv = output_dir / f"{model_name}_test_by_dataset.csv"
            dataset_results_df.to_csv(dataset_csv, index=False, float_format="%.6f")
            print(f"Per-dataset results saved: {dataset_csv}")

    latency_metrics = {}
    if not args.skip_latency and "test" in data["splits"]:
        print("\nProfiling latency...")
        if "test_embedding_indices" not in locals():
            _, _, meta_test, test_embedding_indices = split_arrays_for_eval(data, "test")
        latency_metrics = profile_router_latency(
            router=router,
            X=data["query_embeddings"][test_embedding_indices] if use_joint_query_embedding else None,
            X_text=None if use_joint_query_embedding else data["text_embeddings"][test_embedding_indices],
            X_vision=None if use_joint_query_embedding else data["vision_embeddings"][test_embedding_indices],
            meta=meta_test,
            batch_size=16,
            warmup_runs=5,
            test_runs=50,
        )
        print(f"  Latency: {latency_metrics.get('ms_per_sample', -1):.3f} ms/sample")

    incremental_results = []
    incremental_by_dataset = []
    incremental_eval_csv = None
    incremental_by_dataset_csv = None
    if args.incremental_holdout_eval and "test" in data["splits"]:
        print("\nIncremental holdout evaluation...")
        if "Y_test" not in locals() or "C_test" not in locals() or "meta_test" not in locals():
            Y_test, C_test, meta_test, test_embedding_indices = split_arrays_for_eval(data, "test")

        ordered_holdout_indices = [model_to_idx[name] for name in incremental_holdout_order]
        for step in range(len(ordered_holdout_indices) + 1):
            candidate_indices = seen_indices + ordered_holdout_indices[:step]
            attach_candidate_profile_subset(router, profile_path, all_models, candidate_indices)

            result = evaluate_router_with_indices(
                router,
                Y_test,
                C_test,
                meta_test,
                data["models"],
                data["text_embeddings"],
                data["vision_embeddings"],
                test_embedding_indices,
                cost_bounds_file=cost_bounds_file if cost_bounds_file.exists() else None,
                query_embeddings=data.get("query_embeddings"),
            )
            row = {
                "step": step,
                "num_candidates": len(candidate_indices),
                "added_holdout_models": ",".join(incremental_holdout_order[:step]),
                "latest_added_model": incremental_holdout_order[step - 1] if step else "",
                **result,
            }
            incremental_results.append(row)
            print(
                f"  step={step} candidates={len(candidate_indices)} "
                f"acc={result['accuracy']:.4f} cost=${result['avg_cost']:.6f} "
                f"rank_score={result.get('rank_score', -1):.4f}"
            )

            ds = evaluate_by_dataset(
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
                query_embeddings=data.get("query_embeddings"),
            )
            if not ds.empty:
                ds["step"] = step
                ds["num_candidates"] = len(candidate_indices)
                ds["added_holdout_models"] = ",".join(incremental_holdout_order[:step])
                ds["latest_added_model"] = incremental_holdout_order[step - 1] if step else ""
                incremental_by_dataset.append(ds)

        incremental_eval_csv = output_dir / f"{model_name}_incremental_eval.csv"
        pd.DataFrame(incremental_results).to_csv(incremental_eval_csv, index=False)
        print(f"Incremental eval saved: {incremental_eval_csv}")
        if incremental_by_dataset:
            incremental_by_dataset_csv = output_dir / f"{model_name}_incremental_eval_by_dataset.csv"
            pd.concat(incremental_by_dataset, ignore_index=True).to_csv(
                incremental_by_dataset_csv,
                index=False,
                float_format="%.6f",
            )
            print(f"Incremental by-dataset eval saved: {incremental_by_dataset_csv}")

        attach_candidate_profile_subset(router, profile_path, all_models, eval_candidate_indices)

    report = {
        "model": model_name,
        "profile_path": str(profile_path),
        "selection_split": selection_split,
        "unsafe_select_on_test": bool(args.unsafe_select_on_test),
        "open_set": bool(holdout_models),
        "holdout_models": holdout_models,
        "seen_models": [all_models[i] for i in seen_indices],
        "eval_candidate_scope": args.eval_candidate_scope if holdout_models else "all",
        "eval_candidate_models": eval_candidate_models,
        "best_epoch": router.best_epoch,
        "best_selection_metrics": router.best_dev_metrics,
        "final_blip_alpha": final_blip_alpha,
        "hyperparameters": vars(args),
        "results": results,
        "results_by_dataset": dataset_results,
        "incremental_holdout_eval": incremental_results,
        "incremental_holdout_eval_csv": str(incremental_eval_csv) if incremental_eval_csv else None,
        "incremental_holdout_eval_by_dataset_csv": (
            str(incremental_by_dataset_csv) if incremental_by_dataset_csv else None
        ),
        "latency": latency_metrics,
    }
    report_path = output_dir / f"{model_name}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report saved: {report_path}")

    test_results = results.get("test", {})
    summary = {
        "router": model_name,
        "selection_split": selection_split,
        "unsafe_select_on_test": bool(args.unsafe_select_on_test),
        "open_set": bool(holdout_models),
        "holdout_models": ",".join(holdout_models),
        "eval_candidate_scope": args.eval_candidate_scope if holdout_models else "all",
        "num_candidate_models": len(eval_candidate_models),
        "accuracy": test_results.get("accuracy", 0.0),
        "avg_quality": test_results.get("accuracy", 0.0),
        "avg_cost": test_results.get("avg_cost", 0.0),
        "rank_score": test_results.get("rank_score", -1),
        "final_blip_alpha": final_blip_alpha if final_blip_alpha is not None else -1,
        "num_samples": test_results.get("num_samples", 0),
        "num_correct": test_results.get("num_correct", 0),
        "latency_ms_per_sample": latency_metrics.get("ms_per_sample", -1),
    }
    summary_path = output_dir / f"{model_name}_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path}")

    print("\nDone.")
    return report


if __name__ == "__main__":
    main()
