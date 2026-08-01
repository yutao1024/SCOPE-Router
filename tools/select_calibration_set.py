#!/usr/bin/env python3
"""
Select a small calibration set from the train split.

The calibration set is intended for model-agnostic routing: new models can be
run on this small subset to build a model behavior embedding without touching
dev/test data.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def encoder_to_filename(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def read_jsonl_sample_ids(path: Path) -> List[str]:
    ids = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                ids.append(rec["sample_id"])
    return ids


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def allocate_counts(groups: pd.Series, total_size: int, min_per_dataset: int) -> Dict[str, int]:
    """Allocate calibration samples across datasets with a floor and proportional remainder."""
    counts = groups.to_dict()
    datasets = sorted(counts)

    if total_size <= 0:
        raise ValueError("--size must be positive")
    if total_size < len(datasets):
        raise ValueError(f"--size={total_size} is smaller than dataset count={len(datasets)}")

    allocation = {ds: min(min_per_dataset, counts[ds]) for ds in datasets}
    current = sum(allocation.values())

    if current > total_size:
        # Reduce floors evenly when requested size is too small for min_per_dataset.
        allocation = {ds: 1 for ds in datasets}
        current = len(datasets)

    remaining = total_size - current
    capacities = {ds: counts[ds] - allocation[ds] for ds in datasets}

    if remaining <= 0:
        return allocation

    total_capacity = sum(max(0, v) for v in capacities.values())
    if total_capacity <= 0:
        return allocation

    raw = {}
    for ds in datasets:
        cap = max(0, capacities[ds])
        raw[ds] = remaining * cap / total_capacity
        allocation[ds] += int(math.floor(raw[ds]))

    assigned = sum(allocation.values())
    leftovers = min(total_size, sum(counts.values())) - assigned
    if leftovers > 0:
        ranked = sorted(
            datasets,
            key=lambda ds: (raw[ds] - math.floor(raw[ds]), capacities[ds]),
            reverse=True,
        )
        for ds in ranked:
            if leftovers <= 0:
                break
            if allocation[ds] < counts[ds]:
                allocation[ds] += 1
                leftovers -= 1

    return allocation


def allocate_component_counts(total: int, ratios: Dict[str, float]) -> Dict[str, int]:
    if total <= 0:
        return {name: 0 for name in ratios}
    positive = {name: max(0.0, float(value)) for name, value in ratios.items()}
    ratio_sum = sum(positive.values())
    if ratio_sum <= 0:
        raise ValueError("Hybrid ratios must sum to a positive value")

    raw = {name: total * value / ratio_sum for name, value in positive.items()}
    counts = {name: int(math.floor(value)) for name, value in raw.items()}
    leftovers = total - sum(counts.values())
    ranked = sorted(raw, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in ranked:
        if leftovers <= 0:
            break
        counts[name] += 1
        leftovers -= 1
    return counts


def sample_random(group_df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if n >= len(group_df):
        return group_df.copy()
    idx = rng.choice(group_df.index.to_numpy(), size=n, replace=False)
    return group_df.loc[idx].copy()


def parse_model_list(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_matrix_bundle(
    dataset_dir: Path,
    include_models: Optional[List[str]] = None,
    exclude_models: Optional[List[str]] = None,
    disagreement_weight: float = 0.8,
    cost_weight: float = 0.2,
) -> pd.DataFrame:
    """Load sample-level model behavior used for diagnostic calibration selection."""
    matrices_dir = dataset_dir / "data" / "matrices"
    y_path = matrices_dir / "Y.npz"
    c_path = matrices_dir / "C.npy"
    sample_ids_path = matrices_dir / "sample_ids.pkl"
    model_names_path = matrices_dir / "model_names.pkl"

    if not y_path.exists() or not c_path.exists() or not sample_ids_path.exists():
        raise FileNotFoundError(
            "Diagnostic selection requires data/matrices/Y.npz, C.npy, and sample_ids.pkl"
        )

    y_data = np.load(y_path)
    if "Y" not in y_data:
        raise ValueError(f"{y_path} must contain an array named 'Y'")
    Y = y_data["Y"].astype(np.float32)
    C = np.load(c_path).astype(np.float32)

    with sample_ids_path.open("rb") as f:
        import pickle

        sample_ids = pickle.load(f)

    model_names = None
    if include_models or exclude_models:
        if not model_names_path.exists():
            raise FileNotFoundError(f"Model filtering requires {model_names_path}")
        with model_names_path.open("rb") as f:
            model_names = list(pickle.load(f))
        model_to_idx = {name: idx for idx, name in enumerate(model_names)}
        requested = include_models if include_models else [m for m in model_names if m not in set(exclude_models or [])]
        missing = [name for name in requested if name not in model_to_idx]
        if missing:
            raise ValueError(f"Unknown model name(s) for diagnostic selection: {missing}")
        selected_cols = [model_to_idx[name] for name in requested]
        if not selected_cols:
            raise ValueError("Model filtering removed every model; at least one seen model is required")
        Y = Y[:, selected_cols]
        C = C[:, selected_cols]

    if Y.shape != C.shape:
        raise ValueError(f"Y and C shape mismatch: {Y.shape} vs {C.shape}")
    if len(sample_ids) != Y.shape[0]:
        raise ValueError(f"sample_ids length {len(sample_ids)} does not match Y rows {Y.shape[0]}")

    K = Y.shape[1]
    correct_count = Y.sum(axis=1)
    difficulty = correct_count / max(K, 1)
    # Bernoulli variance peaks at p=0.5; easy-all-correct and impossible-all-wrong
    # samples are less diagnostic for distinguishing model behavior.
    disagreement = 4.0 * difficulty * (1.0 - difficulty)

    correct_mask = Y > 0.5
    cost_spread = np.zeros(Y.shape[0], dtype=np.float32)
    rows_with_correct = np.where(correct_count > 0)[0]
    if len(rows_with_correct) > 0:
        correct_costs = np.where(correct_mask[rows_with_correct], C[rows_with_correct], np.nan)
        min_correct_cost = np.nanmin(correct_costs, axis=1)
        max_correct_cost = np.nanmax(correct_costs, axis=1)
        cost_spread[rows_with_correct] = np.nan_to_num(
            max_correct_cost - min_correct_cost,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    # Normalize cost spread robustly so it can help but not dominate.
    positive_spread = cost_spread[cost_spread > 0]
    if len(positive_spread) > 0:
        denom = np.percentile(positive_spread, 95)
        if denom <= 0:
            denom = float(positive_spread.max())
    else:
        denom = 1.0
    cost_spread_norm = np.clip(cost_spread / max(float(denom), 1e-12), 0.0, 1.0)

    score_weight_sum = disagreement_weight + cost_weight
    if score_weight_sum <= 0:
        raise ValueError("Diagnostic disagreement/cost weights must sum to a positive value")
    disagreement_weight = disagreement_weight / score_weight_sum
    cost_weight = cost_weight / score_weight_sum

    # A diagnostic sample should distinguish model correctness first, with a
    # secondary preference for samples where correct models have different costs.
    diagnostic_score = disagreement_weight * disagreement + cost_weight * cost_spread_norm

    behavior = pd.DataFrame({
        "sample_id": sample_ids,
        "correct_count": correct_count.astype(np.int16),
        "difficulty": difficulty.astype(np.float32),
        "disagreement": disagreement.astype(np.float32),
        "cost_spread": cost_spread.astype(np.float32),
        "cost_spread_norm": cost_spread_norm.astype(np.float32),
        "diagnostic_score": diagnostic_score.astype(np.float32),
    })
    return behavior


def assign_difficulty_bucket(difficulty: pd.Series) -> pd.Series:
    """Bucket by fraction of existing models that answer correctly."""
    return pd.cut(
        difficulty,
        bins=[-0.001, 0.25, 0.75, 1.001],
        labels=["hard", "medium", "easy"],
    ).astype(str)


def sample_diagnostic(
    group_df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    temperature: float,
    medium_ratio: float = 0.7,
    hard_ratio: float = 0.15,
    easy_ratio: float = 0.15,
) -> pd.DataFrame:
    """Sample high-value diagnostic items while preserving difficulty coverage."""
    if n >= len(group_df):
        return group_df.copy()

    df = group_df.copy()
    df["difficulty_bucket"] = assign_difficulty_bucket(df["difficulty"])

    # Prefer medium-disagreement samples, but keep some easy/hard anchors so the
    # calibration vector still captures model behavior on extremes.
    bucket_weights = {"medium": medium_ratio, "hard": hard_ratio, "easy": easy_ratio}
    available_buckets = [b for b in ["medium", "hard", "easy"] if (df["difficulty_bucket"] == b).any()]
    raw_counts = {}
    for bucket in available_buckets:
        raw_counts[bucket] = n * bucket_weights.get(bucket, 0.0)
    raw_total = sum(raw_counts.values())
    if raw_total <= 0:
        raw_counts = {b: n / len(available_buckets) for b in available_buckets}
        raw_total = n

    bucket_targets = {b: int(np.floor(raw_counts[b] / raw_total * n)) for b in available_buckets}
    leftovers = n - sum(bucket_targets.values())
    ranked_buckets = sorted(
        available_buckets,
        key=lambda b: raw_counts[b] / raw_total * n - bucket_targets[b],
        reverse=True,
    )
    for bucket in ranked_buckets:
        if leftovers <= 0:
            break
        bucket_targets[bucket] += 1
        leftovers -= 1

    selected_indices = []
    for bucket in available_buckets:
        bucket_df = df[df["difficulty_bucket"] == bucket].copy()
        target = min(bucket_targets[bucket], len(bucket_df))
        if target <= 0:
            continue
        selected_indices.extend(weighted_select(bucket_df, target, rng, temperature).index.tolist())

    if len(selected_indices) < n:
        remaining_df = df.drop(index=selected_indices)
        fill = min(n - len(selected_indices), len(remaining_df))
        if fill > 0:
            selected_indices.extend(weighted_select(remaining_df, fill, rng, temperature).index.tolist())

    if len(selected_indices) > n:
        selected_indices = selected_indices[:n]

    return df.loc[selected_indices].drop(columns=["difficulty_bucket"]).copy()


def weighted_select(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    temperature: float,
) -> pd.DataFrame:
    """Weighted without-replacement sample from diagnostic_score."""
    if n >= len(df):
        return df.copy()
    scores = df["diagnostic_score"].to_numpy(dtype=np.float64)
    scores = scores - scores.min()
    if scores.max() > 0:
        scores = scores / scores.max()
    weights = np.exp(scores / max(temperature, 1e-6))
    weights = weights / weights.sum()
    chosen = rng.choice(df.index.to_numpy(), size=n, replace=False, p=weights)
    return df.loc[chosen].copy()


def load_fused_embeddings(dataset_dir: Path, text_encoder: str, vision_encoder: str, fusion_method: str) -> pd.DataFrame:
    text_name = encoder_to_filename(text_encoder)
    vision_name = encoder_to_filename(vision_encoder)
    text_path = dataset_dir / "EMBEDDINGS" / "text" / f"{text_name}.parquet"
    vision_path = dataset_dir / "EMBEDDINGS" / "vision" / f"{vision_name}.parquet"

    if not text_path.exists() or not vision_path.exists():
        raise FileNotFoundError(
            "Embedding files are required for --strategy kmeans/hybrid. Missing: "
            f"{text_path if not text_path.exists() else ''} "
            f"{vision_path if not vision_path.exists() else ''}"
        )

    from routers.utils.fusion import fuse_embeddings

    text_df = pd.read_parquet(text_path)
    vision_df = pd.read_parquet(vision_path)
    emb_df = text_df.merge(vision_df, on="sample_id", suffixes=("_text", "_vision"))

    text_emb = np.vstack(emb_df["embedding_text"].values)
    vision_emb = np.vstack(emb_df["embedding_vision"].values)
    fused = fuse_embeddings(text_emb, vision_emb, method=fusion_method)
    emb_df = emb_df[["sample_id"]].copy()
    emb_df["embedding"] = list(fused.astype(np.float32))
    return emb_df


def sample_kmeans(group_df: pd.DataFrame, n: int, embeddings: Dict[str, np.ndarray], seed: int) -> pd.DataFrame:
    if n >= len(group_df):
        return group_df.copy()

    available = group_df[group_df["sample_id"].isin(embeddings)].copy()
    if len(available) < n:
        raise ValueError(
            f"Not enough embeddings for dataset {group_df['dataset'].iloc[0]}: "
            f"need {n}, found {len(available)}"
        )

    from sklearn.cluster import MiniBatchKMeans

    sample_ids = available["sample_id"].tolist()
    X = np.vstack([embeddings[sid] for sid in sample_ids]).astype(np.float32)
    kmeans = MiniBatchKMeans(
        n_clusters=n,
        random_state=seed,
        batch_size=min(2048, max(256, len(available))),
        n_init="auto",
    )
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    selected_positions = []
    for cluster_idx in range(n):
        positions = np.where(labels == cluster_idx)[0]
        if len(positions) == 0:
            continue
        distances = np.linalg.norm(X[positions] - centers[cluster_idx], axis=1)
        selected_positions.append(int(positions[int(np.argmin(distances))]))

    # Extremely rare empty clusters: fill with farthest remaining points.
    selected = set(selected_positions)
    if len(selected) < n:
        centroid = X.mean(axis=0, keepdims=True)
        ranking = np.argsort(np.linalg.norm(X - centroid, axis=1))[::-1]
        for pos in ranking:
            selected.add(int(pos))
            if len(selected) >= n:
                break

    selected_ids = [sample_ids[pos] for pos in sorted(selected)]
    return available[available["sample_id"].isin(selected_ids)].copy()


def sample_hybrid(
    group_df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    temperature: float,
    embeddings: Dict[str, np.ndarray],
    seed: int,
    random_ratio: float,
    diagnostic_ratio: float,
    diversity_ratio: float,
    diagnostic_medium_ratio: float,
    diagnostic_hard_ratio: float,
    diagnostic_easy_ratio: float,
) -> pd.DataFrame:
    if n >= len(group_df):
        return group_df.copy()

    targets = allocate_component_counts(
        n,
        {
            "random": random_ratio,
            "diagnostic": diagnostic_ratio,
            "diversity": diversity_ratio,
        },
    )

    selected_parts = []
    selected_indices = set()

    def remaining() -> pd.DataFrame:
        if not selected_indices:
            return group_df.copy()
        return group_df.drop(index=list(selected_indices), errors="ignore").copy()

    random_n = min(targets["random"], len(group_df))
    if random_n > 0:
        part = sample_random(remaining(), random_n, rng)
        selected_parts.append(part)
        selected_indices.update(part.index.tolist())

    diagnostic_n = min(targets["diagnostic"], len(remaining()))
    if diagnostic_n > 0:
        part = sample_diagnostic(
            remaining(),
            diagnostic_n,
            rng,
            temperature,
            diagnostic_medium_ratio,
            diagnostic_hard_ratio,
            diagnostic_easy_ratio,
        )
        selected_parts.append(part)
        selected_indices.update(part.index.tolist())

    diversity_n = min(targets["diversity"], len(remaining()))
    if diversity_n > 0:
        part = sample_kmeans(remaining(), diversity_n, embeddings, seed)
        selected_parts.append(part)
        selected_indices.update(part.index.tolist())

    fill = n - sum(len(part) for part in selected_parts)
    if fill > 0:
        part = sample_random(remaining(), min(fill, len(remaining())), rng)
        selected_parts.append(part)

    selected = pd.concat(selected_parts, ignore_index=False).drop_duplicates("sample_id")
    if len(selected) > n:
        selected = selected.iloc[:n].copy()
    return selected.copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Select calibration samples from train split")
    parser.add_argument("--dataset-dir", default=".", help="Dataset root directory")
    parser.add_argument("--size", type=int, default=1024, help="Calibration set size")
    parser.add_argument("--min-per-dataset", type=int, default=16, help="Minimum samples per dataset")
    parser.add_argument("--strategy", choices=["diagnostic", "random", "kmeans", "hybrid"], default="diagnostic",
                        help="Selection strategy. diagnostic uses Y/C model behavior; kmeans/hybrid require EMBEDDINGS/")
    parser.add_argument("--temperature", type=float, default=0.25,
                        help="Sampling temperature for diagnostic selection (lower = greedier)")
    parser.add_argument("--diagnostic-disagreement-weight", type=float, default=0.8,
                        help="For diagnostic/hybrid: weight on model disagreement")
    parser.add_argument("--diagnostic-cost-weight", type=float, default=0.2,
                        help="For diagnostic/hybrid: weight on correct-model cost spread")
    parser.add_argument("--diagnostic-medium-ratio", type=float, default=0.7,
                        help="For diagnostic/hybrid: target ratio for medium-difficulty samples")
    parser.add_argument("--diagnostic-hard-ratio", type=float, default=0.15,
                        help="For diagnostic/hybrid: target ratio for hard samples")
    parser.add_argument("--diagnostic-easy-ratio", type=float, default=0.15,
                        help="For diagnostic/hybrid: target ratio for easy samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="CALIBRATION", help="Output directory")
    parser.add_argument("--name", default=None, help="Output name prefix")
    parser.add_argument("--seen-models", default=None,
                        help="Comma-separated model names to use for diagnostic scoring only")
    parser.add_argument("--exclude-models", default=None,
                        help="Comma-separated model names to exclude from diagnostic scoring")
    parser.add_argument("--text-encoder", default="BAAI/bge-m3")
    parser.add_argument("--vision-encoder", default="facebook/dinov2-base")
    parser.add_argument("--fusion-method", default="normalize_concat")
    parser.add_argument("--hybrid-random-ratio", type=float, default=0.5,
                        help="For --strategy hybrid: stratified random component ratio")
    parser.add_argument("--hybrid-diagnostic-ratio", type=float, default=0.3,
                        help="For --strategy hybrid: behavior-diagnostic component ratio")
    parser.add_argument("--hybrid-diversity-ratio", type=float, default=0.2,
                        help="For --strategy hybrid: embedding-diversity component ratio")
    args = parser.parse_args()
    seen_models = parse_model_list(args.seen_models)
    excluded_models = parse_model_list(args.exclude_models)
    if seen_models and excluded_models:
        raise ValueError("Use only one of --seen-models or --exclude-models")

    dataset_dir = Path(args.dataset_dir)
    output_dir = dataset_dir / args.output_dir

    train_file = dataset_dir / "SPLITS" / "train.jsonl"
    meta_file = dataset_dir / "data" / "registry" / "meta.parquet"
    if not train_file.exists():
        raise FileNotFoundError(f"Missing train split: {train_file}")
    if not meta_file.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_file}")

    train_ids = read_jsonl_sample_ids(train_file)
    train_id_set = set(train_ids)
    meta = pd.read_parquet(meta_file)
    train_meta = meta[meta["sample_id"].isin(train_id_set)].copy()
    if len(train_meta) != len(train_ids):
        print(f"⚠️  Warning: train split has {len(train_ids)} ids, meta matched {len(train_meta)} rows")

    dataset_counts = train_meta.groupby("dataset").size()
    allocation = allocate_counts(dataset_counts, args.size, args.min_per_dataset)
    rng = np.random.default_rng(args.seed)

    embeddings = None
    behavior = None
    if args.strategy in {"diagnostic", "hybrid"}:
        behavior = load_matrix_bundle(
            dataset_dir,
            include_models=seen_models or None,
            exclude_models=excluded_models or None,
            disagreement_weight=args.diagnostic_disagreement_weight,
            cost_weight=args.diagnostic_cost_weight,
        )
        train_meta = train_meta.merge(behavior, on="sample_id", how="left")
        missing_behavior = train_meta["diagnostic_score"].isna().sum()
        if missing_behavior:
            raise ValueError(f"{missing_behavior} train samples are missing Y/C behavior rows")

    if args.strategy in {"kmeans", "hybrid"}:
        emb_df = load_fused_embeddings(dataset_dir, args.text_encoder, args.vision_encoder, args.fusion_method)
        embeddings = dict(zip(emb_df["sample_id"], emb_df["embedding"]))

    selected_parts = []
    for dataset, n in sorted(allocation.items()):
        group_df = train_meta[train_meta["dataset"] == dataset].copy()
        if args.strategy == "diagnostic":
            selected = sample_diagnostic(
                group_df,
                n,
                rng,
                args.temperature,
                args.diagnostic_medium_ratio,
                args.diagnostic_hard_ratio,
                args.diagnostic_easy_ratio,
            )
        elif args.strategy == "random":
            selected = sample_random(group_df, n, rng)
        elif args.strategy == "kmeans":
            selected = sample_kmeans(group_df, n, embeddings, args.seed)
        elif args.strategy == "hybrid":
            selected = sample_hybrid(
                group_df,
                n,
                rng,
                args.temperature,
                embeddings,
                args.seed,
                args.hybrid_random_ratio,
                args.hybrid_diagnostic_ratio,
                args.hybrid_diversity_ratio,
                args.diagnostic_medium_ratio,
                args.diagnostic_hard_ratio,
                args.diagnostic_easy_ratio,
            )
        else:
            raise ValueError(f"Unknown strategy: {args.strategy}")
        selected_parts.append(selected)

    selected_df = pd.concat(selected_parts, ignore_index=True)
    selected_df = selected_df.sort_values(["dataset", "sample_id"]).reset_index(drop=True)
    selected_df["calibration_index"] = np.arange(len(selected_df))

    name = args.name or f"calib_{args.size}_{args.strategy}_seed{args.seed}"
    jsonl_path = output_dir / f"{name}.jsonl"
    csv_path = output_dir / f"{name}_summary.csv"
    manifest_path = output_dir / f"{name}_manifest.json"

    optional_cols = [
        "correct_count",
        "difficulty",
        "disagreement",
        "cost_spread",
        "diagnostic_score",
    ]
    output_cols = ["calibration_index", "sample_id", "dataset", "task"]
    output_cols.extend([col for col in optional_cols if col in selected_df.columns])
    rows = selected_df[output_cols].to_dict("records")
    write_jsonl(jsonl_path, rows)

    summary = (
        selected_df.groupby(["task", "dataset"])
        .size()
        .reset_index(name="num_samples")
        .sort_values(["task", "dataset"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_path, index=False)

    manifest = {
        "name": name,
        "strategy": args.strategy,
        "seed": args.seed,
        "requested_size": args.size,
        "actual_size": int(len(selected_df)),
        "source_split": str(train_file),
        "metadata": str(meta_file),
        "min_per_dataset": args.min_per_dataset,
        "seen_models": seen_models or None,
        "excluded_models": excluded_models or None,
        "text_encoder": args.text_encoder if args.strategy in {"kmeans", "hybrid"} else None,
        "vision_encoder": args.vision_encoder if args.strategy in {"kmeans", "hybrid"} else None,
        "fusion_method": args.fusion_method if args.strategy in {"kmeans", "hybrid"} else None,
        "diagnostic_temperature": args.temperature if args.strategy in {"diagnostic", "hybrid"} else None,
        "diagnostic_score_weights": (
            {
                "disagreement": args.diagnostic_disagreement_weight,
                "cost": args.diagnostic_cost_weight,
            }
            if args.strategy in {"diagnostic", "hybrid"} else None
        ),
        "diagnostic_bucket_ratios": (
            {
                "medium": args.diagnostic_medium_ratio,
                "hard": args.diagnostic_hard_ratio,
                "easy": args.diagnostic_easy_ratio,
            }
            if args.strategy in {"diagnostic", "hybrid"} else None
        ),
        "hybrid_ratios": (
            {
                "random": args.hybrid_random_ratio,
                "diagnostic": args.hybrid_diagnostic_ratio,
                "diversity": args.hybrid_diversity_ratio,
            }
            if args.strategy == "hybrid" else None
        ),
        "outputs": {
            "jsonl": str(jsonl_path),
            "summary_csv": str(csv_path),
        },
        "dataset_counts": {k: int(v) for k, v in selected_df["dataset"].value_counts().sort_index().items()},
        "task_counts": {k: int(v) for k, v in selected_df["task"].value_counts().sort_index().items()},
    }
    if args.strategy in {"diagnostic", "hybrid"}:
        manifest["selection_basis"] = {
            "correct_count": "number of existing models that answered the sample correctly",
            "difficulty": "correct_count / num_models",
            "disagreement": "4 * difficulty * (1 - difficulty), highest when models disagree",
            "cost_spread": "max-min cost among correct models",
            "diagnostic_score": "weighted disagreement + weighted normalized_cost_spread",
            "difficulty_bucket_mix": {
                "medium": args.diagnostic_medium_ratio,
                "hard": args.diagnostic_hard_ratio,
                "easy": args.diagnostic_easy_ratio,
            },
        }
        manifest["diagnostic_stats"] = {
            "avg_correct_count": float(selected_df["correct_count"].mean()),
            "avg_difficulty": float(selected_df["difficulty"].mean()),
            "avg_disagreement": float(selected_df["disagreement"].mean()),
            "avg_diagnostic_score": float(selected_df["diagnostic_score"].mean()),
        }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=" * 80)
    print("✅ Calibration set selected")
    print("=" * 80)
    print(f"  Strategy: {args.strategy}")
    print(f"  Size: {len(selected_df)}")
    print(f"  Output: {jsonl_path}")
    print(f"  Summary: {csv_path}")
    print(f"  Manifest: {manifest_path}")
    print("\nBy task:")
    print(selected_df["task"].value_counts().sort_index().to_string())
    print("\nBy dataset:")
    print(selected_df["dataset"].value_counts().sort_index().to_string())
    if args.strategy == "diagnostic":
        print("\nDiagnostic config:")
        print(
            {
                "temperature": args.temperature,
                "score_weights": {
                    "disagreement": args.diagnostic_disagreement_weight,
                    "cost": args.diagnostic_cost_weight,
                },
                "bucket_ratios": {
                    "medium": args.diagnostic_medium_ratio,
                    "hard": args.diagnostic_hard_ratio,
                    "easy": args.diagnostic_easy_ratio,
                },
            }
        )
        print("\nDiagnostic stats:")
        print(selected_df[["correct_count", "difficulty", "disagreement", "diagnostic_score"]].describe().to_string())
        print("\nDifficulty buckets:")
        print(assign_difficulty_bucket(selected_df["difficulty"]).value_counts().sort_index().to_string())
    if args.strategy == "hybrid":
        print("\nHybrid ratios:")
        print(
            {
                "random": args.hybrid_random_ratio,
                "diagnostic": args.hybrid_diagnostic_ratio,
                "diversity": args.hybrid_diversity_ratio,
            }
        )
        print("\nDiagnostic stats:")
        print(selected_df[["correct_count", "difficulty", "disagreement", "diagnostic_score"]].describe().to_string())
        print("\nDifficulty buckets:")
        print(assign_difficulty_bucket(selected_df["difficulty"]).value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
