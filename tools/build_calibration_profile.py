#!/usr/bin/env python3
"""
Build model calibration profiles from a selected calibration set.

The behavior-only profile captures how each model performs on calibration
questions. With --include-query-embeddings, the profile also captures which
query regions each model tends to handle correctly or incorrectly.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def encoder_to_filename(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_matrices(dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    matrices_dir = dataset_dir / "data" / "matrices"
    y_path = matrices_dir / "Y.npz"
    c_path = matrices_dir / "C.npy"
    sample_ids_path = matrices_dir / "sample_ids.pkl"
    model_names_path = matrices_dir / "model_names.pkl"

    missing = [p for p in [y_path, c_path, sample_ids_path, model_names_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing matrix files: " + ", ".join(str(p) for p in missing))

    y_data = np.load(y_path)
    if "Y" not in y_data:
        raise ValueError(f"{y_path} must contain an array named 'Y'")
    Y = y_data["Y"].astype(np.float32)
    C = np.load(c_path).astype(np.float32)

    with sample_ids_path.open("rb") as f:
        sample_ids = pickle.load(f)
    with model_names_path.open("rb") as f:
        model_names = pickle.load(f)

    if Y.shape != C.shape:
        raise ValueError(f"Y and C shape mismatch: {Y.shape} vs {C.shape}")
    if len(sample_ids) != Y.shape[0]:
        raise ValueError(f"sample_ids length {len(sample_ids)} does not match Y rows {Y.shape[0]}")
    if len(model_names) != Y.shape[1]:
        raise ValueError(f"model_names length {len(model_names)} does not match Y columns {Y.shape[1]}")

    return Y, C, list(sample_ids), list(model_names)


def load_cost_bounds(dataset_dir: Path, C: np.ndarray) -> Tuple[float, float, str]:
    bounds_path = dataset_dir / "data" / "matrices" / "cost_bounds.json"
    if bounds_path.exists():
        data = json.loads(bounds_path.read_text(encoding="utf-8"))
        cmin = float(data["cmin"])
        cmax = float(data["cmax"])
        source = str(bounds_path)
    else:
        cmin = float(np.nanmin(C))
        cmax = float(np.nanmax(C))
        source = "matrix_minmax"

    if cmax <= cmin:
        cmax = cmin + 1.0
    return cmin, cmax, source


def normalize_cost(C: np.ndarray, cmin: float, cmax: float) -> np.ndarray:
    return np.clip((C - cmin) / max(cmax - cmin, 1e-12), 0.0, 1.0).astype(np.float32)


def safe_weighted_mean(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    denom = float(weights.sum())
    if denom <= 1e-12:
        return np.zeros(X.shape[1], dtype=np.float32)
    return (X * weights[:, None]).sum(axis=0).astype(np.float32) / denom


def component(start: int, name: str, width: int, description: str) -> Tuple[Dict, int]:
    end = start + width
    return {
        "name": name,
        "start": start,
        "end": end,
        "width": width,
        "description": description,
    }, end


def build_behavior_profile(
    Y_calib: np.ndarray,
    C_calib: np.ndarray,
    cost_norm: np.ndarray,
) -> Tuple[np.ndarray, pd.DataFrame, List[Dict]]:
    S, K = Y_calib.shape
    correct = Y_calib.T.astype(np.float32)
    cost = cost_norm.T.astype(np.float32)
    value = (Y_calib * (1.0 - cost_norm)).T.astype(np.float32)

    correct_count = correct.sum(axis=1)
    wrong_count = S - correct_count
    accuracy = correct.mean(axis=1)
    avg_cost = C_calib.mean(axis=0)
    avg_cost_norm = cost.mean(axis=1)
    avg_value = value.mean(axis=1)

    avg_correct_cost = np.zeros(K, dtype=np.float32)
    avg_wrong_cost = np.zeros(K, dtype=np.float32)
    for j in range(K):
        correct_mask = Y_calib[:, j] > 0.5
        wrong_mask = ~correct_mask
        avg_correct_cost[j] = float(C_calib[correct_mask, j].mean()) if correct_mask.any() else 0.0
        avg_wrong_cost[j] = float(C_calib[wrong_mask, j].mean()) if wrong_mask.any() else 0.0

    stats = np.column_stack([
        accuracy,
        avg_cost,
        avg_cost_norm,
        avg_correct_cost,
        avg_wrong_cost,
        avg_value,
        correct_count / max(S, 1),
        wrong_count / max(S, 1),
    ]).astype(np.float32)

    profile = np.hstack([correct, cost, value, stats]).astype(np.float32)

    layout = []
    pos = 0
    item, pos = component(pos, "correct_vec", S, "Per-calibration-sample correctness, shape K x S")
    layout.append(item)
    item, pos = component(pos, "cost_norm_vec", S, "Per-calibration-sample normalized cost, shape K x S")
    layout.append(item)
    item, pos = component(pos, "value_vec", S, "Correctness times inverse normalized cost, shape K x S")
    layout.append(item)
    item, pos = component(
        pos,
        "behavior_stats",
        stats.shape[1],
        "accuracy, avg_cost, avg_cost_norm, avg_correct_cost, avg_wrong_cost, avg_value, correct_frac, wrong_frac",
    )
    layout.append(item)

    stats_df = pd.DataFrame({
        "accuracy": accuracy,
        "avg_cost": avg_cost,
        "avg_cost_norm": avg_cost_norm,
        "avg_correct_cost": avg_correct_cost,
        "avg_wrong_cost": avg_wrong_cost,
        "avg_value": avg_value,
        "correct_count": correct_count.astype(np.int32),
        "wrong_count": wrong_count.astype(np.int32),
    })
    return profile, stats_df, layout


def load_embedding_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding file: {path}")
    df = pd.read_parquet(path)
    if "sample_id" not in df.columns or "embedding" not in df.columns:
        raise ValueError(f"{path} must contain columns: sample_id, embedding")
    return df[["sample_id", "embedding"]].copy()


def load_query_embeddings(
    dataset_dir: Path,
    sample_ids: List[str],
    query_embedding_file: Optional[Path],
    text_encoder: str,
    vision_encoder: str,
    fusion_method: str,
    text_weight: float,
) -> Tuple[np.ndarray, Dict]:
    if query_embedding_file is not None:
        emb_df = load_embedding_parquet(query_embedding_file)
        source = {"type": "query_embedding_file", "path": str(query_embedding_file)}
    else:
        text_name = encoder_to_filename(text_encoder)
        vision_name = encoder_to_filename(vision_encoder)
        text_path = dataset_dir / "EMBEDDINGS" / "text" / f"{text_name}.parquet"
        vision_path = dataset_dir / "EMBEDDINGS" / "vision" / f"{vision_name}.parquet"

        text_df = load_embedding_parquet(text_path).rename(columns={"embedding": "embedding_text"})
        vision_df = load_embedding_parquet(vision_path).rename(columns={"embedding": "embedding_vision"})
        emb_df = text_df.merge(vision_df, on="sample_id", how="inner")

        from routers.utils.fusion import fuse_embeddings

        text_emb = np.vstack(emb_df["embedding_text"].values).astype(np.float32)
        vision_emb = np.vstack(emb_df["embedding_vision"].values).astype(np.float32)
        fused = fuse_embeddings(text_emb, vision_emb, method=fusion_method, text_weight=text_weight)
        emb_df = emb_df[["sample_id"]].copy()
        emb_df["embedding"] = list(fused.astype(np.float32))
        source = {
            "type": "fused_text_vision",
            "text_encoder": text_encoder,
            "vision_encoder": vision_encoder,
            "fusion_method": fusion_method,
            "text_weight": text_weight,
            "text_path": str(text_path),
            "vision_path": str(vision_path),
        }

    emb_map = dict(zip(emb_df["sample_id"], emb_df["embedding"]))
    missing = [sid for sid in sample_ids if sid not in emb_map]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Missing query embeddings for {len(missing)} calibration samples: {preview}")

    query_embeddings = np.vstack([emb_map[sid] for sid in sample_ids]).astype(np.float32)
    source["dimension"] = int(query_embeddings.shape[1])
    return query_embeddings, source


def build_query_profile(
    Y_calib: np.ndarray,
    cost_norm: np.ndarray,
    query_embeddings: np.ndarray,
    start_pos: int,
) -> Tuple[np.ndarray, List[Dict]]:
    S, K = Y_calib.shape
    D = query_embeddings.shape[1]
    query_norm = query_embeddings / np.maximum(
        np.linalg.norm(query_embeddings, axis=1, keepdims=True),
        1e-12,
    )
    query_norm = query_norm.astype(np.float32)

    correct_means = np.zeros((K, D), dtype=np.float32)
    wrong_means = np.zeros((K, D), dtype=np.float32)
    value_means = np.zeros((K, D), dtype=np.float32)

    for j in range(K):
        correct_weights = Y_calib[:, j].astype(np.float32)
        wrong_weights = 1.0 - correct_weights
        value_weights = correct_weights * (1.0 - cost_norm[:, j].astype(np.float32))
        correct_means[j] = safe_weighted_mean(query_norm, correct_weights)
        wrong_means[j] = safe_weighted_mean(query_norm, wrong_weights)
        value_means[j] = safe_weighted_mean(query_norm, value_weights)

    query_profile = np.hstack([correct_means, wrong_means, value_means]).astype(np.float32)

    layout = []
    pos = start_pos
    item, pos = component(pos, "query_correct_mean", D, "Mean query embedding over samples the model answers correctly")
    layout.append(item)
    item, pos = component(pos, "query_wrong_mean", D, "Mean query embedding over samples the model answers incorrectly")
    layout.append(item)
    item, pos = component(pos, "query_value_mean", D, "Mean query embedding weighted by correctness and low cost")
    layout.append(item)

    return query_profile, layout


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build calibration model profiles")
    parser.add_argument("--dataset-dir", default=".", help="Dataset root directory")
    parser.add_argument("--calibration-file", required=True, help="Calibration JSONL from select_calibration_set.py")
    parser.add_argument("--output-dir", default="CALIBRATION", help="Output directory")
    parser.add_argument("--name", default=None, help="Output name prefix")
    parser.add_argument("--include-query-embeddings", action="store_true",
                        help="Append query-aware aggregate embedding features to the model profile")
    parser.add_argument("--query-embedding-file", default=None,
                        help="Optional parquet with sample_id and embedding columns, e.g. Qwen3-VL query embeddings")
    parser.add_argument("--text-encoder", default="BAAI/bge-m3",
                        help="Text encoder used when --query-embedding-file is not provided")
    parser.add_argument("--vision-encoder", default="facebook/dinov2-base",
                        help="Vision encoder used when --query-embedding-file is not provided")
    parser.add_argument("--fusion-method", default="normalize_concat",
                        help="Fusion method used for text/vision embeddings")
    parser.add_argument("--text-weight", type=float, default=0.5,
                        help="Text weight for weighted_average fusion")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    calibration_file = Path(args.calibration_file)
    if not calibration_file.is_absolute():
        calibration_file = dataset_dir / calibration_file
    output_dir = dataset_dir / args.output_dir

    calibration_rows = read_jsonl(calibration_file)
    if not calibration_rows:
        raise ValueError(f"No rows found in calibration file: {calibration_file}")
    calib_df = pd.DataFrame(calibration_rows)
    if "sample_id" not in calib_df.columns:
        raise ValueError(f"{calibration_file} must contain sample_id")
    calibration_sample_ids = calib_df["sample_id"].astype(str).tolist()

    Y, C, all_sample_ids, model_names = load_matrices(dataset_dir)
    row_map = {sid: idx for idx, sid in enumerate(all_sample_ids)}
    missing = [sid for sid in calibration_sample_ids if sid not in row_map]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} calibration samples are missing from matrices: {preview}")

    matrix_indices = np.array([row_map[sid] for sid in calibration_sample_ids], dtype=np.int64)
    Y_calib = Y[matrix_indices].astype(np.float32)
    C_calib = C[matrix_indices].astype(np.float32)
    cmin, cmax, cost_bounds_source = load_cost_bounds(dataset_dir, C)
    cost_norm = normalize_cost(C_calib, cmin, cmax)

    behavior_profile, stats_df, layout = build_behavior_profile(Y_calib, C_calib, cost_norm)
    model_profile = behavior_profile

    query_embeddings = None
    query_embedding_source = None
    if args.include_query_embeddings:
        query_path = Path(args.query_embedding_file) if args.query_embedding_file else None
        if query_path is not None and not query_path.is_absolute():
            query_path = dataset_dir / query_path
        query_embeddings, query_embedding_source = load_query_embeddings(
            dataset_dir=dataset_dir,
            sample_ids=calibration_sample_ids,
            query_embedding_file=query_path,
            text_encoder=args.text_encoder,
            vision_encoder=args.vision_encoder,
            fusion_method=args.fusion_method,
            text_weight=args.text_weight,
        )
        query_profile, query_layout = build_query_profile(
            Y_calib=Y_calib,
            cost_norm=cost_norm,
            query_embeddings=query_embeddings,
            start_pos=model_profile.shape[1],
        )
        model_profile = np.hstack([model_profile, query_profile]).astype(np.float32)
        layout.extend(query_layout)

    mode = "query_aware" if args.include_query_embeddings else "behavior_only"
    default_name = f"{calibration_file.stem}_{mode}_profile"
    name = args.name or default_name
    npz_path = output_dir / f"{name}.npz"
    meta_path = output_dir / f"{name}_manifest.json"
    stats_path = output_dir / f"{name}_model_stats.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "model_profile": model_profile.astype(np.float32),
        "behavior_profile": behavior_profile.astype(np.float32),
        "Y_calib": Y_calib.astype(np.float32),
        "C_calib": C_calib.astype(np.float32),
        "cost_norm_calib": cost_norm.astype(np.float32),
        "sample_ids": np.array(calibration_sample_ids, dtype=str),
        "model_names": np.array(model_names, dtype=str),
        "matrix_indices": matrix_indices,
    }
    if query_embeddings is not None:
        arrays["query_embeddings"] = query_embeddings.astype(np.float32)
    np.savez_compressed(npz_path, **arrays)

    stats_df.insert(0, "model_name", model_names)
    stats_df.to_csv(stats_path, index=False)

    manifest = {
        "name": name,
        "mode": mode,
        "calibration_file": str(calibration_file),
        "num_calibration_samples": int(len(calibration_sample_ids)),
        "num_models": int(len(model_names)),
        "model_profile_shape": list(model_profile.shape),
        "behavior_profile_shape": list(behavior_profile.shape),
        "query_embeddings_included": bool(args.include_query_embeddings),
        "query_embedding_source": query_embedding_source,
        "cost_bounds": {
            "cmin": cmin,
            "cmax": cmax,
            "source": cost_bounds_source,
        },
        "profile_layout": layout,
        "outputs": {
            "npz": str(npz_path),
            "manifest": str(meta_path),
            "model_stats_csv": str(stats_path),
        },
        "profile_note": (
            "Use mode=behavior_only for calibration behavior without question embeddings; "
            "use mode=query_aware to append question embedding aggregates."
        ),
    }
    for col in ["dataset", "task"]:
        if col in calib_df.columns:
            manifest[f"{col}_counts"] = {
                str(k): int(v) for k, v in calib_df[col].value_counts().sort_index().items()
            }
    write_json(meta_path, manifest)

    print("=" * 80)
    print("Calibration profile built")
    print("=" * 80)
    print(f"  Mode: {mode}")
    print(f"  Samples: {len(calibration_sample_ids)}")
    print(f"  Models: {len(model_names)}")
    print(f"  Model profile shape: {model_profile.shape}")
    print(f"  Output: {npz_path}")
    print(f"  Manifest: {meta_path}")
    print(f"  Model stats: {stats_path}")
    if query_embeddings is not None:
        print(f"  Query embedding shape: {query_embeddings.shape}")


if __name__ == "__main__":
    main()
