"""
Shared dataset loading utilities for multimodal (BENCHMARKS-based) routers.

Used by end-to-end routers that train directly from raw text+image inputs:
- routers/cosinecls/train_and_eval.py
- routers/routerdc/train_and_eval.py
- routers/zooter/train_and_eval.py
- routers/vlc/train_and_eval.py

We keep the behavior compatible with the existing per-router loaders, while
allowing VLC to opt into extra fallbacks (legacy splits, meta fallbacks, etc.).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_samples_from_benchmarks(benchmarks_dir: Path) -> List[dict]:
    """Load all samples under BENCHMARKS/*/*_samples.jsonl."""
    samples: List[dict] = []
    for task_dir in benchmarks_dir.iterdir():
        if not task_dir.is_dir():
            continue
        for samples_file in task_dir.glob("*_samples.jsonl"):
            with open(samples_file, "r", encoding="utf-8") as f:
                for line in f:
                    samples.append(json.loads(line))
    return samples


def resolve_asset_paths(assets: Any, dataset_dir: Path) -> list:
    """Resolve relative image/TSV asset paths against the dataset root."""
    if not isinstance(assets, list):
        return []
    resolved = []
    for asset in assets:
        if not isinstance(asset, dict):
            resolved.append(asset)
            continue
        asset = dict(asset)
        for key in ["tsv_file", "uri", "path"]:
            value = asset.get(key)
            if not value:
                continue
            path = Path(str(value))
            if not path.is_absolute():
                asset[key] = str((dataset_dir / path).resolve())
        resolved.append(asset)
    return resolved


def _load_Y_from_npz(y_file: Path) -> np.ndarray:
    y_data = np.load(y_file)
    if "Y" in y_data.files:
        Y = y_data["Y"]
    elif "data" in y_data.files:
        Y = y_data["data"]
    elif "arr_0" in y_data.files:
        Y = y_data["arr_0"]
    else:
        Y = y_data[y_data.files[0]]
    if sp.issparse(Y):
        Y = Y.toarray()
    return np.asarray(Y)


def load_quality_and_cost_matrices(dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    data_dir = Path(dataset_dir) / "data"
    y_file = data_dir / "matrices/Y.npz"
    c_file = data_dir / "matrices/C.npy"
    if not y_file.exists() or not c_file.exists():
        raise FileNotFoundError(f"Matrix file not found: {y_file} or {c_file}")
    Y = _load_Y_from_npz(y_file)
    C = np.load(c_file)
    return Y, C


def attach_text_assets_from_benchmarks(
    meta: pd.DataFrame,
    benchmarks_dir: Path,
    *,
    verbose: int = 1,
    fallback_on_zero_match: bool = False,
    allow_meta_text_fallback: bool = False,
    allow_meta_image_fallback: bool = False,
) -> pd.DataFrame:
    """
    Attach `text` and `assets` columns to meta from BENCHMARKS samples.

    Behavior knobs:
    - fallback_on_zero_match: if BENCHMARKS exists but no sample_id matches, optionally fallback to meta columns
    - allow_meta_text_fallback: allow fallback to meta['question'|'prompt'|'text'] when needed
    - allow_meta_image_fallback: allow building assets from meta['image'] if missing
    """
    meta = meta.copy()

    if not benchmarks_dir.exists():
        if verbose > 0:
            print(f"  ⚠️ BENCHMARKS directory does not exist: {benchmarks_dir}")
        if allow_meta_text_fallback:
            if "question" in meta.columns:
                meta["text"] = meta["question"]
                if verbose > 0:
                    print("  ✓ Loaded text from meta['question']")
            elif "text" not in meta.columns and "prompt" in meta.columns:
                meta["text"] = meta["prompt"]
                if verbose > 0:
                    print("  ✓ Loaded text from meta['prompt']")
            elif "text" not in meta.columns:
                raise ValueError("Unable to find text data")
        else:
            # keep old behavior of cosinecls/routerdc/zooter loaders
            meta["text"] = meta.get("question", "")

        if "assets" not in meta.columns:
            if allow_meta_image_fallback and "image" in meta.columns:
                meta["assets"] = meta["image"].apply(lambda x: [{"type": "image", "path": x}] if x else [])
                if verbose > 0:
                    print("  ✓ Built assets from meta['image']")
            else:
                meta["assets"] = [[] for _ in range(len(meta))]
        return meta

    if verbose > 0:
        print("  Loading text and images from BENCHMARKS...")

    samples = load_samples_from_benchmarks(benchmarks_dir)
    dataset_dir = benchmarks_dir.parent
    if verbose > 0:
        print(f"    Total BENCHMARKS samples: {len(samples)}")

    # Optional debug info (kept for VLC's previous behavior)
    if verbose > 1:
        if samples:
            first_sample = samples[0]
            print(f"    First sample fields: {list(first_sample.keys())}")
        else:
            print("    ⚠️ No samples found under BENCHMARKS!")

    sample_map: Dict[Any, dict] = {}
    for s in samples:
        sid = s.get("sample_id") or s.get("id")
        if sid is not None:
            sample_map[sid] = s

    if verbose > 0:
        print(f"    sample_map size: {len(sample_map)}")

    texts: List[str] = []
    assets_list: List[list] = []
    matched_count = 0

    for _, row in meta.iterrows():
        sample_id = row["sample_id"]
        if sample_id in sample_map:
            matched_count += 1
            sample = sample_map[sample_id]
            text = sample.get("prompt", sample.get("text", sample.get("question", "")))
            texts.append(text)
            assets_list.append(resolve_asset_paths(sample.get("assets", []), dataset_dir))
        else:
            texts.append("")
            assets_list.append([])

    meta["text"] = texts
    meta["assets"] = assets_list

    if verbose > 0:
        print(f"    Matched samples: {matched_count}/{len(meta)}")
        if allow_meta_text_fallback or allow_meta_image_fallback:
            print(f"  ✓ Loaded {len([t for t in texts if t])} texts")
            print(f"  ✓ Loaded {len([a for a in assets_list if a])} images")

    # VLC-only fallback when BENCHMARKS exists but matches nothing
    if fallback_on_zero_match and matched_count == 0:
        if verbose > 0:
            print("  ⚠️  No BENCHMARKS matches; trying to read directly from meta...")
        if allow_meta_text_fallback:
            if "question" in meta.columns:
                meta["text"] = meta["question"]
                if verbose > 0:
                    print("    ✓ Loaded text from meta['question']")
            elif "text" in meta.columns:
                if verbose > 0:
                    print("    ✓ meta already contains 'text' field")
            elif "prompt" in meta.columns:
                meta["text"] = meta["prompt"]
                if verbose > 0:
                    print("    ✓ Loaded text from meta['prompt']")
            else:
                raise ValueError("meta is missing text fields (text/question/prompt)")

        if "assets" not in meta.columns:
            if allow_meta_image_fallback and "image" in meta.columns:
                meta["assets"] = meta["image"].apply(lambda x: [{"type": "image", "path": x}] if x else [])
                if verbose > 0:
                    print("    ✓ Built assets from meta['image']")
            else:
                if verbose > 0:
                    print("    ⚠️ No image data in meta; using empty images")
                meta["assets"] = [[] for _ in range(len(meta))]

    return meta


def load_splits_jsonl(dataset_dir: Path, *, verbose: int = 1) -> Dict[str, List[Any]]:
    splits: Dict[str, List[Any]] = {}
    splits_dir = Path(dataset_dir) / "SPLITS"
    for split_name in ["train", "dev", "test"]:
        split_file = splits_dir / f"{split_name}.jsonl"
        if split_file.exists():
            with open(split_file, "r", encoding="utf-8") as f:
                splits[split_name] = [json.loads(line)["sample_id"] for line in f]
            if verbose > 0:
                print(f"  {split_name}: {len(splits[split_name])} samples")
    return splits


def load_splits_with_legacy_fallback(dataset_dir: Path, meta: pd.DataFrame, *, verbose: int = 1) -> Dict[str, List[Any]]:
    """VLC-compatible splits loader: JSONL under SPLITS/, then fallback to legacy splits.json, else all-train."""
    splits_dir = Path(dataset_dir) / "SPLITS"
    splits: Dict[str, List[Any]] = {}

    if splits_dir.exists():
        if verbose > 0:
            print("  Loading splits from SPLITS directory...")
        for split_name in ["train", "dev", "test"]:
            split_file = splits_dir / f"{split_name}.jsonl"
            if split_file.exists():
                sample_ids: List[Any] = []
                with open(split_file, "r", encoding="utf-8") as f:
                    for line in f:
                        sample_ids.append(json.loads(line)["sample_id"])
                splits[split_name] = sample_ids
        if verbose > 0:
            print(
                f"  Splits: train={len(splits.get('train', []))}, "
                f"dev={len(splits.get('dev', []))}, "
                f"test={len(splits.get('test', []))}"
            )
        return splits

    # Legacy fallback
    data_dir = Path(dataset_dir) / "data"
    splits_file = data_dir / "registry/splits.json"
    if not splits_file.exists():
        splits_file = data_dir / "splits.json"

    if splits_file.exists():
        with open(splits_file, "r", encoding="utf-8") as f:
            splits = json.load(f)
        if verbose > 0:
            print(
                f"  Splits: train={len(splits.get('train', []))}, "
                f"dev={len(splits.get('dev', []))}, "
                f"test={len(splits.get('test', []))}"
            )
        return splits

    # Last fallback
    all_ids = meta["sample_id"].tolist()
    splits = {"train": all_ids, "dev": [], "test": []}
    if verbose > 0:
        print("  ⚠️  Warning: split files not found; using all data as training set")
    return splits


def load_models_list_txt(dataset_dir: Path, num_models: int, *, verbose: int = 1) -> List[str]:
    """Load models from data/registry/models.txt, fallback to model_{i}."""
    data_dir = Path(dataset_dir) / "data"
    models_file = data_dir / "registry/models.txt"
    if models_file.exists():
        with open(models_file, "r", encoding="utf-8") as f:
            models = [line.strip() for line in f if line.strip()]
    else:
        models = [f"model_{i}" for i in range(num_models)]
    if verbose > 0:
        print(f"  Num models: {len(models)}")
    return models


def load_model_mapping_json(dataset_dir: Path, num_models: int, *, verbose: int = 1) -> Dict[int, str]:
    """VLC-style: load model mapping from data/registry/models.json, fallback to model_{i} mapping."""
    data_dir = Path(dataset_dir) / "data"
    models_file = data_dir / "registry/models.json"
    if models_file.exists():
        with open(models_file, "r", encoding="utf-8") as f:
            models_data = json.load(f)
        model_mapping = {i: m["name"] for i, m in enumerate(models_data["models"])}
    else:
        model_mapping = {i: f"model_{i}" for i in range(num_models)}
    if verbose > 0:
        print(f"  Num models: {len(model_mapping)}")
    return model_mapping


def load_multimodal_data_for_end_to_end_router(
    dataset_dir: Union[str, Path],
    *,
    verbose: int = 1,
    mode: Literal["simple", "vlc"] = "simple",
) -> Dict[str, Any]:
    """
    Unified loader for end-to-end routers.

    - mode="simple": cosinecls/routerdc/zooter behavior; returns {Y,C,meta,models,splits}
    - mode="vlc": VLC behavior; returns {Y,C,meta,splits,model_mapping}
    """
    dataset_dir = Path(dataset_dir)
    data_dir = dataset_dir / "data"

    if mode == "simple":
        if verbose > 0:
            print("📥 Loading training data...")
        Y, C = load_quality_and_cost_matrices(dataset_dir)
        if verbose > 0:
            print(f"  Quality matrix: {Y.shape}")
            print(f"  Cost matrix: {C.shape}")
        meta_file = data_dir / "registry/meta.parquet"
        if not meta_file.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_file}")
        meta = pd.read_parquet(meta_file)
        if verbose > 0:
            print(f"  Meta rows: {len(meta)}")

        meta = attach_text_assets_from_benchmarks(
            meta,
            dataset_dir / "BENCHMARKS",
            verbose=verbose,
            fallback_on_zero_match=False,
            allow_meta_text_fallback=False,
            allow_meta_image_fallback=False,
        )
        models = load_models_list_txt(dataset_dir, num_models=Y.shape[1], verbose=verbose)
        splits = load_splits_jsonl(dataset_dir, verbose=verbose)
        return {"Y": Y, "C": C, "meta": meta, "models": models, "splits": splits}

    if mode == "vlc":
        if verbose > 0:
            print("📥 Loading VLC training data...")
        Y, C = load_quality_and_cost_matrices(dataset_dir)
        if verbose > 0:
            print(f"  Quality matrix: {Y.shape}")
            print(f"  Cost matrix: {C.shape}")
        meta_file = data_dir / "registry/meta.parquet"
        if not meta_file.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_file}")
        meta = pd.read_parquet(meta_file)
        if verbose > 0:
            print(f"  Meta rows: {len(meta)}")

        meta = attach_text_assets_from_benchmarks(
            meta,
            dataset_dir / "BENCHMARKS",
            verbose=verbose,
            fallback_on_zero_match=True,
            allow_meta_text_fallback=True,
            allow_meta_image_fallback=True,
        )
        splits = load_splits_with_legacy_fallback(dataset_dir, meta, verbose=verbose)
        model_mapping = load_model_mapping_json(dataset_dir, num_models=Y.shape[1], verbose=verbose)
        return {"Y": Y, "C": C, "meta": meta, "splits": splits, "model_mapping": model_mapping}

    raise ValueError(f"Unknown mode: {mode}")


# Backward-compatible alias (old name used in earlier refactor)
def load_multimodal_data_for_deep_router(*args, **kwargs):
    return load_multimodal_data_for_end_to_end_router(*args, **kwargs)

