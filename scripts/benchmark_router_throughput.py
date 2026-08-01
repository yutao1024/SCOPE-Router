#!/usr/bin/env python3
"""Benchmark inference throughput for trained VL-RouterBench routers.

The script measures router decision throughput, not downstream VLM serving time.
For feature-based routers it uses pre-extracted embeddings. For VLM-backbone
routers such as RouterDC/Zooter/VLC/CosineCLS it calls their meta-based predict
path, so image/text preprocessing and backbone inference are included.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for CPU-only baselines
    torch = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routers.baselines.cheapest_global import CheapestGlobal
from routers.baselines.oracle import Oracle
from routers.baselines.random_router import RandomRouter
from routers.baselines.strongest_global import StrongestGlobal
from routers.baselines.strongest_per_dataset import StrongestPerDataset
from routers.features.text_encoder import TextEncoder
from routers.features.vision_encoder import VisionEncoder
from routers.utils.benchmarks_data import attach_text_assets_from_benchmarks
from routers.utils.latency_profiler import estimate_tokens_from_meta
from routers.utils.train_utils import load_data_for_training


LOADERS: Dict[str, Tuple[str, str]] = {
    "mlp": ("routers.mlp.router", "MLPRouter"),
    "linear": ("routers.linear.router", "LinearRouter"),
    "ovr": ("routers.ovr.router", "OVRRouter"),
    "kmeans": ("routers.kmeans.router", "KMeansRouter"),
    "knn": ("routers.knn.router", "KNNRouter"),
    "prknn": ("routers.prknn.router", "PRKNNRouter"),
    "scope_router": ("routers.scope_router.router", "ScopeRouter"),
    "scope_router_online": ("routers.scope_router_online.router", "OnlineScopeRouter"),
    "routerdc": ("routers.routerdc.router", "RouterDCRouter"),
    "zooter": ("routers.zooter.router", "ZooterRouter"),
    "vlc": ("routers.vlc.router", "VLCRouter"),
    "cosinecls": ("routers.cosinecls.router", "CosineCLSRRouter"),
}

BASELINE_NAMES = {
    "strongest_global",
    "strongest_per_dataset",
    "cheapest_global",
    "oracle",
    "random",
}

FEATURE_ROUTER_METHODS = {
    "mlp",
    "linear",
    "ovr",
    "kmeans",
    "knn",
    "prknn",
    "scope_router",
}

END_TO_END_ROUTER_METHODS = {
    "routerdc",
    "zooter",
    "vlc",
    "cosinecls",
    "scope_router_online",
}


@dataclass
class RouterSpec:
    name: str
    method: str
    path: Optional[Path] = None


def parse_batch_sizes(value: str) -> List[int]:
    sizes = [int(x) for x in value.replace(",", " ").split() if x.strip()]
    if not sizes:
        raise ValueError("At least one batch size is required")
    return sizes


def parse_router_spec(spec: str) -> RouterSpec:
    """Parse name=method:path, method:path, method=baseline, or baseline."""
    name: Optional[str] = None
    body = spec
    if "=" in spec:
        name, body = spec.split("=", 1)

    if ":" in body:
        method, path_s = body.split(":", 1)
        path = Path(path_s)
    else:
        method = body
        path = None

    method = method.strip().lower().replace("-", "_")
    if not method:
        raise ValueError(f"Invalid router spec: {spec}")
    if name is None or not name.strip():
        name = method if path is None else f"{method}:{path.parent.name or path.stem}"
    return RouterSpec(name=name.strip(), method=method, path=path)


def infer_method_from_path(path: Path) -> Optional[str]:
    text = str(path).lower().replace("-", "_")
    candidates = [
        "scope_router_online",
        "scope_router",
        "cosinecls",
        "routerdc",
        "zooter",
        "linear",
        "kmeans",
        "prknn",
        "knn",
        "mlp",
        "vlc",
        "ovr",
    ]
    for method in candidates:
        if method in text:
            return method
    return None


def scan_router_paths(scan_dirs: Sequence[Path]) -> List[RouterSpec]:
    specs: List[RouterSpec] = []
    seen: set[Path] = set()
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            print(f"[warn] scan dir not found: {scan_dir}")
            continue
        for path in sorted(scan_dir.rglob("*.pkl")):
            if path.name.endswith("_model.pkl"):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            method = infer_method_from_path(path)
            if method is None:
                print(f"[warn] cannot infer method, skipped: {path}")
                continue
            seen.add(resolved)
            rel = path
            try:
                rel = path.relative_to(ROOT)
            except ValueError:
                pass
            name = f"{method}:{path.parent.name}/{path.stem}"
            specs.append(RouterSpec(name=name, method=method, path=rel))
    return specs


def import_loader(method: str):
    if method not in LOADERS:
        raise ValueError(f"Unknown method: {method}")
    module_name, class_name = LOADERS[method]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_trained_router(spec: RouterSpec, device: str):
    if spec.path is None:
        return build_baseline(spec.method)

    path = spec.path
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Router file not found: {path}")

    cls = import_loader(spec.method)
    try:
        return cls.load(str(path), device=device)
    except TypeError:
        return cls.load(str(path))


def build_baseline(method: str):
    if method == "strongest_global":
        return StrongestGlobal()
    if method == "strongest_per_dataset":
        return StrongestPerDataset()
    if method == "cheapest_global":
        return CheapestGlobal()
    if method == "oracle":
        return Oracle()
    if method == "random":
        return RandomRouter(seed=45)
    raise ValueError(f"Unknown baseline: {method}")


def aligned_split_indices(data: Dict[str, Any], split: str, limit: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    meta = data["meta"]
    sample_ids = np.asarray(data["sample_ids"])
    split_ids = data["splits"].get(split) or meta["sample_id"].tolist()
    split_set = set(split_ids)

    meta_ids = meta["sample_id"].to_numpy()
    meta_idx = np.array([i for i, sid in enumerate(meta_ids) if sid in split_set], dtype=int)
    emb_id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
    keep_meta: List[int] = []
    keep_emb: List[int] = []
    for mi in meta_idx:
        sid = meta_ids[mi]
        if sid in emb_id_to_idx:
            keep_meta.append(int(mi))
            keep_emb.append(int(emb_id_to_idx[sid]))

    if limit is not None and limit > 0:
        keep_meta = keep_meta[:limit]
        keep_emb = keep_emb[:limit]

    if not keep_meta:
        raise ValueError(f"No samples available for split={split}")
    return np.asarray(keep_meta, dtype=int), np.asarray(keep_emb, dtype=int)


def get_inputs(data: Dict[str, Any], meta_idx: np.ndarray, emb_idx: np.ndarray) -> Dict[str, Any]:
    meta = data["meta"].iloc[meta_idx].copy()
    if data.get("query_embeddings") is not None:
        return {
            "meta": meta,
            "X": data["query_embeddings"][emb_idx].astype(np.float32),
            "X_text": None,
            "X_vision": None,
            "Y": data["Y"][meta_idx],
            "C": data["C"][meta_idx],
        }
    return {
        "meta": meta,
        "X": None,
        "X_text": data["text_embeddings"][emb_idx].astype(np.float32),
        "X_vision": data["vision_embeddings"][emb_idx].astype(np.float32),
        "Y": data["Y"][meta_idx],
        "C": data["C"][meta_idx],
    }


def prepare_online_inputs(inputs: Dict[str, Any], dataset_dir: Path) -> Dict[str, Any]:
    meta = attach_text_assets_from_benchmarks(
        inputs["meta"],
        dataset_dir / "BENCHMARKS",
        verbose=0,
        fallback_on_zero_match=True,
        allow_meta_text_fallback=True,
        allow_meta_image_fallback=True,
    )
    out = dict(inputs)
    out["meta"] = meta
    out["online_texts"] = [
        str(row.get("text", row.get("prompt", row.get("question", ""))))
        for _, row in meta.iterrows()
    ]
    out["online_samples"] = [
        {
            "sample_id": row.get("sample_id"),
            "prompt": row.get("text", row.get("prompt", row.get("question", ""))),
            "assets": row.get("assets", []),
        }
        for _, row in meta.iterrows()
    ]
    return out


def resolve_tsv_images(df: pd.DataFrame) -> pd.DataFrame:
    if "image" not in df.columns or "index" not in df.columns:
        return df
    df = df.copy()
    df["image"] = df["image"].astype(str)
    image_map = {str(idx): img for idx, img in zip(df["index"], df["image"])}
    for key in list(image_map.keys()):
        if len(image_map[key]) <= 64:
            ref_idx = image_map[key]
            if ref_idx in image_map and len(image_map[ref_idx]) > 64:
                image_map[key] = image_map[ref_idx]
    for idx_val, img_data in image_map.items():
        mask = df["index"].astype(str) == idx_val
        df.loc[mask, "image"] = img_data
    return df


def preload_image_groups(samples: List[dict]) -> List[List[Any]]:
    from collections import defaultdict
    import base64
    from io import BytesIO

    from PIL import Image

    image_groups: List[List[Any]] = [[] for _ in samples]
    tsv_assets: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    path_assets: List[Tuple[int, Any]] = []

    for sample_idx, sample in enumerate(samples):
        for asset in sample.get("assets", []):
            if isinstance(asset, dict) and asset.get("type") == "image_tsv":
                tsv_assets[str(asset["tsv_file"])].append((sample_idx, int(asset["index"])))
            elif isinstance(asset, dict) and asset.get("path"):
                path_assets.append((sample_idx, asset["path"]))
            elif isinstance(asset, dict) and asset.get("uri"):
                path_assets.append((sample_idx, asset["uri"]))
            elif isinstance(asset, (str, Path)):
                path_assets.append((sample_idx, asset))

    for tsv_file, entries in tsv_assets.items():
        df = resolve_tsv_images(pd.read_csv(tsv_file, sep="\t"))
        for sample_idx, row_idx in entries:
            if row_idx >= len(df):
                continue
            try:
                img_data = base64.b64decode(df.iloc[row_idx]["image"])
                image_groups[sample_idx].append(Image.open(BytesIO(img_data)).convert("RGB"))
            except Exception:
                continue

    for sample_idx, image_path in path_assets:
        try:
            image_groups[sample_idx].append(Image.open(image_path).convert("RGB"))
        except Exception:
            continue

    return image_groups


def extract_vision_from_image_groups(
    vision_encoder: VisionEncoder,
    image_groups: List[List[Any]],
    batch_size: int,
    pooling: str = "mean",
) -> np.ndarray:
    flat_images: List[Any] = []
    image_to_sample: List[int] = []
    for sample_idx, images in enumerate(image_groups):
        for image in images:
            image_to_sample.append(sample_idx)
            flat_images.append(image)

    if flat_images:
        flat_features = vision_encoder._extract_batch_from_images(flat_images, batch_size)
    else:
        flat_features = np.zeros((0, vision_encoder.dimension), dtype=np.float32)

    per_sample_features: List[List[np.ndarray]] = [[] for _ in image_groups]
    for feature, sample_idx in zip(flat_features, image_to_sample):
        per_sample_features[sample_idx].append(np.asarray(feature, dtype=np.float32))

    embeddings: List[np.ndarray] = []
    for features_list in per_sample_features:
        if not features_list:
            embedding = np.zeros(vision_encoder.dimension, dtype=np.float32)
        elif len(features_list) == 1:
            embedding = features_list[0]
        else:
            features = np.stack(features_list).astype(np.float32)
            if pooling == "max":
                embedding = np.max(features, axis=0)
            elif pooling == "first":
                embedding = features[0]
            else:
                embedding = np.mean(features, axis=0)
            if vision_encoder.normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
        embeddings.append(embedding)
    return np.asarray(embeddings, dtype=np.float32)


def get_train_for_baseline(data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    train_ids = data["splits"].get("train")
    if not train_ids:
        return data["Y"], data["C"], data["meta"]
    train_set = set(train_ids)
    meta_ids = data["meta"]["sample_id"].to_numpy()
    idx = np.array([i for i, sid in enumerate(meta_ids) if sid in train_set], dtype=int)
    if len(idx) == 0:
        return data["Y"], data["C"], data["meta"]
    return data["Y"][idx], data["C"][idx], data["meta"].iloc[idx].copy()


def sync_cuda(device: str) -> None:
    if device == "cuda" and torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def call_predict(router, inputs: Dict[str, Any], batch_size: int, method: str):
    meta = inputs["meta"]
    X = inputs["X"]
    X_text = inputs["X_text"]
    X_vision = inputs["X_vision"]

    if method == "oracle":
        return router.predict(meta=meta, Y_test=inputs["Y"], C_test=inputs["C"])

    attempts = [
        lambda: router.predict(
            X=X,
            X_text=X_text,
            X_vision=X_vision,
            meta=meta,
            batch_size=batch_size,
            show_progress=False,
        ),
        lambda: router.predict(X=X, X_text=X_text, X_vision=X_vision, meta=meta),
        lambda: router.predict(X=X, meta=meta),
        lambda: router.predict(meta=meta, batch_size=batch_size, show_progress=False),
        lambda: router.predict(meta=meta),
        lambda: router.predict(X=X),
    ]
    last_error: Optional[Exception] = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_error = exc
            continue
    raise RuntimeError(f"Could not call predict for {type(router).__name__}") from last_error


def call_predict_online_feature_router(
    router,
    inputs: Dict[str, Any],
    batch_size: int,
    method: str,
    text_encoder: TextEncoder,
    vision_encoder: VisionEncoder,
    image_groups: List[List[Any]],
):
    texts = inputs["online_texts"]
    text_emb = text_encoder.extract(texts).astype(np.float32)
    vision_emb = extract_vision_from_image_groups(
        vision_encoder,
        image_groups,
        batch_size=batch_size,
    ).astype(np.float32)
    online_inputs = dict(inputs)
    online_inputs["X"] = None
    online_inputs["X_text"] = text_emb
    online_inputs["X_vision"] = vision_emb
    return call_predict(router, online_inputs, batch_size=batch_size, method=method)


def slice_inputs(inputs: Dict[str, Any], n: int) -> Dict[str, Any]:
    sliced = {
        "meta": inputs["meta"].iloc[:n].copy(),
        "X": inputs["X"][:n] if inputs["X"] is not None else None,
        "X_text": inputs["X_text"][:n] if inputs["X_text"] is not None else None,
        "X_vision": inputs["X_vision"][:n] if inputs["X_vision"] is not None else None,
        "Y": inputs["Y"][:n],
        "C": inputs["C"][:n],
    }
    if "online_texts" in inputs:
        sliced["online_texts"] = inputs["online_texts"][:n]
    if "online_samples" in inputs:
        sliced["online_samples"] = inputs["online_samples"][:n]
    return sliced


def percentile_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def measure_feature_encoder_latency(
    inputs: Dict[str, Any],
    token_counts: np.ndarray,
    image_groups: List[List[Any]],
    batch_sizes: Sequence[int],
    warmup_runs: int,
    test_runs: int,
    device: str,
    text_encoder: TextEncoder,
    vision_encoder: VisionEncoder,
) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}
    total_available = len(inputs["meta"])
    for batch_size in batch_sizes:
        if batch_size > total_available:
            continue
        batch_inputs = slice_inputs(inputs, batch_size)
        batch_image_groups = image_groups[:batch_size]
        batch_tokens = int(token_counts[:batch_size].sum()) if len(token_counts) else -1

        def encode_once() -> Tuple[np.ndarray, np.ndarray]:
            text_emb = text_encoder.extract(batch_inputs["online_texts"]).astype(np.float32)
            vision_emb = extract_vision_from_image_groups(
                vision_encoder,
                batch_image_groups,
                batch_size=batch_size,
            ).astype(np.float32)
            return text_emb, vision_emb

        for _ in range(warmup_runs):
            _ = encode_once()
            sync_cuda(device)

        elapsed_s: List[float] = []
        last_text: Optional[np.ndarray] = None
        last_vision: Optional[np.ndarray] = None
        for _ in range(test_runs):
            sync_cuda(device)
            start = time.perf_counter()
            last_text, last_vision = encode_once()
            sync_cuda(device)
            elapsed_s.append(time.perf_counter() - start)

        stats_ms = percentile_stats([x * 1000.0 for x in elapsed_s])
        mean_s = stats_ms["mean"] / 1000.0
        results[batch_size] = {
            "stats_ms": stats_ms,
            "batch_tokens": batch_tokens,
            "samples_per_sec": float(batch_size / mean_s) if mean_s > 0 else math.inf,
            "tokens_per_sec": float(batch_tokens / mean_s) if batch_tokens > 0 and mean_s > 0 else -1.0,
            "X_text": last_text,
            "X_vision": last_vision,
        }
        print(
            f"[encoder] bs={batch_size} "
            f"{results[batch_size]['samples_per_sec']:.2f} samples/s, "
            f"{stats_ms['mean'] / batch_size:.4f} ms/sample"
        )
    return results


def combine_stats_ms(encoder_stats: Dict[str, float], router_stats: Dict[str, float]) -> Dict[str, float]:
    combined = {}
    for key in ["mean", "min", "p50", "p95", "p99", "max"]:
        combined[key] = float(encoder_stats[key] + router_stats[key])
    combined["std"] = float(math.sqrt(encoder_stats["std"] ** 2 + router_stats["std"] ** 2))
    return combined


def benchmark_one(
    name: str,
    method: str,
    router,
    inputs: Dict[str, Any],
    token_counts: np.ndarray,
    batch_sizes: Sequence[int],
    warmup_runs: int,
    test_runs: int,
    device: str,
    mode: str,
    text_encoder: Optional[TextEncoder] = None,
    vision_encoder: Optional[VisionEncoder] = None,
    all_image_groups: Optional[List[List[Any]]] = None,
    online_feature_strategy: str = "composed",
    encoder_latency_by_bs: Optional[Dict[int, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    total_available = len(inputs["meta"])
    for batch_size in batch_sizes:
        if batch_size > total_available:
            print(f"[skip] {name} batch_size={batch_size} > samples={total_available}")
            continue
        batch_inputs = slice_inputs(inputs, batch_size)
        batch_image_groups = all_image_groups[:batch_size] if all_image_groups is not None else None
        batch_tokens = int(token_counts[:batch_size].sum()) if len(token_counts) else -1
        includes_feature_encoder = bool(
            mode == "online"
            and method in FEATURE_ROUTER_METHODS
            and text_encoder is not None
            and vision_encoder is not None
            and batch_image_groups is not None
        )
        composed_feature_encoder = bool(
            includes_feature_encoder
            and online_feature_strategy == "composed"
            and encoder_latency_by_bs is not None
            and batch_size in encoder_latency_by_bs
        )

        if composed_feature_encoder:
            enc = encoder_latency_by_bs[batch_size]
            batch_inputs["X"] = None
            batch_inputs["X_text"] = enc["X_text"]
            batch_inputs["X_vision"] = enc["X_vision"]

        for _ in range(warmup_runs):
            if includes_feature_encoder and not composed_feature_encoder:
                _ = call_predict_online_feature_router(
                    router,
                    batch_inputs,
                    batch_size=batch_size,
                    method=method,
                    text_encoder=text_encoder,
                    vision_encoder=vision_encoder,
                    image_groups=batch_image_groups,
                )
            else:
                _ = call_predict(router, batch_inputs, batch_size=batch_size, method=method)
            sync_cuda(device)

        elapsed_s: List[float] = []
        for _ in range(test_runs):
            sync_cuda(device)
            start = time.perf_counter()
            if includes_feature_encoder and not composed_feature_encoder:
                _ = call_predict_online_feature_router(
                    router,
                    batch_inputs,
                    batch_size=batch_size,
                    method=method,
                    text_encoder=text_encoder,
                    vision_encoder=vision_encoder,
                    image_groups=batch_image_groups,
                )
            else:
                _ = call_predict(router, batch_inputs, batch_size=batch_size, method=method)
            sync_cuda(device)
            elapsed_s.append(time.perf_counter() - start)

        router_stats_ms = percentile_stats([x * 1000.0 for x in elapsed_s])
        stats_ms = router_stats_ms
        encoder_stats_ms = None
        if composed_feature_encoder:
            encoder_stats_ms = encoder_latency_by_bs[batch_size]["stats_ms"]
            stats_ms = combine_stats_ms(encoder_stats_ms, router_stats_ms)
        mean_s = stats_ms["mean"] / 1000.0
        samples_per_sec = float(batch_size / mean_s) if mean_s > 0 else math.inf
        tokens_per_sec = float(batch_tokens / mean_s) if batch_tokens > 0 and mean_s > 0 else -1.0
        ms_per_sample = float(stats_ms["mean"] / batch_size)
        ms_per_token = float(stats_ms["mean"] / batch_tokens) if batch_tokens > 0 else -1.0

        row = {
            "router": name,
            "method": method,
            "router_type": type(router).__name__,
            "device": device,
            "batch_size": int(batch_size),
            "warmup_runs": int(warmup_runs),
            "test_runs": int(test_runs),
            "num_profile_samples": int(total_available),
            "batch_tokens": int(batch_tokens),
            "latency_ms_mean": stats_ms["mean"],
            "latency_ms_std": stats_ms["std"],
            "latency_ms_min": stats_ms["min"],
            "latency_ms_p50": stats_ms["p50"],
            "latency_ms_p95": stats_ms["p95"],
            "latency_ms_p99": stats_ms["p99"],
            "latency_ms_max": stats_ms["max"],
            "ms_per_sample": ms_per_sample,
            "samples_per_sec": samples_per_sec,
            "ms_per_token": ms_per_token,
            "tokens_per_sec": tokens_per_sec,
            "mode": mode,
            "includes_feature_encoder": includes_feature_encoder,
            "online_feature_strategy": online_feature_strategy if includes_feature_encoder else "",
            "encoder_latency_ms_mean": encoder_stats_ms["mean"] if encoder_stats_ms else (-1 if includes_feature_encoder else 0),
            "router_latency_ms_mean": router_stats_ms["mean"],
        }
        rows.append(row)
        print(
            f"[result] {name} bs={batch_size} "
            f"{samples_per_sec:.2f} samples/s, {ms_per_sample:.4f} ms/sample"
        )
    return rows


def write_outputs(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "detailed_results.json"
    if rows:
        preferred = [
            "router", "method", "router_type", "device", "batch_size",
            "warmup_runs", "test_runs", "num_profile_samples", "batch_tokens",
            "latency_ms_mean", "latency_ms_std", "latency_ms_min", "latency_ms_p50",
            "latency_ms_p95", "latency_ms_p99", "latency_ms_max", "ms_per_sample",
            "samples_per_sec", "ms_per_token", "tokens_per_sec", "mode",
            "includes_feature_encoder", "online_feature_strategy",
            "encoder_latency_ms_mean", "router_latency_ms_mean", "path", "split",
            "text_encoder", "vision_encoder", "error",
        ]
        all_keys = set().union(*(row.keys() for row in rows))
        fieldnames = [key for key in preferred if key in all_keys]
        fieldnames.extend(sorted(all_keys - set(fieldnames)))
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"[save] {csv_path}")
    print(f"[save] {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark throughput for trained routers")
    parser.add_argument("--dataset_dir", default=".", help="VL-RouterBench dataset dir")
    parser.add_argument("--router", action="append", default=[],
                        help="Router spec: name=method:path, method:path, baseline, or name=baseline")
    parser.add_argument("--router-list", type=str, default=None,
                        help="Text file with one router spec per line")
    parser.add_argument("--scan-dir", action="append", default=[],
                        help="Directory to scan recursively for trained .pkl routers")
    parser.add_argument("--include-baselines", action="store_true",
                        help="Also benchmark strongest/cheapest/random/oracle baselines")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"],
                        help="Split used for profiling samples")
    parser.add_argument("--limit-samples", type=int, default=1024,
                        help="Max samples to keep available for profiling; first N of split")
    parser.add_argument("--batch-sizes", default="1,8,16,32,64",
                        help="Comma or space separated batch sizes")
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--test-runs", type=int, default=50)
    parser.add_argument("--text-encoder", default="BAAI/bge-m3")
    parser.add_argument("--vision-encoder", default="facebook/dinov2-large")
    parser.add_argument("--query-embedding-file", default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--mode", default="offline", choices=["offline", "online"],
                        help="offline: precomputed embedding decision throughput; online: include feature encoder inference for feature-based routers")
    parser.add_argument("--online-feature-strategy", default="composed", choices=["composed", "inline"],
                        help="For feature routers in online mode: composed measures shared encoders once and adds router latency; inline measures encoder+router inside every router run")
    parser.add_argument("--output-dir", default="outputs/router_throughput")
    args = parser.parse_args()

    if os.environ.get("VLM_ROUTER_SKIP_LATENCY"):
        print("[warn] VLM_ROUTER_SKIP_LATENCY is set; ignored by this standalone benchmark.")

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    batch_sizes = parse_batch_sizes(args.batch_sizes)

    specs = [parse_router_spec(x) for x in args.router]
    if args.router_list:
        with open(args.router_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    specs.append(parse_router_spec(line))
    specs.extend(scan_router_paths([Path(x) for x in args.scan_dir]))

    if args.include_baselines:
        for name in ["strongest_global", "strongest_per_dataset", "cheapest_global", "random", "oracle"]:
            specs.append(RouterSpec(name=name, method=name, path=None))

    if not specs:
        raise SystemExit("No routers to benchmark. Use --router, --router-list, --scan-dir, or --include-baselines.")

    method_counts = Counter(spec.method for spec in specs)
    print("[routers] discovered:")
    for method in sorted(method_counts):
        print(f"  {method}: {method_counts[method]}")

    print("[load] data")
    data = load_data_for_training(
        dataset_dir,
        text_encoder=args.text_encoder,
        vision_encoder=args.vision_encoder,
        query_embedding_file=args.query_embedding_file,
    )
    meta_idx, emb_idx = aligned_split_indices(data, args.split, args.limit_samples)
    inputs = get_inputs(data, meta_idx, emb_idx)
    if args.mode == "online":
        inputs = prepare_online_inputs(inputs, dataset_dir)
    token_counts = estimate_tokens_from_meta(inputs["meta"])
    print(f"[data] split={args.split}, samples={len(inputs['meta'])}, avg_tokens={float(token_counts.mean()):.1f}")

    online_text_encoder: Optional[TextEncoder] = None
    online_vision_encoder: Optional[VisionEncoder] = None
    online_image_groups: Optional[List[List[Any]]] = None
    encoder_latency_by_bs: Optional[Dict[int, Dict[str, Any]]] = None
    if args.mode == "online":
        needs_feature_encoder = any(spec.method in FEATURE_ROUTER_METHODS for spec in specs)
        if needs_feature_encoder:
            print("[online] loading shared feature encoders")
            online_text_encoder = TextEncoder(
                model_name=args.text_encoder,
                device=args.device,
                batch_size=max(batch_sizes),
                show_progress=False,
            )
            online_vision_encoder = VisionEncoder(
                model_name=args.vision_encoder,
                device=args.device,
                batch_size=max(batch_sizes),
            )
            print("[online] preloading/decompressing images once; timed runs measure encoder inference + router")
            online_image_groups = preload_image_groups(inputs["online_samples"])
            num_images = sum(len(images) for images in online_image_groups)
            print(f"[online] loaded images: {num_images} across {sum(bool(x) for x in online_image_groups)}/{len(online_image_groups)} samples")
            if args.online_feature_strategy == "composed":
                print("[online] measuring shared feature encoder latency once")
                encoder_latency_by_bs = measure_feature_encoder_latency(
                    inputs=inputs,
                    token_counts=token_counts,
                    image_groups=online_image_groups,
                    batch_sizes=batch_sizes,
                    warmup_runs=args.warmup_runs,
                    test_runs=args.test_runs,
                    device=args.device,
                    text_encoder=online_text_encoder,
                    vision_encoder=online_vision_encoder,
                )

    train_Y, train_C, train_meta = get_train_for_baseline(data)
    all_rows: List[Dict[str, Any]] = []
    for spec in specs:
        print(f"\n[router] {spec.name} method={spec.method} path={spec.path or '<baseline>'}")
        try:
            router = load_trained_router(spec, device=args.device)
            if spec.method in BASELINE_NAMES:
                router.fit(train_Y, train_C, train_meta)
            rows = benchmark_one(
                name=spec.name,
                method=spec.method,
                router=router,
                inputs=inputs,
                token_counts=token_counts,
                batch_sizes=batch_sizes,
                warmup_runs=args.warmup_runs,
                test_runs=args.test_runs,
                device=args.device,
                mode=args.mode,
                text_encoder=online_text_encoder,
                vision_encoder=online_vision_encoder,
                all_image_groups=online_image_groups,
                online_feature_strategy=args.online_feature_strategy,
                encoder_latency_by_bs=encoder_latency_by_bs,
            )
            for row in rows:
                row["path"] = str(spec.path or "")
                row["split"] = args.split
                row["text_encoder"] = args.text_encoder
                row["vision_encoder"] = args.vision_encoder
                row["mode"] = args.mode
            all_rows.extend(rows)
            write_outputs(all_rows, output_dir)
        except Exception as exc:
            print(f"[error] {spec.name}: {exc}")
            all_rows.append({
                "router": spec.name,
                "method": spec.method,
                "router_type": "",
                "device": args.device,
                "batch_size": -1,
                "warmup_runs": args.warmup_runs,
                "test_runs": args.test_runs,
                "num_profile_samples": len(inputs["meta"]),
                "batch_tokens": -1,
                "latency_ms_mean": -1,
                "latency_ms_std": -1,
                "latency_ms_min": -1,
                "latency_ms_p50": -1,
                "latency_ms_p95": -1,
                "latency_ms_p99": -1,
                "latency_ms_max": -1,
                "ms_per_sample": -1,
                "samples_per_sec": -1,
                "ms_per_token": -1,
                "tokens_per_sec": -1,
                "mode": args.mode,
                "includes_feature_encoder": False,
                "online_feature_strategy": "",
                "encoder_latency_ms_mean": -1,
                "router_latency_ms_mean": -1,
                "path": str(spec.path or ""),
                "split": args.split,
                "text_encoder": args.text_encoder,
                "vision_encoder": args.vision_encoder,
                "error": str(exc),
            })
            write_outputs(all_rows, output_dir)

    print("\n[done]")
    write_outputs(all_rows, output_dir)


if __name__ == "__main__":
    main()
