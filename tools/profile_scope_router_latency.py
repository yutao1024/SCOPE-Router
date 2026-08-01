#!/usr/bin/env python3
"""Profile online latency for SCOPE-Router with text/vision feature extraction."""

import argparse
import base64
import json
import random
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from routers.features.text_encoder import TextEncoder
from routers.features.vision_encoder import VisionEncoder
from routers.scope_router.router import ScopeRouter
from routers.utils.benchmarks_data import load_samples_from_benchmarks, load_splits_jsonl


def encoder_to_filename(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def timed(label: str, fn: Callable[[], object]) -> tuple[object, float]:
    sync_cuda()
    start = time.perf_counter()
    result = fn()
    sync_cuda()
    elapsed = time.perf_counter() - start
    print(f"  {label}: {elapsed:.4f}s")
    return result, elapsed


def select_samples(dataset_dir: Path, split: str, num_samples: int, seed: int) -> List[dict]:
    samples = load_samples_from_benchmarks(dataset_dir / "BENCHMARKS")
    sample_by_id = {sample["sample_id"]: sample for sample in samples if "sample_id" in sample}
    splits = load_splits_jsonl(dataset_dir, verbose=0)
    split_ids = [sid for sid in splits.get(split, []) if sid in sample_by_id]
    if not split_ids:
        raise ValueError(f"No BENCHMARKS samples found for split={split}")

    rng = random.Random(seed)
    if num_samples > 0 and num_samples < len(split_ids):
        split_ids = rng.sample(split_ids, num_samples)
    return [sample_by_id[sid] for sid in split_ids]


def resolve_tsv_images(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve VLMEvalKit-style short image references inside a TSV dataframe."""
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


def preload_image_groups(samples: List[dict], asset_key: str = "assets") -> List[List[Image.Image]]:
    """Load selected sample images into memory once, outside timed encoder runs."""
    from collections import defaultdict

    image_groups: List[List[Image.Image]] = [[] for _ in samples]
    tsv_assets = defaultdict(list)
    path_assets = []

    for sample_idx, sample in enumerate(samples):
        for asset in sample.get(asset_key, []):
            if isinstance(asset, dict) and asset.get("type") == "image_tsv":
                tsv_assets[asset["tsv_file"]].append((sample_idx, int(asset["index"])))
            elif isinstance(asset, dict) and asset.get("path"):
                path_assets.append((sample_idx, asset["path"]))
            elif isinstance(asset, (str, Path)):
                path_assets.append((sample_idx, asset))

    for tsv_file, entries in tsv_assets.items():
        df = resolve_tsv_images(pd.read_csv(tsv_file, sep="\t"))
        for sample_idx, row_idx in entries:
            if row_idx >= len(df):
                continue
            try:
                img_base64 = df.iloc[row_idx]["image"]
                img_data = base64.b64decode(img_base64)
                image_groups[sample_idx].append(Image.open(BytesIO(img_data)).convert("RGB"))
            except Exception:
                continue

    for sample_idx, image_path in path_assets:
        try:
            image_groups[sample_idx].append(Image.open(image_path).convert("RGB"))
        except Exception:
            continue

    return image_groups


def extract_from_preloaded_image_groups(
    vision_encoder: VisionEncoder,
    image_groups: List[List[Image.Image]],
    batch_size: int,
    pooling: str = "mean",
) -> np.ndarray:
    flat_images = []
    image_to_sample = []
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
        per_sample_features[sample_idx].append(feature)

    embeddings = []
    for features_list in per_sample_features:
        if not features_list:
            embedding = np.zeros(vision_encoder.dimension, dtype=np.float32)
        elif len(features_list) == 1:
            embedding = np.asarray(features_list[0], dtype=np.float32)
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


def summarize(times: List[Dict[str, float]], num_samples: int) -> Dict[str, float]:
    keys = [key for key in times[0] if key.endswith("_s")]
    summary = {"num_samples": int(num_samples), "runs": len(times)}
    for key in keys:
        values = np.array([row[key] for row in times], dtype=np.float64)
        prefix = key[:-2]
        summary[f"{prefix}_ms_per_sample_mean"] = float(values.mean() * 1000.0 / num_samples)
        summary[f"{prefix}_ms_per_sample_std"] = float(values.std(ddof=0) * 1000.0 / num_samples)
        summary[f"{prefix}_throughput_samples_per_sec"] = float(num_samples / values.mean())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile online SCOPE-Router latency")
    parser.add_argument("--dataset-dir", default=".", help="Dataset root directory")
    parser.add_argument("--model-path", required=True, help="Trained SCOPE-Router .pkl")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--num-samples", type=int, default=128, help="Number of split samples to profile; <=0 uses all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-encoder", default="BAAI/bge-m3")
    parser.add_argument("--vision-encoder", default="facebook/dinov2-large")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--preload-images", action="store_true",
                        help="Decode/load selected images before timed runs; timed vision latency is encoder compute only")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = dataset_dir / model_path

    print("=" * 80)
    print("Online SCOPE-Router latency profiling")
    print("=" * 80)
    print(f"Dataset: {dataset_dir}")
    print(f"Model: {model_path}")
    print(f"Split: {args.split}, num_samples={args.num_samples}")
    print(f"Text encoder: {args.text_encoder}")
    print(f"Vision encoder: {args.vision_encoder}")
    print(f"Batch size: {args.batch_size}")
    print(f"Preload images: {args.preload_images}")

    samples = select_samples(dataset_dir, args.split, args.num_samples, args.seed)
    texts = [sample.get("prompt", sample.get("text", sample.get("question", ""))) for sample in samples]
    print(f"Selected samples: {len(samples)}")

    router = ScopeRouter.load(str(model_path))
    router.verbose = 0
    text_encoder = TextEncoder(
        model_name=args.text_encoder,
        device=args.device,
        batch_size=args.batch_size,
        show_progress=False,
    )
    vision_encoder = VisionEncoder(
        model_name=args.vision_encoder,
        device=args.device,
        batch_size=args.batch_size,
    )

    image_groups = None
    image_preload_s = None
    if args.preload_images:
        print("\nPreloading images outside timed runs...")
        image_groups, image_preload_s = timed(
            "image preload/decode",
            lambda: preload_image_groups(samples),
        )
        num_images = sum(len(images) for images in image_groups)
        samples_with_images = sum(1 for images in image_groups if images)
        print(f"  Preloaded images: {num_images} across {samples_with_images}/{len(samples)} samples")

    def run_once() -> Dict[str, float]:
        text_emb, text_s = timed(
            "text encoder",
            lambda: text_encoder.extract(texts).astype(np.float32),
        )
        if image_groups is not None:
            vision_emb, vision_s = timed(
                "vision encoder compute-only",
                lambda: extract_from_preloaded_image_groups(
                    vision_encoder,
                    image_groups,
                    batch_size=args.batch_size,
                ),
            )
        else:
            vision_emb, vision_s = timed(
                "vision encoder",
                lambda: vision_encoder.extract_from_samples(samples, batch_size=args.batch_size).astype(np.float32),
            )
        _, route_s = timed(
            "fusion + SCOPE-Router router",
            lambda: router.predict(X_text=text_emb, X_vision=vision_emb),
        )
        return {
            "text_s": text_s,
            "vision_s": vision_s,
            "router_s": route_s,
            "total_s": text_s + vision_s + route_s,
        }

    for idx in range(args.warmup_runs):
        print(f"\nWarmup {idx + 1}/{args.warmup_runs}")
        run_once()

    times = []
    for idx in range(args.runs):
        print(f"\nRun {idx + 1}/{args.runs}")
        times.append(run_once())

    summary = summarize(times, len(samples))
    result = {
        "model_path": str(model_path),
        "split": args.split,
        "text_encoder": args.text_encoder,
        "vision_encoder": args.vision_encoder,
        "batch_size": args.batch_size,
        "num_samples": len(samples),
        "warmup_runs": args.warmup_runs,
        "runs": args.runs,
        "preload_images": args.preload_images,
        "image_preload_s": image_preload_s,
        "summary": summary,
        "raw_runs": times,
    }

    print("\n" + "=" * 80)
    print("Latency summary")
    print("=" * 80)
    for key, value in summary.items():
        if key in {"num_samples", "runs"}:
            print(f"  {key}: {value}")
        elif key.endswith("mean") or key.endswith("std"):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value:.2f}")

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = dataset_dir / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
