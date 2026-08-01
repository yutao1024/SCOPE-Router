#!/usr/bin/env python3
"""Fixed-input latency benchmark for VL-RouterBench routers.

This script fixes the input scale and measures architecture/runtime latency
for method-level latency tables, not accuracy evaluation.

Feature-level routers use a shared BGE/DINO feature extraction latency plus a
synthetic decision head with matching input/model/calibration scale. End-to-end
routers can load real checkpoints and run their actual predict(meta=...) path
on the same fixed text/image input.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"PyTorch is required: {exc}")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

E2E_LOADERS: Dict[str, Tuple[str, str]] = {
    "routerdc": ("routers.routerdc.router", "RouterDCRouter"),
    "zooter": ("routers.zooter.router", "ZooterRouter"),
    "vlc": ("routers.vlc.router", "VLCRouter"),
    "cosinecls": ("routers.cosinecls.router", "CosineCLSRRouter"),
}

_INPUT_CACHE: Optional[Dict[str, Any]] = None


def parse_router_spec(spec: str) -> Tuple[str, str, Path]:
    """Parse name=method:path or method:path."""
    name: Optional[str] = None
    body = spec
    if "=" in spec:
        name, body = spec.split("=", 1)
    if ":" not in body:
        raise ValueError(f"End-to-end checkpoint spec must be method:path, got: {spec}")
    method, path_s = body.split(":", 1)
    method = method.strip().lower().replace("-", "_")
    if method not in E2E_LOADERS:
        raise ValueError(f"Unsupported checkpoint method: {method}. Use one of {sorted(E2E_LOADERS)}")
    path = Path(path_s)
    if name is None or not name.strip():
        name = f"{method}:{path.parent.name}/{path.stem}"
    return name.strip(), method, path


def load_e2e_router(method: str, path: Path, device: str):
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Router checkpoint not found: {path}")
    module_name, class_name = E2E_LOADERS[method]
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    try:
        return cls.load(str(path), device=device)
    except TypeError:
        router = cls.load(str(path))
        if hasattr(router, "device"):
            router.device = device
        return router


def sync_cuda(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def stats_ms(times_s: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(times_s, dtype=np.float64) * 1000.0
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def timed_runs(fn: Callable[[], Any], warmup_runs: int, test_runs: int, device: str) -> Dict[str, Any]:
    for _ in range(warmup_runs):
        _ = fn()
        sync_cuda(device)

    times = []
    for _ in range(test_runs):
        sync_cuda(device)
        start = time.perf_counter()
        _ = fn()
        sync_cuda(device)
        times.append(time.perf_counter() - start)
    return {"stats": stats_ms(times), "raw_s": times}


def throughput_row(
    *,
    router: str,
    method: str,
    component: str,
    batch_size: int,
    num_models: int,
    input_dim: int,
    result: Dict[str, Any],
    device: str,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    st = result["stats"]
    mean_s = st["mean"] / 1000.0
    row = {
        "router": router,
        "method": method,
        "component": component,
        "device": device,
        "batch_size": int(batch_size),
        "num_models": int(num_models),
        "input_dim": int(input_dim),
        "latency_ms_mean": st["mean"],
        "latency_ms_std": st["std"],
        "latency_ms_min": st["min"],
        "latency_ms_p50": st["p50"],
        "latency_ms_p95": st["p95"],
        "latency_ms_p99": st["p99"],
        "latency_ms_max": st["max"],
        "ms_per_sample": st["mean"] / batch_size,
        "samples_per_sec": float(batch_size / mean_s) if mean_s > 0 else math.inf,
        "checkpoint_free": True,
        "error": "",
    }
    if extra:
        row.update(extra)
    return row


def combine_rows(router_row: Dict[str, Any], encoder_row: Dict[str, Any], *, router: str, method: str) -> Dict[str, Any]:
    total = dict(router_row)
    total["router"] = router
    total["method"] = method
    total["component"] = "online_total_encoder_plus_router"
    total["latency_ms_mean"] = encoder_row["latency_ms_mean"] + router_row["latency_ms_mean"]
    total["latency_ms_std"] = math.sqrt(encoder_row["latency_ms_std"] ** 2 + router_row["latency_ms_std"] ** 2)
    for key in ["min", "p50", "p95", "p99", "max"]:
        total[f"latency_ms_{key}"] = encoder_row[f"latency_ms_{key}"] + router_row[f"latency_ms_{key}"]
    total["ms_per_sample"] = total["latency_ms_mean"] / total["batch_size"]
    total["samples_per_sec"] = float(total["batch_size"] / (total["latency_ms_mean"] / 1000.0))
    total["encoder_latency_ms_mean"] = encoder_row["latency_ms_mean"]
    total["router_latency_ms_mean"] = router_row["latency_ms_mean"]
    return total


def measure_feature_encoder(args) -> Dict[str, Any]:
    from routers.features.text_encoder import TextEncoder
    from routers.features.vision_encoder import VisionEncoder

    payload = get_latency_inputs(args)
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

    def run():
        text_emb = text_encoder.extract(payload["texts"]).astype(np.float32)
        vision_emb = vision_encoder._extract_batch_from_images(payload["images"], args.batch_size).astype(np.float32)
        return text_emb, vision_emb

    result = timed_runs(run, args.warmup_runs, args.test_runs, args.device)
    return throughput_row(
        router="encoder_bge_dino",
        method="feature_encoder",
        component="bge_m3_plus_dinov2_large",
        batch_size=args.batch_size,
        num_models=args.num_models,
        input_dim=args.input_dim,
        result=result,
        device=args.device,
        extra={
            "text_encoder": args.text_encoder,
            "vision_encoder": args.vision_encoder,
            "sample_source": args.sample_source,
            "sample_ids": ";".join(payload["sample_ids"]),
            "text_chars_mean": payload["text_chars_mean"],
        },
    )


def linear_head(input_dim: int, num_models: int) -> nn.Module:
    return nn.Linear(input_dim, num_models)


def mlp_head(input_dim: int, num_models: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, num_models),
    )


def scope_router_head(input_dim: int, num_models: int, embedding_dim: int = 64, hidden_dim: int = 128) -> nn.Module:
    class SyntheticScopeRouter(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, embedding_dim),
            )
            self.profile = nn.Parameter(torch.randn(num_models, embedding_dim))

        def forward(self, x):
            q = self.query(x)
            return q @ self.profile.T

    return SyntheticScopeRouter()


def measure_torch_head(name: str, module: nn.Module, args, encoder_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    module = module.to(args.device).eval()
    x = torch.randn(args.batch_size, args.input_dim, device=args.device)

    def run():
        with torch.inference_mode():
            return module(x).argmax(dim=-1)

    result = timed_runs(run, args.warmup_runs, args.test_runs, args.device)
    decision = throughput_row(
        router=name,
        method=name,
        component="router_decision_only",
        batch_size=args.batch_size,
        num_models=args.num_models,
        input_dim=args.input_dim,
        result=result,
        device=args.device,
    )
    return [decision, combine_rows(decision, encoder_row, router=name, method=name)]


def measure_numpy_router(name: str, args, encoder_row: Dict[str, Any], kind: str) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(45)
    x = rng.normal(size=(args.batch_size, args.input_dim)).astype(np.float32)
    if kind == "kmeans":
        prototypes = rng.normal(size=(args.num_models, args.input_dim)).astype(np.float32)

        def run():
            return np.argmax(x @ prototypes.T, axis=1)
    elif kind == "knn":
        memory = rng.normal(size=(args.calib_size, args.input_dim)).astype(np.float32)
        labels = rng.integers(0, args.num_models, size=args.calib_size)

        def run():
            dist = ((x[:, None, :] - memory[None, :, :]) ** 2).sum(axis=-1)
            nn_idx = np.argpartition(dist, args.knn_k, axis=1)[:, :args.knn_k]
            return np.array([np.bincount(labels[row], minlength=args.num_models).argmax() for row in nn_idx])
    elif kind == "prknn":
        memory = rng.normal(size=(args.calib_size, args.input_dim)).astype(np.float32)
        scores = rng.normal(size=(args.calib_size, args.num_models)).astype(np.float32)

        def run():
            dist = ((x[:, None, :] - memory[None, :, :]) ** 2).sum(axis=-1)
            nn_idx = np.argpartition(dist, args.knn_k, axis=1)[:, :args.knn_k]
            return scores[nn_idx].mean(axis=1).argmax(axis=1)
    else:
        weights = rng.normal(size=(args.input_dim, args.num_models)).astype(np.float32)

        def run():
            return np.argmax(x @ weights, axis=1)

    result = timed_runs(run, args.warmup_runs, args.test_runs, "cpu")
    decision = throughput_row(
        router=name,
        method=name,
        component="router_decision_only",
        batch_size=args.batch_size,
        num_models=args.num_models,
        input_dim=args.input_dim,
        result=result,
        device="cpu",
        extra={"calib_size": args.calib_size if kind in {"knn", "prknn"} else ""},
    )
    return [decision, combine_rows(decision, encoder_row, router=name, method=name)]


def synthetic_lxmert_inputs(args):
    images = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=args.device)
    input_ids = torch.ones(args.batch_size, args.seq_len, dtype=torch.long, device=args.device)
    attention_mask = torch.ones(args.batch_size, args.seq_len, dtype=torch.long, device=args.device)
    return images, input_ids, attention_mask


def measure_routerdc(args) -> Dict[str, Any]:
    from routers.routerdc.router import RouterDCModel

    model = RouterDCModel(num_classes=args.num_models).to(args.device).eval()
    images, input_ids, attention_mask = synthetic_lxmert_inputs(args)

    def run():
        with torch.inference_mode():
            return model(images, input_ids, attention_mask)

    result = timed_runs(run, args.warmup_runs, args.test_runs, args.device)
    return throughput_row(
        router="routerdc",
        method="routerdc",
        component="untrained_vit_lxmert_forward",
        batch_size=args.batch_size,
        num_models=args.num_models,
        input_dim=args.input_dim,
        result=result,
        device=args.device,
        extra={"seq_len": args.seq_len, "image_size": args.image_size},
    )


def measure_cosinecls(args) -> Dict[str, Any]:
    from routers.cosinecls.router import CosineCLSRModel

    model = CosineCLSRModel(num_classes=args.num_models).to(args.device).eval()
    images, input_ids, attention_mask = synthetic_lxmert_inputs(args)

    def run():
        with torch.inference_mode():
            return model(images, input_ids, attention_mask)

    result = timed_runs(run, args.warmup_runs, args.test_runs, args.device)
    return throughput_row(
        router="cosinecls",
        method="cosinecls",
        component="untrained_vit_lxmert_forward",
        batch_size=args.batch_size,
        num_models=args.num_models,
        input_dim=args.input_dim,
        result=result,
        device=args.device,
        extra={"seq_len": args.seq_len, "image_size": args.image_size},
    )


def measure_zooter(args) -> Dict[str, Any]:
    from routers.zooter.router import LXMERTClassifier

    model = LXMERTClassifier(num_classes=args.num_models).to(args.device).eval()
    images, input_ids, attention_mask = synthetic_lxmert_inputs(args)

    def run():
        with torch.inference_mode():
            return model(images, input_ids, attention_mask)

    result = timed_runs(run, args.warmup_runs, args.test_runs, args.device)
    return throughput_row(
        router="zooter",
        method="zooter",
        component="untrained_vit_lxmert_classifier_forward",
        batch_size=args.batch_size,
        num_models=args.num_models,
        input_dim=args.input_dim,
        result=result,
        device=args.device,
        extra={"seq_len": args.seq_len, "image_size": args.image_size},
    )


def measure_vlc(args) -> Dict[str, Any]:
    from routers.vlc.router import VLClassifier

    model = VLClassifier(
        num_classes=args.num_models,
        model_type=args.vlc_model_type,
        freeze_backbone=True,
    ).to(args.device).eval()
    images = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=args.device)
    texts = [args.text] * args.batch_size

    def run():
        with torch.inference_mode():
            return model(images, texts)

    result = timed_runs(run, args.warmup_runs, args.test_runs, args.device)
    return throughput_row(
        router=f"vlc_{args.vlc_model_type}",
        method="vlc",
        component=f"untrained_vit_{args.vlc_model_type}_forward",
        batch_size=args.batch_size,
        num_models=args.num_models,
        input_dim=args.input_dim,
        result=result,
        device=args.device,
        extra={"seq_len": args.seq_len, "image_size": args.image_size, "vlc_model_type": args.vlc_model_type},
    )


def build_fixed_image_path(args) -> Path:
    from PIL import Image

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "fixed_latency_image.png"
    image = Image.new("RGB", (args.image_size, args.image_size), color="white")
    image.save(image_path)
    return image_path


def load_benchmark_samples(dataset_dir: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    benchmarks_dir = dataset_dir / "BENCHMARKS"
    if not benchmarks_dir.exists():
        raise FileNotFoundError(f"Missing BENCHMARKS directory: {benchmarks_dir}")
    for task_dir in sorted(benchmarks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        for samples_file in sorted(task_dir.glob("*_samples.jsonl")):
            with samples_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
    if not samples:
        raise ValueError(f"No *_samples.jsonl rows found under {benchmarks_dir}")
    return samples


def sample_text(sample: Dict[str, Any]) -> str:
    text = sample.get("prompt") or sample.get("text") or sample.get("question") or ""
    return str(text) if text else "No text available"


def load_tsv_image(asset: Dict[str, Any], dataset_dir: Path, cache: Dict[str, Any]):
    from PIL import Image

    tsv_file = asset.get("tsv_file") or asset.get("path") or asset.get("uri")
    if not tsv_file:
        return None
    path = Path(tsv_file)
    if not path.is_absolute():
        path = dataset_dir / path
    key = str(path)
    if key not in cache:
        df = pd.read_csv(path, sep="\t")
        if "image" not in df.columns:
            raise ValueError(f"TSV image file missing image column: {path}")
        image_map = None
        if "index" in df.columns:
            image_map = {str(idx): img for idx, img in zip(df["index"], df["image"])}
            for idx, value in list(image_map.items()):
                value = str(value)
                if len(value) <= 64 and value in image_map and len(str(image_map[value])) > 64:
                    image_map[idx] = image_map[value]
        cache[key] = (df, image_map)

    df, image_map = cache[key]
    row_idx = int(asset.get("index", asset.get("lineno", 0)))
    if row_idx >= len(df):
        raise IndexError(f"TSV row index {row_idx} out of range for {path}")
    row = df.iloc[row_idx]
    if image_map is not None and "index" in row:
        img_str = image_map.get(str(row["index"]), row.get("image", ""))
    else:
        img_str = row["image"]
    return Image.open(BytesIO(base64.b64decode(str(img_str)))).convert("RGB")


def load_sample_image(sample: Dict[str, Any], dataset_dir: Path, tsv_cache: Dict[str, Any], image_size: int):
    from PIL import Image

    assets = sample.get("assets") or []
    if not assets:
        return Image.new("RGB", (image_size, image_size), color="white")
    asset = assets[0]
    if isinstance(asset, dict) and asset.get("type") == "image_tsv":
        image = load_tsv_image(asset, dataset_dir, tsv_cache)
    else:
        image_path = (asset.get("path") or asset.get("uri")) if isinstance(asset, dict) else asset
        path = Path(image_path)
        if not path.is_absolute():
            path = dataset_dir / path
        image = Image.open(path).convert("RGB")
    return image if image is not None else Image.new("RGB", (image_size, image_size), color="white")


def build_fixed_payload(args) -> Dict[str, Any]:
    from PIL import Image

    image_path = str(build_fixed_image_path(args).resolve())
    texts = [args.text] * args.batch_size
    images = [Image.new("RGB", (args.image_size, args.image_size), color="white") for _ in range(args.batch_size)]
    sample_ids = [f"latency-{i}" for i in range(args.batch_size)]
    text_lens = [len(t) for t in texts]
    return {
        "meta": pd.DataFrame(
            {
                "sample_id": sample_ids,
                "text": texts,
                "prompt": texts,
                "question": texts,
                "image": [image_path] * args.batch_size,
                "image_path": [image_path] * args.batch_size,
                "assets": [[{"type": "image", "path": image_path}] for _ in range(args.batch_size)],
            }
        ),
        "texts": texts,
        "images": images,
        "sample_ids": sample_ids,
        "text_chars_mean": float(np.mean(text_lens)) if text_lens else 0.0,
        "text_chars_min": int(min(text_lens)) if text_lens else 0,
        "text_chars_max": int(max(text_lens)) if text_lens else 0,
    }


def build_dataset_payload(args) -> Dict[str, Any]:
    dataset_dir = Path(args.dataset_dir)
    if args.sample_file:
        with Path(args.sample_file).open("r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]
    else:
        samples = load_benchmark_samples(dataset_dir)
    if len(samples) < args.batch_size:
        raise ValueError(f"Need batch_size={args.batch_size} samples, found {len(samples)}")

    rng = np.random.default_rng(args.sample_seed)
    if args.sample_strategy == "random":
        indices = rng.choice(len(samples), size=args.batch_size, replace=False)
        selected = [samples[int(i)] for i in indices]
    else:
        selected = samples[: args.batch_size]

    materialized_dir = Path(args.output_dir) / "fixed_latency_samples"
    materialized_dir.mkdir(parents=True, exist_ok=True)
    tsv_cache: Dict[str, Any] = {}
    rows = []
    texts = []
    images = []
    sample_ids = []
    for i, sample in enumerate(selected):
        sid = str(sample.get("sample_id", f"latency-{i}"))
        text = sample_text(sample)
        image = load_sample_image(sample, dataset_dir, tsv_cache, args.image_size)
        image_path = (materialized_dir / f"sample_{i}.png").resolve()
        image.save(image_path)
        image_path_s = str(image_path)
        rows.append(
            {
                "sample_id": sid,
                "dataset": sample.get("dataset", ""),
                "task_type": sample.get("task_type", ""),
                "text": text,
                "prompt": text,
                "question": text,
                "image": image_path_s,
                "image_path": image_path_s,
                "assets": [{"type": "image", "path": image_path_s}],
            }
        )
        texts.append(text)
        images.append(image)
        sample_ids.append(sid)

    text_lens = [len(t) for t in texts]
    return {
        "meta": pd.DataFrame(rows),
        "texts": texts,
        "images": images,
        "sample_ids": sample_ids,
        "text_chars_mean": float(np.mean(text_lens)) if text_lens else 0.0,
        "text_chars_min": int(min(text_lens)) if text_lens else 0,
        "text_chars_max": int(max(text_lens)) if text_lens else 0,
    }


def get_latency_inputs(args) -> Dict[str, Any]:
    global _INPUT_CACHE
    if _INPUT_CACHE is None:
        _INPUT_CACHE = build_dataset_payload(args) if args.sample_source == "dataset" else build_fixed_payload(args)
    return _INPUT_CACHE


def build_fixed_meta(args) -> pd.DataFrame:
    return get_latency_inputs(args)["meta"]


def measure_e2e_checkpoint(name: str, method: str, path: Path, args) -> Dict[str, Any]:
    router = load_e2e_router(method, path, args.device)
    if hasattr(router, "batch_size"):
        router.batch_size = args.batch_size
    if hasattr(router, "verbose"):
        router.verbose = 0
    payload = get_latency_inputs(args)
    meta = payload["meta"]

    def run():
        return router.predict(
            meta=meta,
            batch_size=args.batch_size,
            num_workers=0,
            show_progress=False,
        )

    result = timed_runs(run, args.warmup_runs, args.test_runs, args.device)
    return throughput_row(
        router=name,
        method=method,
        component="real_checkpoint_predict_fixed_input",
        batch_size=args.batch_size,
        num_models=args.num_models,
        input_dim=args.input_dim,
        result=result,
        device=args.device,
        extra={
            "checkpoint_path": str(path),
            "checkpoint_free": False,
            "image_size": args.image_size,
            "sample_source": args.sample_source,
            "sample_ids": ";".join(payload["sample_ids"]),
            "text_chars_mean": payload["text_chars_mean"],
            "text_chars_min": payload["text_chars_min"],
            "text_chars_max": payload["text_chars_max"],
            "checkpoint_input_mode": "raw_meta_predict",
        },
    )


def write_outputs(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "detailed_results.json"
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[save] {csv_path}")
    print(f"[save] {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-input router latency benchmark")
    parser.add_argument("--output-dir", default="outputs/router_throughput/synthetic_bs1")
    parser.add_argument("--dataset-dir", default=".")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-models", type=int, default=7)
    parser.add_argument("--input-dim", type=int, default=2048, help="bge-m3 1024 + dinov2-large 1024 under normalize_concat")
    parser.add_argument("--calib-size", type=int, default=1024)
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--text", default="What is shown in the image? Answer briefly.")
    parser.add_argument("--sample-source", default="fixed", choices=["fixed", "dataset"],
                        help="fixed uses --text plus a blank image; dataset uses the same BENCHMARKS samples for encoder and checkpoint routers")
    parser.add_argument("--sample-file", default="",
                        help="Optional JSONL sample file for --sample-source dataset; defaults to all BENCHMARKS/*/*_samples.jsonl")
    parser.add_argument("--sample-strategy", default="first", choices=["first", "random"])
    parser.add_argument("--sample-seed", type=int, default=45)
    parser.add_argument("--text-encoder", default="BAAI/bge-m3")
    parser.add_argument("--vision-encoder", default="facebook/dinov2-large")
    parser.add_argument("--vlc-model-type", default="lxmert", choices=["visualbert", "lxmert", "uniter", "vilbert"])
    parser.add_argument("--e2e-mode", default="synthetic", choices=["synthetic", "checkpoint"],
                        help="For RouterDC/Zooter/VLC/CosineCLS: synthetic initializes architecture; checkpoint loads real trained checkpoints")
    parser.add_argument("--router", action="append", default=[],
                        help="End-to-end checkpoint spec for --e2e-mode checkpoint: name=method:path or method:path")
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--test-runs", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument("--methods", default="all",
                        help="Comma-separated methods or all. Methods: encoder,linear,mlp,ovr,kmeans,knn,prknn,scope_router,routerdc,zooter,vlc,cosinecls")
    args = parser.parse_args()

    methods = [
        "encoder", "linear", "mlp", "ovr", "kmeans", "knn", "prknn",
        "scope_router",
        "routerdc", "zooter", "vlc", "cosinecls",
    ] if args.methods == "all" else [m.strip() for m in args.methods.split(",") if m.strip()]

    rows: List[Dict[str, Any]] = []
    encoder_row = None
    if any(m in methods for m in ["encoder", "linear", "mlp", "ovr", "kmeans", "knn", "prknn", "scope_router"]):
        print("[measure] shared BGE/DINO encoder")
        try:
            encoder_row = measure_feature_encoder(args)
            rows.append(encoder_row)
        except Exception as exc:
            encoder_row = throughput_row(
                router="encoder_bge_dino",
                method="feature_encoder",
                component="bge_m3_plus_dinov2_large",
                batch_size=args.batch_size,
                num_models=args.num_models,
                input_dim=args.input_dim,
                result={"stats": {k: -1 for k in ["mean", "std", "min", "p50", "p95", "p99", "max"]}},
                device=args.device,
                extra={"error": str(exc)},
            )
            rows.append(encoder_row)

    feature_builders = {
        "linear": lambda: linear_head(args.input_dim, args.num_models),
        "ovr": lambda: linear_head(args.input_dim, args.num_models),
        "mlp": lambda: mlp_head(args.input_dim, args.num_models),
        "scope_router": lambda: scope_router_head(args.input_dim, args.num_models),
    }
    for method, builder in feature_builders.items():
        if method not in methods:
            continue
        print(f"[measure] {method}")
        try:
            rows.extend(measure_torch_head(method, builder(), args, encoder_row))
        except Exception as exc:
            rows.append({"router": method, "method": method, "component": "error", "error": str(exc), "checkpoint_free": True})

    numpy_kinds = {
        "kmeans": "kmeans",
        "knn": "knn",
        "prknn": "prknn",
    }
    for method, kind in numpy_kinds.items():
        if method not in methods:
            continue
        print(f"[measure] {method}")
        try:
            rows.extend(measure_numpy_router(method, args, encoder_row, kind))
        except Exception as exc:
            rows.append({"router": method, "method": method, "component": "error", "error": str(exc), "checkpoint_free": True})

    if args.e2e_mode == "checkpoint":
        if not args.router:
            print("[warn] --e2e-mode checkpoint requested but no --router checkpoint specs were supplied")
        for spec in args.router:
            try:
                name, method, path = parse_router_spec(spec)
                if method not in methods and "all" not in args.methods:
                    continue
                print(f"[measure] checkpoint {name}")
                rows.append(measure_e2e_checkpoint(name, method, path, args))
            except Exception as exc:
                rows.append({
                    "router": spec,
                    "method": "unknown",
                    "component": "error",
                    "device": args.device,
                    "batch_size": args.batch_size,
                    "num_models": args.num_models,
                    "input_dim": args.input_dim,
                    "checkpoint_free": False,
                    "error": str(exc),
                })
    else:
        e2e = {
            "routerdc": measure_routerdc,
            "zooter": measure_zooter,
            "vlc": measure_vlc,
            "cosinecls": measure_cosinecls,
        }
        for method, fn in e2e.items():
            if method not in methods:
                continue
            print(f"[measure] {method}")
            try:
                rows.append(fn(args))
            except Exception as exc:
                rows.append({
                    "router": method,
                    "method": method,
                    "component": "error",
                    "device": args.device,
                    "batch_size": args.batch_size,
                    "num_models": args.num_models,
                    "input_dim": args.input_dim,
                    "checkpoint_free": True,
                    "error": str(exc),
                })

    write_outputs(rows, Path(args.output_dir))


if __name__ == "__main__":
    main()
