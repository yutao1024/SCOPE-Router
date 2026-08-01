#!/usr/bin/env python3
"""Extract joint query embeddings from a Qwen-VL style model.

The output parquet has columns:
  - sample_id
  - embedding

It can be consumed by:
  tools/build_calibration_profile.py --query-embedding-file ...
  routers/scope_router/train_and_eval.py --query_embedding_file ...
"""

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_samples(benchmarks_dir: Path):
    samples = []
    for task_dir in sorted(benchmarks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        for samples_file in sorted(task_dir.glob("*_samples.jsonl")):
            with samples_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
    return samples


def blank_image(size: int = 224):
    from PIL import Image

    return Image.new("RGB", (size, size), color="white")


def load_tsv_image(asset: dict, dataset_dir: Path, cache: dict):
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
        if "image" in df.columns and "index" in df.columns:
            df["image"] = df["image"].astype(str)
            image_map = {str(idx): img for idx, img in zip(df["index"], df["image"])}
            for idx, value in list(image_map.items()):
                if len(value) <= 64 and value in image_map and len(image_map[value]) > 64:
                    image_map[idx] = image_map[value]
            cache[key] = (df, image_map)
        else:
            cache[key] = (df, None)

    df, image_map = cache[key]
    row_idx = int(asset.get("index", asset.get("lineno", 0)))
    if row_idx >= len(df):
        return None
    row = df.iloc[row_idx]
    if image_map is not None and "index" in row:
        img_str = image_map.get(str(row["index"]), row.get("image", ""))
    else:
        img_str = row["image"] if "image" in df.columns else row.iloc[1]
    if not img_str:
        return None
    return Image.open(BytesIO(base64.b64decode(str(img_str)))).convert("RGB")


def load_image(sample: dict, dataset_dir: Path, tsv_cache: dict):
    from PIL import Image

    assets = sample.get("assets") or []
    if not assets:
        return None
    asset = assets[0]
    try:
        if isinstance(asset, dict) and asset.get("type") == "image_tsv":
            return load_tsv_image(asset, dataset_dir, tsv_cache)
        if isinstance(asset, dict):
            image_path = asset.get("path") or asset.get("uri")
        else:
            image_path = asset
        if not image_path:
            return None
        path = Path(image_path)
        if not path.is_absolute():
            path = dataset_dir / path
        if path.exists():
            return Image.open(path).convert("RGB")
    except Exception:
        return None
    return None


def get_text(sample: dict) -> str:
    text = sample.get("prompt") or sample.get("text") or sample.get("question") or ""
    return str(text) if text else "No text available"


def load_model(model_name: str, device: str, dtype: str):
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    torch_dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    model_kwargs = {
        "trust_remote_code": True,
        "output_hidden_states": True,
    }
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModel.from_pretrained(model_name, **model_kwargs)
    model.eval()
    model.to(device)
    return processor, model


def prepare_inputs(processor, text: str, image, device: str):
    import torch

    if image is not None and hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True, truncation=True)
    elif image is None and hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            }
        ]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        inputs = processor(text=[prompt], return_tensors="pt", padding=True, truncation=True)
    elif image is None:
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    else:
        # Fallback for non-chat-template processors.
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True, truncation=True)
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}


def build_prompt(processor, text: str, image) -> str:
    if hasattr(processor, "apply_chat_template"):
        if image is None:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": text}],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text},
                    ],
                }
            ]
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return text


def prepare_batch_inputs(processor, texts, images, device: str):
    import torch

    prompts = [build_prompt(processor, text, image) for text, image in zip(texts, images)]
    image_list = [image for image in images if image is not None]
    if image_list:
        inputs = processor(text=prompts, images=image_list, return_tensors="pt", padding=True, truncation=True)
    else:
        inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}


def mean_pool_last_hidden(outputs, inputs):
    hidden = None
    if getattr(outputs, "hidden_states", None):
        hidden = outputs.hidden_states[-1]
    elif getattr(outputs, "last_hidden_state", None) is not None:
        hidden = outputs.last_hidden_state
    elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        hidden = outputs[0]
    if hidden is None:
        raise ValueError("Could not find hidden states in model outputs")

    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        pooled = hidden.mean(dim=1)
    else:
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
    return pooled


def write_rows(rows, output_path: Path):
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    pd.DataFrame(rows).to_parquet(tmp_path, index=False)
    tmp_path.replace(output_path)


def load_existing_rows(output_path: Path):
    if not output_path.exists():
        return [], set()
    df = pd.read_parquet(output_path)
    if "sample_id" not in df.columns or "embedding" not in df.columns:
        raise ValueError(f"Existing file has unexpected columns: {output_path}")
    rows = df[["sample_id", "embedding"]].to_dict("records")
    seen = set(df["sample_id"].astype(str))
    return rows, seen


def main():
    parser = argparse.ArgumentParser(description="Extract Qwen-VL joint query embeddings")
    parser.add_argument("--dataset_dir", default=".", help="Dataset root directory")
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-4B-Instruct",
                        help="HuggingFace model name or local path")
    parser.add_argument("--output_file", default="EMBEDDINGS/query/qwen3-vl-4b.parquet")
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu. Defaults to cuda if available.")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for Qwen-VL forward passes. Default 1 is safest; try 2/4 if VRAM allows.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from an existing output parquet by skipping saved sample_ids.")
    parser.add_argument("--save_every", type=int, default=500,
                        help="Write intermediate parquet every N new samples. Set <=0 to only save at the end.")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test sample limit")
    parser.add_argument("--use_blank_image_for_text_only", action="store_true",
                        help="Pass a blank image for text-only samples instead of processor text-only mode")
    args = parser.parse_args()

    import torch
    from tqdm import tqdm

    dataset_dir = Path(args.dataset_dir)
    benchmarks_dir = dataset_dir / "BENCHMARKS"
    if not benchmarks_dir.exists():
        raise FileNotFoundError(f"Missing BENCHMARKS directory: {benchmarks_dir}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_samples(benchmarks_dir)
    if args.limit:
        samples = samples[:args.limit]
    print(f"Loaded {len(samples)} samples")
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")

    processor, model = load_model(args.model_name, device=device, dtype=args.dtype)
    output_path = Path(args.output_file)
    if not output_path.is_absolute():
        output_path = dataset_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    seen_sample_ids = set()
    if args.resume:
        rows, seen_sample_ids = load_existing_rows(output_path)
        if seen_sample_ids:
            samples = [sample for sample in samples if str(sample["sample_id"]) not in seen_sample_ids]
            print(f"Resume: loaded {len(seen_sample_ids)} existing embeddings")
            print(f"Remaining samples: {len(samples)}")

    tsv_cache = {}
    batch_size = max(1, int(args.batch_size))
    save_every = int(args.save_every)
    new_rows_since_save = 0
    with torch.inference_mode():
        for batch_start in tqdm(range(0, len(samples), batch_size), desc="extract qwen-vl embeddings"):
            batch_samples = samples[batch_start:batch_start + batch_size]
            texts = []
            images = []
            for sample in batch_samples:
                image = load_image(sample, dataset_dir, tsv_cache)
                if image is None and args.use_blank_image_for_text_only:
                    image = blank_image()
                texts.append(get_text(sample))
                images.append(image)

            inputs = prepare_batch_inputs(processor, texts, images, device)
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            pooled = mean_pool_last_hidden(outputs, inputs)
            batch_embeddings = pooled.detach().float().cpu().numpy()
            norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            batch_embeddings = batch_embeddings / norms
            for sample, embedding in zip(batch_samples, batch_embeddings):
                rows.append({
                    "sample_id": sample["sample_id"],
                    "embedding": embedding.astype(np.float32),
                })
                seen_sample_ids.add(str(sample["sample_id"]))
                new_rows_since_save += 1

            if save_every > 0 and new_rows_since_save >= save_every:
                write_rows(rows, output_path)
                new_rows_since_save = 0

    write_rows(rows, output_path)
    print(f"Saved: {output_path}")
    print(f"Embedding dim: {len(rows[0]['embedding']) if rows else 0}")


if __name__ == "__main__":
    main()
