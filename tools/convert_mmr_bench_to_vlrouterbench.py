#!/usr/bin/env python3
"""
Convert MMR-Bench offline outcomes into the VL-RouterBench dataset layout.

Input:
  MMR-Bench/data/MMR_Bench.csv plus image folders.

Output:
  BENCHMARKS/
  SPLITS/
  data/registry/
  data/matrices/
  ORACLE/score/
  vlm_router_data/TSV_images/

The image files are packed into base64 TSVs because the existing VL-RouterBench
vision feature extractor is optimized for image_tsv assets.
"""

from __future__ import annotations

import argparse
import base64
import json
import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PREFIX_TO_DATASET = {
    "MathVerse": "MathVerse",
    "MathVision": "MathVision",
    "MathVista": "MathVista",
    "OCRBench": "OCRBench",
    "RealWorldQA": "RealWorldQA",
    "MMStar": "MMStar",
    "SEEDBench2_Plus": "SEEDBenchv2Plus",
}

DATASET_TO_TASK = {
    "MathVerse": "math",
    "MathVision": "math",
    "MathVista": "math",
    "OCRBench": "ocr_qa",
    "RealWorldQA": "vqa_oe",
    "MMStar": "vqa_mc",
    "SEEDBenchv2Plus": "vqa_mc",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert MMR-Bench to VL-RouterBench format")
    p.add_argument("--mmr-data-root", type=Path, required=True, help="MMR-Bench data dir containing MMR_Bench.csv")
    p.add_argument("--output-dir", type=Path, required=True, help="New VL-RouterBench-format dataset root")
    p.add_argument("--models", default=None, help="Comma-separated model list. Default: complete-coverage models only")
    p.add_argument("--min-model-coverage", type=float, default=0.999,
                   help="Keep auto-detected models with correctness/cost coverage >= this value")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--dev-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--force", action="store_true", help="Overwrite output dir if it exists")
    p.add_argument("--no-image-tsv", action="store_true",
                   help="Do not pack images into TSVs; write direct image uri assets instead")
    return p.parse_args()


def choose_prefix(dataset_idx: str) -> str:
    matches = [p for p in PREFIX_TO_DATASET if dataset_idx == p or dataset_idx.startswith(p + "_")]
    if not matches:
        raise ValueError(f"Cannot infer MMR dataset prefix from dataset_idx={dataset_idx!r}")
    return max(matches, key=len)


def parse_dataset_idx(dataset_idx: str) -> tuple[str, str]:
    prefix = choose_prefix(str(dataset_idx))
    dataset = PREFIX_TO_DATASET[prefix]
    value = str(dataset_idx)
    if value.startswith(prefix + "_"):
        item_id = value[len(prefix) + 1:]
    else:
        item_id = value.split("_")[-1]
    return dataset, item_id


def find_image(mmr_data_root: Path, dataset: str, item_id: str) -> Path | None:
    base = mmr_data_root / dataset
    for ext in [".jpg", ".png", ".jpeg", ".webp"]:
        candidate = base / f"{item_id}{ext}"
        if candidate.exists():
            return candidate.resolve()
    return None


def infer_models(df: pd.DataFrame, models_arg: str | None, min_coverage: float) -> list[str]:
    correct = {c[: -len("_correct")] for c in df.columns if c.endswith("_correct")}
    cost = {c[: -len("_cost")] for c in df.columns if c.endswith("_cost")}
    available = sorted(correct & cost)

    if models_arg:
        requested = [m.strip() for m in models_arg.split(",") if m.strip()]
        missing = [m for m in requested if m not in available]
        if missing:
            raise ValueError(f"Requested model(s) missing from CSV correct/cost pairs: {missing}")
        return requested

    kept = []
    dropped = []
    for model in available:
        ok = df[f"{model}_correct"].notna() & df[f"{model}_cost"].notna()
        coverage = float(ok.mean())
        if coverage >= min_coverage:
            kept.append(model)
        else:
            dropped.append((model, coverage))

    if not kept:
        raise ValueError("No models passed coverage filter")
    if dropped:
        print("Dropped incomplete models:")
        for model, coverage in dropped:
            print(f"  {model}: coverage={coverage:.4f}")
    return kept


def coerce_correct(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(np.float32)
    if series.dtype.kind in {"f", "i", "u"}:
        return pd.to_numeric(series, errors="coerce").astype(np.float32)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map({
        "true": 1.0,
        "false": 0.0,
        "1": 1.0,
        "0": 0.0,
        "yes": 1.0,
        "no": 0.0,
    }).astype(np.float32)


def clean_output_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Output dir exists: {path}. Use --force to overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_splits(meta: pd.DataFrame, train_ratio: float, dev_ratio: float, test_ratio: float, seed: int) -> dict[str, list[str]]:
    ratio_sum = train_ratio + dev_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("Split ratios must sum to a positive value")
    train_ratio /= ratio_sum
    dev_ratio /= ratio_sum

    rng = np.random.default_rng(seed)
    splits = {"train": [], "dev": [], "test": []}
    for _, group in meta.groupby("dataset", sort=True):
        ids = group["sample_id"].to_numpy()
        ids = ids[rng.permutation(len(ids))]
        n = len(ids)
        n_train = int(n * train_ratio)
        n_dev = int(n * dev_ratio)
        if n >= 3:
            n_train = max(1, n_train)
            n_dev = max(1, n_dev)
            if n_train + n_dev >= n:
                n_train = max(1, n - 2)
                n_dev = 1
        splits["train"].extend(ids[:n_train].tolist())
        splits["dev"].extend(ids[n_train:n_train + n_dev].tolist())
        splits["test"].extend(ids[n_train + n_dev:].tolist())
    return splits


def pack_dataset_tsv(dataset: str, items: list[tuple[str, Path | None]], tsv_dir: Path) -> dict[str, int]:
    tsv_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = tsv_dir / f"{dataset}.tsv"
    id_to_row = {}
    rows = []
    for row_idx, (item_id, image_path) in enumerate(items):
        id_to_row[item_id] = row_idx
        image_b64 = ""
        if image_path is not None and image_path.exists():
            image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        rows.append({"index": item_id, "image": image_b64})
    pd.DataFrame(rows).to_csv(tsv_path, sep="\t", index=False)
    return id_to_row


def main() -> None:
    args = parse_args()
    csv_path = args.mmr_data_root / "MMR_Bench.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing MMR CSV: {csv_path}")

    clean_output_dir(args.output_dir, args.force)

    print(f"Loading {csv_path}")
    df = pd.read_csv(csv_path)
    if "dataset_idx" not in df.columns or "question" not in df.columns:
        raise ValueError("MMR_Bench.csv must contain dataset_idx and question columns")

    models = infer_models(df, args.models, args.min_model_coverage)
    print(f"Models ({len(models)}): {models}")

    parsed = df["dataset_idx"].astype(str).apply(parse_dataset_idx)
    df["dataset"] = [x[0] for x in parsed]
    df["item_id"] = [x[1] for x in parsed]
    df["sample_id"] = df["dataset"] + "/" + df["item_id"].astype(str)
    df["task"] = df["dataset"].map(DATASET_TO_TASK).fillna("unknown")
    df["img_path_resolved"] = [
        str(find_image(args.mmr_data_root, dataset, item_id) or "")
        for dataset, item_id in zip(df["dataset"], df["item_id"], strict=False)
    ]

    before = len(df)
    df = df.drop_duplicates("sample_id", keep="first").reset_index(drop=True)
    if len(df) != before:
        print(f"Dropped duplicate sample_ids: {before - len(df)}")

    valid_mask = np.ones(len(df), dtype=bool)
    for model in models:
        valid_mask &= df[f"{model}_correct"].notna().to_numpy()
        valid_mask &= df[f"{model}_cost"].notna().to_numpy()
    if not valid_mask.all():
        print(f"Dropping rows with missing selected-model outcomes: {int((~valid_mask).sum())}")
        df = df[valid_mask].reset_index(drop=True)

    sample_ids = df["sample_id"].tolist()
    N = len(df)
    K = len(models)
    Y = np.zeros((N, K), dtype=np.int8)
    C = np.zeros((N, K), dtype=np.float32)
    for j, model in enumerate(models):
        Y[:, j] = coerce_correct(df[f"{model}_correct"]).fillna(0).to_numpy(dtype=np.int8)
        C[:, j] = pd.to_numeric(df[f"{model}_cost"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)

    meta = df[["sample_id", "dataset", "task"]].copy()
    splits = build_splits(meta, args.train_ratio, args.dev_ratio, args.test_ratio, args.split_seed)
    split_of = {sid: split for split, ids in splits.items() for sid in ids}
    meta["split"] = meta["sample_id"].map(split_of).fillna("unknown")
    meta = meta[["sample_id", "dataset", "split", "task"]]

    tsv_row_maps = {}
    tsv_dir = args.output_dir / "vlm_router_data" / "TSV_images"
    if not args.no_image_tsv:
        print("Packing images into TSV files...")
        for dataset, group in df.groupby("dataset", sort=True):
            items = [(str(row.item_id), Path(row.img_path_resolved) if row.img_path_resolved else None)
                     for row in group.itertuples(index=False)]
            tsv_row_maps[dataset] = pack_dataset_tsv(dataset, items, tsv_dir)

    benchmark_rows_by_task_dataset = defaultdict(list)
    for row in df.itertuples(index=False):
        assets = []
        if args.no_image_tsv:
            if row.img_path_resolved:
                assets.append({"type": "image", "uri": row.img_path_resolved})
        else:
            rel_tsv = Path("vlm_router_data") / "TSV_images" / f"{row.dataset}.tsv"
            tsv_file = str((args.output_dir / rel_tsv).resolve())
            assets.append({
                "type": "image_tsv",
                "tsv_file": tsv_file,
                "index": int(tsv_row_maps[row.dataset][str(row.item_id)]),
                "description": f"Base64 encoded image from {rel_tsv.name} at row {tsv_row_maps[row.dataset][str(row.item_id)]}",
            })

        rec = {
            "sample_id": row.sample_id,
            "dataset": row.dataset,
            "task_type": row.task,
            "modality": ["image", "text"] if assets else ["text"],
            "prompt": "" if pd.isna(row.question) else str(row.question),
            "answer": "" if not hasattr(row, "answer") or pd.isna(row.answer) else str(row.answer),
            "assets": assets,
            "source_dataset_idx": row.dataset_idx,
        }
        benchmark_rows_by_task_dataset[(row.task, row.dataset)].append(rec)

    for (task, dataset), rows in benchmark_rows_by_task_dataset.items():
        out = args.output_dir / "BENCHMARKS" / task / f"{dataset.lower()}_samples.jsonl"
        write_jsonl(out, rows)

    splits_dir = args.output_dir / "SPLITS"
    for split, ids in splits.items():
        write_jsonl(splits_dir / f"{split}.jsonl", [{"sample_id": sid} for sid in ids])

    registry_dir = args.output_dir / "data" / "registry"
    matrices_dir = args.output_dir / "data" / "matrices"
    oracle_dir = args.output_dir / "ORACLE" / "score"
    registry_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir.mkdir(parents=True, exist_ok=True)

    meta.to_parquet(registry_dir / "meta.parquet", index=False)
    with (registry_dir / "sample_index.pkl").open("wb") as f:
        pickle.dump(sample_ids, f)
    with (registry_dir / "model_index.pkl").open("wb") as f:
        pickle.dump(models, f)
    (registry_dir / "models.txt").write_text("\n".join(models) + "\n", encoding="utf-8")
    (registry_dir / "models.json").write_text(
        json.dumps({"models": [{"name": model} for model in models]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    np.savez_compressed(matrices_dir / "Y.npz", Y=Y)
    np.save(matrices_dir / "C.npy", C)
    np.savez_compressed(matrices_dir / "C.npz", C=C)
    with (matrices_dir / "sample_ids.pkl").open("wb") as f:
        pickle.dump(sample_ids, f)
    with (matrices_dir / "model_names.pkl").open("wb") as f:
        pickle.dump(models, f)

    positive_costs = C[C > 0]
    cmin = float(positive_costs.min()) if len(positive_costs) else float(C.min())
    cmax = float(C.max())
    if cmax <= cmin:
        cmax = cmin + 1.0
    cost_bounds = {
        "cmin": cmin,
        "cmax": cmax,
        "source": "MMR_Bench.csv model cost columns",
        "cmin_model": models[int(np.nanargmin(C.mean(axis=0)))],
        "cmax_model": models[int(np.nanargmax(C.mean(axis=0)))],
    }
    (matrices_dir / "cost_bounds.json").write_text(json.dumps(cost_bounds, indent=2) + "\n", encoding="utf-8")

    cost_details = []
    for j, model in enumerate(models):
        cost_details.append({
            "model": model,
            "avg_cost": float(C[:, j].mean()),
            "min_cost": float(C[:, j].min()),
            "max_cost": float(C[:, j].max()),
            "accuracy": float(Y[:, j].mean()),
        })
    pd.DataFrame(cost_details).to_csv(matrices_dir / "cost_details.csv", index=False)

    score_records_by_dataset = defaultdict(list)
    for i, sid in enumerate(sample_ids):
        dataset = meta.loc[i, "dataset"]
        for j, model in enumerate(models):
            score_records_by_dataset[dataset].append({
                "sample_id": sid,
                "model_id": model,
                "dataset": dataset,
                "quality_raw": int(Y[i, j]),
                "quality": float(Y[i, j]),
                "cost": float(C[i, j]),
            })
    for dataset, records in score_records_by_dataset.items():
        pd.DataFrame(records).to_parquet(oracle_dir / f"{dataset.lower()}.parquet", index=False)

    manifest = {
        "source": str(csv_path),
        "num_samples": N,
        "models": models,
        "datasets": sorted(meta["dataset"].unique().tolist()),
        "tasks": sorted(meta["task"].unique().tolist()),
        "splits": {k: len(v) for k, v in splits.items()},
        "model_mean_accuracy": {m: float(Y[:, j].mean()) for j, m in enumerate(models)},
        "model_mean_cost": {m: float(C[:, j].mean()) for j, m in enumerate(models)},
        "cost_bounds": cost_bounds,
        "image_assets": "image_tsv" if not args.no_image_tsv else "image_uri",
    }
    (args.output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 80)
    print("MMR-Bench conversion complete")
    print("=" * 80)
    print(f"Output: {args.output_dir}")
    print(f"Samples: {N}")
    print(f"Models: {K}")
    print(f"Splits: {manifest['splits']}")
    print("By dataset:")
    print(meta["dataset"].value_counts().sort_index().to_string())
    print("Artifacts:")
    print(f"  {registry_dir / 'meta.parquet'}")
    print(f"  {matrices_dir / 'Y.npz'}")
    print(f"  {matrices_dir / 'C.npy'}")
    print(f"  {splits_dir}")
    print(f"  {args.output_dir / 'BENCHMARKS'}")


if __name__ == "__main__":
    main()
