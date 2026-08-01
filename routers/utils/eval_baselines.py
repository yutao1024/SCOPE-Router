#!/usr/bin/env python3
"""
Evaluate baseline routers and write a lightweight report.

This script is used by `scripts/run_step6_evaluate_baselines.sh`.

Baselines:
- StrongestGlobal
- StrongestPerDataset
- CheapestGlobal
- Oracle
- Random

Outputs (under --output_dir):
- summary.csv
- detailed_results.json
- <baseline>_samples.csv
- <baseline>_test_by_dataset.csv
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from routers.baselines.cheapest_global import CheapestGlobal
from routers.baselines.oracle import Oracle
from routers.baselines.random_router import RandomRouter
from routers.baselines.strongest_global import StrongestGlobal
from routers.baselines.strongest_per_dataset import StrongestPerDataset
from routers.utils.rank_score import get_cost_bounds_from_config, rank_score


def _load_npz_any(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    # common keys across this repo
    for k in ("Y", "C", "data", "arr_0"):
        if k in getattr(data, "files", []):
            arr = data[k]
            break
    else:
        # fallback to the first entry
        arr = data[data.files[0]]
    # Convert sparse-like to dense if needed
    try:
        import scipy.sparse as sp  # optional dependency

        if sp.issparse(arr):
            arr = arr.toarray()
    except Exception:
        pass
    return arr


def _load_cost_matrix(dataset_dir: Path) -> np.ndarray:
    c_npy = dataset_dir / "data/matrices/C.npy"
    c_npz = dataset_dir / "data/matrices/C.npz"
    if c_npy.exists():
        return np.load(c_npy)
    if c_npz.exists():
        return _load_npz_any(c_npz)
    raise FileNotFoundError(f"Missing cost matrix: {c_npy} (or {c_npz})")


def _load_models(dataset_dir: Path, K: int) -> List[str]:
    model_index_pkl = dataset_dir / "data/registry/model_index.pkl"
    if model_index_pkl.exists():
        try:
            with open(model_index_pkl, "rb") as f:
                obj = pickle.load(f)
            # common patterns: dict[name]->idx, list of names, dict with 'models'
            if isinstance(obj, dict):
                if "models" in obj and isinstance(obj["models"], list):
                    return list(obj["models"])
                # assume mapping name->idx
                if all(isinstance(v, (int, np.integer)) for v in obj.values()):
                    inv = {int(v): str(k) for k, v in obj.items()}
                    return [inv.get(i, f"model_{i}") for i in range(K)]
            if isinstance(obj, (list, tuple)) and all(isinstance(x, str) for x in obj):
                return list(obj)
        except Exception:
            pass
    # fallback
    return [f"model_{i}" for i in range(K)]


def _load_split_ids(dataset_dir: Path, split: str) -> List[Any]:
    split_file = dataset_dir / "SPLITS" / f"{split}.jsonl"
    if not split_file.exists():
        return []
    ids: List[Any] = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ids.append(rec.get("sample_id", rec.get("id")))
    return ids


def _subset_by_ids(meta: pd.DataFrame, Y: np.ndarray, C: np.ndarray, split_ids: List[Any]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not split_ids:
        return meta, Y, C
    mask = meta["sample_id"].isin(set(split_ids))
    idx = np.where(mask.values)[0]
    return meta.iloc[idx].copy(), Y[idx], C[idx]


def _evaluate_predictions(
    predictions: np.ndarray,
    Y: np.ndarray,
    C: np.ndarray,
    cmin: float,
    cmax: float,
    beta: float = 0.1,
) -> Dict[str, Any]:
    predictions = np.asarray(predictions).astype(int)
    correct = Y[np.arange(len(predictions)), predictions]
    costs = C[np.arange(len(predictions)), predictions]
    acc = float(correct.mean()) if len(correct) else 0.0
    avg_cost = float(costs.mean()) if len(costs) else 0.0
    rs = float(rank_score(acc, avg_cost, cmin, cmax, beta=beta)) if len(correct) else 0.0
    return {
        "accuracy": acc,
        "avg_cost": avg_cost,
        "rank_score": rs,
        "rank_score_beta": beta,
        "num_correct": int(correct.sum()),
        "num_samples": int(len(correct)),
    }


def _evaluate_by_dataset_from_predictions(
    meta: pd.DataFrame,
    predictions: np.ndarray,
    Y: np.ndarray,
    C: np.ndarray,
    cmin: float,
    cmax: float,
    beta: float = 0.1,
) -> pd.DataFrame:
    if "dataset" not in meta.columns:
        return pd.DataFrame()
    predictions = np.asarray(predictions).astype(int)
    rows = []
    for ds in sorted(meta["dataset"].unique()):
        mask = (meta["dataset"] == ds).values
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        preds = predictions[idx]
        Yd = Y[idx]
        Cd = C[idx]
        correct = Yd[np.arange(len(preds)), preds]
        costs = Cd[np.arange(len(preds)), preds]
        acc = float(correct.mean()) if len(correct) else 0.0
        avg_cost = float(costs.mean()) if len(costs) else 0.0
        rs = float(rank_score(acc, avg_cost, cmin, cmax, beta=beta)) if len(correct) else 0.0
        rows.append(
            {
                "dataset": ds,
                "accuracy": acc,
                "avg_cost": avg_cost,
                "rank_score": rs,
                "num_correct": int(correct.sum()),
                "num_samples": int(len(correct)),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("rank_score", ascending=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate baseline routers and save reports")
    p.add_argument("--dataset_dir", type=str, default=".", help="Dataset root (default: .)")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory for baseline reports")
    p.add_argument("--random_seed", type=int, default=42, help="Random seed for RandomRouter (default: 42)")
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load core data
    Y = _load_npz_any(dataset_dir / "data/matrices/Y.npz")
    C = _load_cost_matrix(dataset_dir)
    meta = pd.read_parquet(dataset_dir / "data/registry/meta.parquet")

    # Normalize types
    if "sample_id" not in meta.columns:
        raise ValueError("meta.parquet must include 'sample_id' column")

    K = int(Y.shape[1])
    models = _load_models(dataset_dir, K)

    # Cost bounds for rank_score
    cost_bounds_file = dataset_dir / "data/matrices/cost_bounds.json"
    if cost_bounds_file.exists():
        cmin, cmax = get_cost_bounds_from_config(str(cost_bounds_file))
    else:
        cmin = float(C.min())
        cmax = float(C.max())

    # Use test split if present
    test_ids = _load_split_ids(dataset_dir, "test")
    meta_test, Y_test, C_test = _subset_by_ids(meta, Y, C, test_ids)

    baselines = [
        ("strongest_global", StrongestGlobal()),
        ("strongest_per_dataset", StrongestPerDataset()),
        ("cheapest_global", CheapestGlobal()),
        ("oracle", Oracle()),
        ("random", RandomRouter(seed=int(args.random_seed))),
    ]

    detailed: Dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "num_samples_test": int(len(meta_test)),
        "num_models": int(K),
        "models": models,
        "cost_bounds": {"cmin": float(cmin), "cmax": float(cmax)},
        "baselines": {},
    }

    summary_rows = []

    for name, router in baselines:
        print(f"\n=== Evaluating baseline: {name} ===")
        router.fit(Y_test, C_test, meta_test)

        if name == "oracle":
            preds = router.predict(meta=meta_test, Y_test=Y_test, C_test=C_test)
        else:
            preds = router.predict(meta=meta_test)
        preds = np.asarray(preds).astype(int)

        overall = _evaluate_predictions(preds, Y_test, C_test, cmin=cmin, cmax=cmax, beta=0.1)
        summary_rows.append({"router": name, **overall})

        # Save per-sample CSV (test only)
        correct = Y_test[np.arange(len(preds)), preds]
        costs = C_test[np.arange(len(preds)), preds]
        samples_df = meta_test.copy()
        samples_df["pred_model_idx"] = preds
        samples_df["pred_model_name"] = [models[i] if 0 <= i < len(models) else f"model_{i}" for i in preds]
        samples_df["correct"] = correct.astype(int)
        samples_df["cost"] = costs.astype(float)
        samples_path = out_dir / f"{name}_samples.csv"
        samples_df.to_csv(samples_path, index=False)

        # Save per-dataset CSV (test only)
        by_ds = _evaluate_by_dataset_from_predictions(meta_test, preds, Y_test, C_test, cmin=cmin, cmax=cmax, beta=0.1)
        by_ds_path = None
        if not by_ds.empty:
            by_ds_path = out_dir / f"{name}_test_by_dataset.csv"
            by_ds.to_csv(by_ds_path, index=False, float_format="%.6f")

        detailed["baselines"][name] = {
            "overall_test": overall,
            "samples_csv": str(samples_path),
            "test_by_dataset_csv": str(by_ds_path) if by_ds_path is not None else None,
        }

    # Write summary + detailed json
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False, float_format="%.6f")
    with open(out_dir / "detailed_results.json", "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Wrote: {out_dir / 'summary.csv'}")
    print(f"✓ Wrote: {out_dir / 'detailed_results.json'}")


if __name__ == "__main__":
    main()


