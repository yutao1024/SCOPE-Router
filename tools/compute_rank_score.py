#!/usr/bin/env python3
"""Compute VL-RouterBench Rank Score from average accuracy and cost.

Defaults match table units:
  --avg-acc is percent, e.g. 79.11
  --avg-cost is dollars per 10K samples, e.g. 0.98
  output rank_score is percent, e.g. 76.18
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routers.utils.rank_score import get_cost_bounds_from_config, rank_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Rank Score from Avg. Acc. and Avg. Cost")
    parser.add_argument("--dataset-dir", default=".", help="VL-RouterBench dataset directory")
    parser.add_argument("--avg-acc", type=float, default=None, help="Average accuracy. Default unit: percent.")
    parser.add_argument("--avg-cost", type=float, default=None, help="Average cost. Default unit: $/10K samples.")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--acc-unit", choices=["percent", "fraction"], default="percent")
    parser.add_argument("--cost-unit", choices=["per_10k", "per_sample"], default="per_10k")
    parser.add_argument("--score-unit", choices=["percent", "fraction"], default="percent")
    parser.add_argument("--input-csv", default=None, help="Optional CSV with avg_acc and avg_cost columns")
    parser.add_argument("--output-csv", default=None, help="Optional output CSV path for batch mode")
    parser.add_argument("--acc-col", default="avg_acc")
    parser.add_argument("--cost-col", default="avg_cost")
    return parser.parse_args()


def normalize_acc(value: float, unit: str) -> float:
    return value / 100.0 if unit == "percent" else value


def normalize_cost(value: float, unit: str) -> float:
    return value / 10000.0 if unit == "per_10k" else value


def format_score(value: float, unit: str) -> float:
    return value * 100.0 if unit == "percent" else value


def compute_one(avg_acc: float, avg_cost: float, args: argparse.Namespace, cmin: float, cmax: float) -> Dict[str, float]:
    acc = normalize_acc(avg_acc, args.acc_unit)
    cost = normalize_cost(avg_cost, args.cost_unit)
    score = rank_score(acc, cost, cmin, cmax, beta=args.beta)
    return {
        "avg_acc_input": avg_acc,
        "avg_cost_input": avg_cost,
        "avg_acc_fraction": acc,
        "avg_cost_per_sample": cost,
        "rank_score": format_score(score, args.score_unit),
    }


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    bounds_path = Path(args.dataset_dir) / "data" / "matrices" / "cost_bounds.json"
    cmin, cmax = get_cost_bounds_from_config(bounds_path)

    if args.input_csv:
        rows = read_rows(Path(args.input_csv))
        output_rows = []
        for row in rows:
            avg_acc = float(row[args.acc_col])
            avg_cost = float(row[args.cost_col])
            metrics = compute_one(avg_acc, avg_cost, args, cmin, cmax)
            out_row = dict(row)
            out_row["rank_score"] = f"{metrics['rank_score']:.6f}"
            out_row["avg_acc_fraction"] = f"{metrics['avg_acc_fraction']:.12g}"
            out_row["avg_cost_per_sample"] = f"{metrics['avg_cost_per_sample']:.12g}"
            output_rows.append(out_row)
        if args.output_csv:
            write_rows(Path(args.output_csv), output_rows)
            print(f"[save] {args.output_csv}")
        else:
            writer = csv.DictWriter(sys.stdout, fieldnames=list(output_rows[0].keys()))
            writer.writeheader()
            writer.writerows(output_rows)
        return

    if args.avg_acc is None or args.avg_cost is None:
        raise SystemExit("Provide --avg-acc and --avg-cost, or use --input-csv.")

    metrics = compute_one(args.avg_acc, args.avg_cost, args, cmin, cmax)
    print(f"cmin={cmin:.12g}")
    print(f"cmax={cmax:.12g}")
    print(f"avg_acc_fraction={metrics['avg_acc_fraction']:.12g}")
    print(f"avg_cost_per_sample={metrics['avg_cost_per_sample']:.12g}")
    print(f"rank_score={metrics['rank_score']:.6f}")


if __name__ == "__main__":
    main()
