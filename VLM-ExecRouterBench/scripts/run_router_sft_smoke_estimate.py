#!/usr/bin/env python3
"""One-command smoke estimate runner for router SFT executor batches.

This script orchestrates multiple generate_router_sft.py invocations because
ordinary OpenClaw tasks and SWE-bench real-repo tasks require different
executor wrappers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SMOKE_TASKS = "manifests/executor_task_pool_smoke_2_per_source.jsonl"
SMOKE_SUMMARY = "manifests/executor_task_pool_smoke_2_per_source_summary.json"
SWEBENCH_MULTIMODAL_DATASET = "princeton-nlp/SWE-bench_Multimodal"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def run(cmd: list[str], *, dry_run: bool = False) -> None:
    print("[run] " + " ".join(cmd), flush=True)
    if dry_run:
        return
    result = subprocess.run(cmd, cwd=ROOT, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def source_dataset(row: dict[str, Any]) -> str:
    return str(row.get("source_dataset") or "unknown")


def source_is_swebench(source: str) -> bool:
    normalized = source.lower()
    return "swe-bench" in normalized or "swebench" in normalized


def manifest_sources(path: Path) -> tuple[list[str], list[str]]:
    ordinary = set()
    swebench = set()
    for row in read_jsonl(path) or []:
        source = source_dataset(row)
        if source_is_swebench(source):
            swebench.add(source)
        else:
            ordinary.add(source)
    return sorted(ordinary), sorted(swebench)


def append_candidate_models(cmd: list[str], candidate_models: list[str] | None) -> None:
    for model in candidate_models or []:
        cmd.extend(["--candidate-model", model])


def append_common_generate_args(cmd: list[str], args: argparse.Namespace, *, stage_dir: Path) -> None:
    cmd.extend(
        [
            "--tasks",
            args.tasks,
            "--results-out",
            str(args.results_out),
            "--sft-out",
            str(stage_dir / "router_sft.jsonl"),
            "--summary-out",
            str(stage_dir / "summary.json"),
            "--budget-policy",
            args.budget_policy,
            "--executor-backend",
            args.executor_backend,
            "--max-tokens",
            str(args.max_tokens),
            "--timeout",
            str(args.timeout),
            "--retries",
            str(args.retries),
            "--retry-sleep",
            str(args.retry_sleep),
            "--http-transport",
            args.http_transport,
            "--deepseek-thinking",
            args.deepseek_thinking,
            "--target-sft-rows",
            str(args.target_sft_rows),
        ]
    )
    cmd.append("--run-all" if args.run_all else "--no-run-all")
    if args.skip_errors:
        cmd.append("--skip-errors")
    if args.rerun_failed:
        cmd.append("--rerun-failed")
    if args.dry_run_generate:
        cmd.append("--dry-run")
    append_candidate_models(cmd, args.candidate_models)


def generate_cmd(args: argparse.Namespace, *, stage_dir: Path, sources: list[str], openclaw_command: str) -> list[str]:
    cmd = [sys.executable, "scripts/generate_router_sft.py"]
    append_common_generate_args(cmd, args, stage_dir=stage_dir)
    for source in sources:
        cmd.extend(["--source-dataset", source])
    if args.executor_backend == "openclaw":
        cmd.extend(["--openclaw-command", openclaw_command])
        if args.openclaw_model_ref_map:
            cmd.extend(["--openclaw-model-ref-map", args.openclaw_model_ref_map])
        if args.openclaw_keep_io:
            cmd.append("--openclaw-keep-io")
    return cmd


def final_build_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/generate_router_sft.py",
        "--build-sft-only",
        "--tasks",
        args.tasks,
        "--results-out",
        str(args.results_out),
        "--sft-out",
        str(args.sft_out),
        "--summary-out",
        str(args.summary_out),
        "--per-dataset-output-dir",
        str(args.per_dataset_output_dir),
        "--budget-policy",
        args.budget_policy,
        "--executor-backend",
        args.executor_backend,
        "--target-sft-rows",
        str(args.target_sft_rows),
    ]
    if args.skip_errors:
        cmd.append("--skip-errors")
    append_candidate_models(cmd, args.candidate_models)
    return cmd


def resample_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/select_executor_tasks.py",
        "--profile",
        "smoke",
        "--debug-per-source",
        str(args.per_source),
        "--include-default-excluded-source-datasets",
        "--out",
        args.tasks,
        "--summary-out",
        args.tasks_summary,
        "--seed",
        str(args.seed),
    ]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=SMOKE_TASKS)
    parser.add_argument("--tasks-summary", default=SMOKE_SUMMARY)
    parser.add_argument("--per-source", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--force-resample", action="store_true")
    parser.add_argument("--out-root", type=Path, default=Path("outputs/smoke_2_per_source"))
    parser.add_argument("--results-out", type=Path, default=None)
    parser.add_argument("--sft-out", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--per-dataset-output-dir", type=Path, default=None)
    parser.add_argument("--candidate-model", action="append", dest="candidate_models", default=None)
    parser.add_argument("--budget-policy", default="full")
    parser.add_argument("--executor-backend", choices=["raw_api", "openclaw"], default="openclaw")
    parser.add_argument(
        "--ordinary-openclaw-command",
        default="python3 scripts/run_openclaw_executor.py --input {input} --output {output}",
    )
    parser.add_argument(
        "--swebench-openclaw-command",
        default="python3 scripts/run_swebench_openclaw_executor.py --input {input} --output {output}",
    )
    parser.add_argument("--openclaw-model-ref-map", default="")
    parser.add_argument("--openclaw-keep-io", action="store_true")
    parser.add_argument("--target-sft-rows", type=int, default=1500)
    parser.add_argument(
        "--run-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run every eligible candidate model per task; use --no-run-all for first-pass cascade mode.",
    )
    parser.add_argument("--skip-errors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--http-transport", choices=["curl", "urllib"], default="curl")
    parser.add_argument("--deepseek-thinking", choices=["disabled", "enabled"], default="disabled")
    parser.add_argument(
        "--swebench-official-verify",
        action="store_true",
        help="Run official verification/export stages for SWE-bench sources after generation.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print orchestrated commands without running them.")
    parser.add_argument(
        "--dry-run-generate",
        action="store_true",
        help="Pass --dry-run to generate_router_sft.py stages while still building summaries.",
    )
    args = parser.parse_args()

    args.results_out = args.results_out or args.out_root / "executor_results.jsonl"
    args.sft_out = args.sft_out or args.out_root / "router_sft.jsonl"
    args.summary_out = args.summary_out or args.out_root / "summary.json"
    args.per_dataset_output_dir = args.per_dataset_output_dir or args.out_root / "by_dataset"

    task_path = ROOT / args.tasks
    if args.force_resample or not task_path.exists():
        run(resample_cmd(args), dry_run=args.dry_run)

    ordinary_sources, swebench_sources = manifest_sources(task_path)
    stages_dir = args.out_root / "stages"
    print(
        json.dumps(
            {
                "ordinary_sources": ordinary_sources,
                "swebench_sources": swebench_sources,
                "results_out": str(args.results_out),
                "summary_out": str(args.summary_out),
                "per_dataset_output_dir": str(args.per_dataset_output_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    if ordinary_sources:
        run(
            generate_cmd(
                args,
                stage_dir=stages_dir / "ordinary",
                sources=ordinary_sources,
                openclaw_command=args.ordinary_openclaw_command,
            ),
            dry_run=args.dry_run,
        )

    for source in swebench_sources:
        stage_dir = stages_dir / safe_stage_name(source)
        cmd = generate_cmd(
            args,
            stage_dir=stage_dir,
            sources=[source],
            openclaw_command=args.swebench_openclaw_command,
        )
        if args.swebench_official_verify:
            cmd.append("--swebench-official-verify")
            cmd.extend(["--swebench-official-dataset", source])
            cmd.extend(["--swebench-official-source-dataset", source])
            cmd.extend(["--swebench-official-results-out", str(stage_dir / "official_results.json")])
            cmd.extend(["--swebench-official-predictions-out", str(stage_dir / "official_predictions.jsonl")])
        run(cmd, dry_run=args.dry_run)

    run(final_build_cmd(args), dry_run=args.dry_run)


def safe_stage_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "__" for char in value).strip("_") or "stage"


if __name__ == "__main__":
    main()
