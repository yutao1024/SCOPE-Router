#!/usr/bin/env python3
"""Export SWE-bench official harness predictions from real-repo smoke rows."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value).strip("_") or "unknown"


def task_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("task_id")): row for row in read_jsonl(path)}


def verifier_reference(task: dict[str, Any]) -> dict[str, Any]:
    verifier = task.get("verifier") if isinstance(task.get("verifier"), dict) else {}
    reference = verifier.get("reference") if isinstance(verifier.get("reference"), dict) else {}
    return reference or {}


def raw_row_from_source_ref(task: dict[str, Any]) -> dict[str, Any]:
    source_ref = task.get("source_ref") if isinstance(task.get("source_ref"), dict) else {}
    raw_path = source_ref.get("raw_path")
    raw_line = source_ref.get("raw_line")
    if not raw_path or not isinstance(raw_line, int):
        return {}
    path = Path(str(raw_path))
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if index == raw_line:
                return json.loads(line)
    return {}


def instance_id_for_task(task: dict[str, Any]) -> str:
    reference = verifier_reference(task)
    for value in (
        reference.get("instance_id"),
        task.get("instance_id"),
        task.get("source_id"),
    ):
        if value:
            return str(value)
    raw = raw_row_from_source_ref(task)
    for key in ("instance_id", "id"):
        if raw.get(key):
            return str(raw[key])
    raise RuntimeError(f"Cannot resolve SWE-bench instance_id for task {task.get('task_id')!r}")


def source_dataset_for_task(task: dict[str, Any]) -> str:
    return str(task.get("source_dataset") or "")


def real_repo_meta(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("openclaw")
    return meta if isinstance(meta, dict) else {}


def extract_model_patch(row: dict[str, Any]) -> str:
    text = str(row.get("assistant_text") or "")
    match = re.search(r"```diff\s*\n(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip() + "\n"
    if text.lstrip().startswith("diff --git "):
        return text.strip() + "\n"
    raise RuntimeError(f"Cannot extract diff patch for task={row.get('task_id')} model={row.get('candidate_model')}")


def selected_rows(
    rows: list[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    *,
    source_datasets: set[str],
    statuses: set[str],
    models: set[str],
    task_ids: set[str],
    require_git_diff: bool,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        task_id = str(row.get("task_id"))
        task = tasks.get(task_id)
        if not task:
            continue
        if source_datasets and source_dataset_for_task(task) not in source_datasets:
            continue
        if statuses and str(row.get("status")) not in statuses:
            continue
        if models and str(row.get("candidate_model")) not in models:
            continue
        if task_ids and task_id not in task_ids:
            continue
        meta = real_repo_meta(row)
        if require_git_diff and not meta.get("git_diff_present"):
            continue
        output.append(row)
    return output


def selection_diagnostics(
    rows: list[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    *,
    source_datasets: set[str],
    statuses: set[str],
    models: set[str],
    task_ids: set[str],
    require_git_diff: bool,
) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        reason = "selected"
        task_id = str(row.get("task_id"))
        task = tasks.get(task_id)
        if not task:
            reason = "missing_task"
        elif source_datasets and source_dataset_for_task(task) not in source_datasets:
            reason = "source_dataset_filter"
        elif statuses and str(row.get("status")) not in statuses:
            reason = "status_filter"
        elif models and str(row.get("candidate_model")) not in models:
            reason = "candidate_model_filter"
        elif task_ids and task_id not in task_ids:
            reason = "task_id_filter"
        elif require_git_diff and not real_repo_meta(row).get("git_diff_present"):
            reason = "missing_git_diff"
        counters[reason] += 1
        examples.setdefault(reason, [])
        if len(examples[reason]) < 5:
            examples[reason].append(
                {
                    "task_id": row.get("task_id"),
                    "candidate_model": row.get("candidate_model"),
                    "status": row.get("status"),
                    "source_dataset": source_dataset_for_task(task) if task else row.get("source_dataset"),
                    "git_diff_present": bool(real_repo_meta(row).get("git_diff_present")),
                    "git_diff_bytes": real_repo_meta(row).get("git_diff_bytes"),
                    "verify_reason": row.get("verify_reason"),
                }
            )
    return {
        "counts": dict(sorted(counters.items())),
        "examples": examples,
        "filters": {
            "source_datasets": sorted(source_datasets),
            "statuses": sorted(statuses),
            "models": sorted(models),
            "task_ids": sorted(task_ids),
            "require_git_diff": require_git_diff,
        },
    }


def prediction_for_row(row: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    model = str(row.get("candidate_model") or "unknown-model")
    return {
        "instance_id": instance_id_for_task(task),
        "model_name_or_path": model,
        "model_patch": extract_model_patch(row),
    }


def status_rank(row: dict[str, Any]) -> tuple[int, int]:
    status = str(row.get("status"))
    passed = row.get("passed") is True
    rank = {
        "ok": 0,
        "model_wrong": 1,
        "verifier_error": 2,
        "error": 3,
    }.get(status, 9)
    return rank, 0 if passed else 1


def duplicate_instance_ids(rows: list[dict[str, Any]], tasks: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        task = tasks[str(row.get("task_id"))]
        grouped.setdefault(instance_id_for_task(task), []).append(row)
    return {instance_id: items for instance_id, items in grouped.items() if len(items) > 1}


def enforce_unique_instances(
    rows: list[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    duplicates = duplicate_instance_ids(rows, tasks)
    if not duplicates:
        return rows
    if mode == "error":
        details = []
        for instance_id, items in sorted(duplicates.items()):
            models = ", ".join(str(item.get("candidate_model")) for item in items)
            details.append(f"{instance_id}: {models}")
        raise SystemExit(
            "selected rows contain duplicate SWE-bench instance_id values; official harness "
            "expects one prediction per instance per run. Use --candidate-model to export one "
            "model at a time, or pass --dedupe-instance first/best intentionally. Duplicates: "
            + "; ".join(details[:20])
        )
    if mode not in {"first", "best"}:
        raise ValueError(f"unknown duplicate mode: {mode}")

    selected_by_instance: dict[str, dict[str, Any]] = {}
    for row in rows:
        task = tasks[str(row.get("task_id"))]
        instance_id = instance_id_for_task(task)
        current = selected_by_instance.get(instance_id)
        if current is None:
            selected_by_instance[instance_id] = row
            continue
        if mode == "best" and status_rank(row) < status_rank(current):
            selected_by_instance[instance_id] = row
    return list(selected_by_instance.values())


def rows_by_model(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("candidate_model") or "unknown-model"), []).append(row)
    return dict(sorted(grouped.items()))


def summary_for(rows: list[dict[str, Any]], tasks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_source = Counter(source_dataset_for_task(tasks.get(str(row.get("task_id")), {})) for row in rows)
    by_status = Counter(str(row.get("status")) for row in rows)
    by_model = Counter(str(row.get("candidate_model")) for row in rows)
    by_task = Counter(str(row.get("task_id")) for row in rows)
    instances = [instance_id_for_task(tasks[str(row.get("task_id"))]) for row in rows]
    by_instance = Counter(instances)
    return {
        "rows": len(rows),
        "unique_instances": len(by_instance),
        "duplicate_instance_ids": {
            instance_id: count for instance_id, count in sorted(by_instance.items()) if count > 1
        },
        "by_source_dataset": dict(sorted(by_source.items())),
        "by_status": dict(sorted(by_status.items())),
        "by_model": dict(sorted(by_model.items())),
        "by_task": dict(sorted(by_task.items())),
        "items": [
            {
                "task_id": row.get("task_id"),
                "instance_id": instance_id_for_task(tasks[str(row.get("task_id"))]),
                "candidate_model": row.get("candidate_model"),
                "status": row.get("status"),
                "source_dataset": source_dataset_for_task(tasks[str(row.get("task_id"))]),
                "git_diff_bytes": real_repo_meta(row).get("git_diff_bytes"),
                "verify_reason": row.get("verify_reason"),
            }
            for row in rows
        ],
    }


def write_prediction_file(
    *,
    rows: list[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    output: Path,
    summary_out: Path | None,
) -> dict[str, Any]:
    predictions = [prediction_for_row(row, tasks[str(row.get("task_id"))]) for row in rows]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in predictions),
        encoding="utf-8",
    )

    summary_path = summary_out or output.with_name(output.stem + "_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summary_for(rows, tasks)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "output": str(output),
        "summary": str(summary_path),
        "rows": len(predictions),
        "unique_instances": summary["unique_instances"],
        "by_model": summary["by_model"],
        "by_status": summary["by_status"],
    }


def split_output_path(base_output: Path, model: str) -> Path:
    suffix = base_output.suffix or ".jsonl"
    stem = base_output.stem if base_output.suffix else base_output.name
    return base_output.with_name(f"{stem}__{safe_name(model)}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--source-dataset", action="append", default=[])
    parser.add_argument("--status", action="append", default=None)
    parser.add_argument("--candidate-model", action="append", default=[])
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-missing-diff", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument(
        "--split-by-model",
        action="store_true",
        help=(
            "Write one official predictions file per candidate_model. This is the recommended "
            "mode for comparing multiple models on the same SWE-bench instances."
        ),
    )
    parser.add_argument(
        "--dedupe-instance",
        choices=["error", "first", "best"],
        default="error",
        help=(
            "How to handle multiple selected rows for the same SWE-bench instance_id. "
            "Default errors because official SWE-bench predictions should be one row per instance."
        ),
    )
    args = parser.parse_args()
    if args.output is None:
        raise SystemExit("--output is required.")
    if args.split_by_model and args.summary_out is not None:
        raise SystemExit("--summary-out is only supported for single-file export; split mode writes per-file summaries.")

    tasks = task_by_id(args.tasks)
    all_rows = read_jsonl(args.results)
    source_datasets = set(args.source_dataset)
    statuses = set(args.status or ["ok", "model_wrong", "needs_official"])
    models = set(args.candidate_model)
    task_ids = set(args.task_id)
    require_git_diff = not args.allow_missing_diff
    rows = selected_rows(
        all_rows,
        tasks,
        source_datasets=source_datasets,
        statuses=statuses,
        models=models,
        task_ids=task_ids,
        require_git_diff=require_git_diff,
    )
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows and not args.allow_empty:
        diagnostics = selection_diagnostics(
            all_rows,
            tasks,
            source_datasets=source_datasets,
            statuses=statuses,
            models=models,
            task_ids=task_ids,
            require_git_diff=require_git_diff,
        )
        raise SystemExit(
            "no SWE-bench official predictions selected; check --results/--tasks filters, "
            "or pass --allow-empty intentionally. diagnostics="
            + json.dumps(diagnostics, ensure_ascii=False, sort_keys=True)
        )

    if args.split_by_model:
        written = []
        for model, model_rows in rows_by_model(rows).items():
            model_rows = enforce_unique_instances(model_rows, tasks, mode=args.dedupe_instance)
            written.append(
                write_prediction_file(
                    rows=model_rows,
                    tasks=tasks,
                    output=split_output_path(args.output, model),
                    summary_out=None,
                )
            )
        index_path = args.output.with_name(args.output.stem + "_index.json")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps({"files": written}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {len(written)} per-model prediction files")
        print(f"wrote index to {index_path}")
        for item in written:
            print(f"  {item['rows']} rows -> {item['output']}")
        return

    rows = enforce_unique_instances(rows, tasks, mode=args.dedupe_instance)
    written = write_prediction_file(rows=rows, tasks=tasks, output=args.output, summary_out=args.summary_out)
    print(f"wrote {written['rows']} predictions to {written['output']}")
    print(f"wrote summary to {written['summary']}")


if __name__ == "__main__":
    main()
