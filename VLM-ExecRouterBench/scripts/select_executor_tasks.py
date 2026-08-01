import argparse
import json
import random
from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "openclaw_tasks"

CATEGORY_FILES = {
    "code_debug_edit": TASK_DIR / "code_debug_edit" / "tasks.jsonl",
    "tool_workflow": TASK_DIR / "tool_workflow" / "tasks.jsonl",
    "multimodal_doc_visual": TASK_DIR / "multimodal_doc_visual" / "tasks.jsonl",
}

DIFFICULTIES = ("easy", "medium", "hard")

PROFILES = {
    # Good first-pass executor run: cheap enough, but tilted toward harder tasks
    # so mid/high-end model labels have a chance to appear.
    "v1_seed": {"easy": 0.25, "medium": 0.35, "hard": 0.40},
    # Use when you only want a quick plumbing check.
    "smoke": {"easy": 0.40, "medium": 0.40, "hard": 0.20},
    # Use if early runs overproduce qwen3-vl-8b-instruct labels.
    "hard_tilt": {"easy": 0.15, "medium": 0.30, "hard": 0.55},
    # Use if early runs underproduce cheap-model labels.
    "balanced": {"easy": 1 / 3, "medium": 1 / 3, "hard": 1 / 3},
}

CANDIDATE_MODELS = (
    "qwen3-vl-8b-instruct",
    "qwen3.5-35b-a3b",
    "gpt-5.4-mini",
    "gpt-5.4",
)

DEFAULT_EXCLUDED_SOURCE_DATASETS = {"tuandunghcmut/toolbench-v1"}
DEFAULT_OUT = "manifests/executor_task_pool_v1_seed.jsonl"
DEFAULT_SUMMARY_OUT = "manifests/executor_task_pool_v1_seed_summary.json"
DEBUG_OUT = "manifests/executor_task_pool_debug_per_source.jsonl"
DEBUG_SUMMARY_OUT = "manifests/executor_task_pool_debug_per_source_summary.json"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_task_ids(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {str(row.get("task_id")) for row in read_jsonl(path) if row.get("task_id")}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def instruction_text(task: dict[str, Any]) -> str:
    return str(task.get("router_view", {}).get("instruction", ""))


def cost_proxy(task: dict[str, Any]) -> int:
    """Cheap deterministic proxy for API spend; not a tokenizer."""
    text_chars = len(json.dumps(task.get("executor_input", {}).get("messages", []), ensure_ascii=False))
    tools_chars = len(json.dumps(task.get("executor_input", {}).get("tools", []), ensure_ascii=False))
    image_penalty = 1500 if task.get("router_view", {}).get("metadata", {}).get("has_images") else 0
    return text_chars + tools_chars + image_penalty


def keep_task(task: dict[str, Any], max_instruction_chars: int, max_cost_proxy: int, max_tools: int) -> bool:
    if len(instruction_text(task)) > max_instruction_chars:
        return False
    if cost_proxy(task) > max_cost_proxy:
        return False
    metadata = task.get("router_view", {}).get("metadata", {})
    if int(metadata.get("tool_count") or 0) > max_tools:
        return False
    return True


def source_dataset(task: dict[str, Any]) -> str:
    return str(task.get("source_dataset") or "")


def quotas(total: int, weights: dict[str, float]) -> dict[str, int]:
    raw = {difficulty: total * weights[difficulty] for difficulty in DIFFICULTIES}
    result = {difficulty: int(raw[difficulty]) for difficulty in DIFFICULTIES}
    remainder = total - sum(result.values())
    order = sorted(DIFFICULTIES, key=lambda d: raw[d] - result[d], reverse=True)
    for difficulty in order[:remainder]:
        result[difficulty] += 1
    return result


def grouped(tasks: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        buckets[str(task.get(key))].append(task)
    return buckets


def round_robin_take(groups: dict[str, list[dict[str, Any]]], target: int, rng: random.Random) -> list[dict[str, Any]]:
    queues = []
    for rows in groups.values():
        rng.shuffle(rows)
        queues.append(rows)
    rng.shuffle(queues)

    selected = []
    while queues and len(selected) < target:
        next_queues = []
        for queue in queues:
            if len(selected) >= target:
                break
            if queue:
                selected.append(queue.pop(0))
            if queue:
                next_queues.append(queue)
        queues = next_queues
    return selected


def select_category(
    tasks: list[dict[str, Any]],
    total: int,
    weights: dict[str, float],
    rng: random.Random,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_diff = grouped(tasks, "difficulty_prior")
    diff_quotas = quotas(total, weights)

    for difficulty in DIFFICULTIES:
        rows = by_diff.get(difficulty, [])
        by_source = grouped(rows, "source_dataset")
        selected.extend(round_robin_take(by_source, diff_quotas[difficulty], rng))

    if len(selected) < total:
        selected_ids = {task["task_id"] for task in selected}
        leftovers = [task for task in tasks if task["task_id"] not in selected_ids]
        selected.extend(round_robin_take(grouped(leftovers, "source_dataset"), total - len(selected), rng))

    rng.shuffle(selected)
    return selected[:total]


def parse_source_limits(values: list[str]) -> dict[str, int]:
    limits = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--debug-source-limit must use SOURCE=COUNT, got {value!r}")
        source, count_text = value.rsplit("=", 1)
        source = source.strip()
        if not source:
            raise ValueError(f"--debug-source-limit source is empty: {value!r}")
        count = int(count_text)
        if count < 0:
            raise ValueError(f"--debug-source-limit count must be non-negative: {value!r}")
        limits[source] = count
    return limits


def select_debug_per_source(
    tasks: list[dict[str, Any]],
    per_source: int,
    rng: random.Random,
    source_limits: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    selected = []
    source_limits = source_limits or {}
    for source, rows in grouped(tasks, "source_dataset").items():
        rng.shuffle(rows)
        selected.extend(rows[: source_limits.get(source, per_source)])
    rng.shuffle(selected)
    return selected


def annotate(task: dict[str, Any], profile: str, wave: str) -> dict[str, Any]:
    item = dict(task)
    item["_selection"] = {
        "profile": profile,
        "wave": wave,
        "cost_proxy": cost_proxy(task),
    }
    return item


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, int] = defaultdict(int)
    by_category_difficulty: dict[str, int] = defaultdict(int)
    by_category_source: dict[str, int] = defaultdict(int)
    proxies = []

    for row in rows:
        category = str(row.get("category"))
        difficulty = str(row.get("difficulty_prior"))
        source = str(row.get("source_dataset"))
        by_category[category] += 1
        by_category_difficulty[f"{category}::{difficulty}"] += 1
        by_category_source[f"{category}::{source}"] += 1
        proxies.append(int(row.get("_selection", {}).get("cost_proxy", cost_proxy(row))))

    proxies.sort()
    if proxies:
        proxy_summary = {
            "min": proxies[0],
            "p50": proxies[len(proxies) // 2],
            "p95": proxies[int(len(proxies) * 0.95)],
            "max": proxies[-1],
        }
    else:
        proxy_summary = {"min": 0, "p50": 0, "p95": 0, "max": 0}

    return {
        "total": len(rows),
        "cost_proxy": proxy_summary,
        "by_category": dict(sorted(by_category.items())),
        "by_category_difficulty": dict(sorted(by_category_difficulty.items())),
        "by_category_source": dict(sorted(by_category_source.items())),
    }


def derive_per_category(target_sft_rows: int, expected_yield: float) -> int:
    if not 0 < expected_yield <= 1:
        raise ValueError("--expected-yield must be in (0, 1].")
    return max(1, ceil(target_sft_rows / expected_yield / len(CATEGORY_FILES)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a budget-aware executor task batch from openclaw_tasks.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="v1_seed")
    parser.add_argument("--per-category", type=int, default=None)
    parser.add_argument("--target-sft-rows", type=int, default=1500)
    parser.add_argument("--expected-yield", type=float, default=0.65)
    parser.add_argument(
        "--debug-per-source",
        type=int,
        default=None,
        help="If set, sample this many tasks from every source dataset in every category.",
    )
    parser.add_argument(
        "--debug-source-limit",
        action="append",
        default=[],
        help=(
            "Override --debug-per-source for one source dataset, as SOURCE=COUNT. "
            "Can be repeated; only applies with --debug-per-source."
        ),
    )
    parser.add_argument("--candidate-model", action="append", dest="candidate_models", default=None)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--wave", default="wave1")
    parser.add_argument(
        "--exclude-task-file",
        action="append",
        default=[],
        help="JSONL task pool whose task_id values should be excluded. Can be passed multiple times.",
    )
    parser.add_argument("--max-instruction-chars", type=int, default=6000)
    parser.add_argument("--max-cost-proxy", type=int, default=30000)
    parser.add_argument("--max-tools", type=int, default=12)
    parser.add_argument(
        "--exclude-source-dataset",
        action="append",
        default=sorted(DEFAULT_EXCLUDED_SOURCE_DATASETS),
        help="Source dataset to exclude from the selected task pool. Can be passed multiple times.",
    )
    parser.add_argument(
        "--include-default-excluded-source-datasets",
        action="store_true",
        help="Do not apply the script's built-in source dataset exclusions.",
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--summary-out", default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    weights = PROFILES[args.profile]
    selected: list[dict[str, Any]] = []
    candidate_models = args.candidate_models or list(CANDIDATE_MODELS)
    per_category = args.per_category or derive_per_category(args.target_sft_rows, args.expected_yield)
    debug_source_limits = parse_source_limits(args.debug_source_limit)
    if args.debug_per_source is None and debug_source_limits:
        raise ValueError("--debug-source-limit requires --debug-per-source.")
    excluded_source_datasets = set(args.exclude_source_dataset or [])
    if args.include_default_excluded_source_datasets:
        excluded_source_datasets -= DEFAULT_EXCLUDED_SOURCE_DATASETS
    excluded_task_ids: set[str] = set()
    for exclude_path in args.exclude_task_file:
        path = Path(exclude_path)
        if not path.is_absolute():
            path = ROOT / path
        excluded_task_ids.update(read_task_ids(path))

    if args.debug_per_source is not None:
        if args.debug_per_source <= 0:
            raise ValueError("--debug-per-source must be positive.")
        if args.out == DEFAULT_OUT:
            args.out = DEBUG_OUT
        if args.summary_out == DEFAULT_SUMMARY_OUT:
            args.summary_out = DEBUG_SUMMARY_OUT

    for category, path in CATEGORY_FILES.items():
        tasks = [
            task
            for task in read_jsonl(path)
            if keep_task(
                task,
                max_instruction_chars=args.max_instruction_chars,
                max_cost_proxy=args.max_cost_proxy,
                max_tools=args.max_tools,
            )
            and source_dataset(task) not in excluded_source_datasets
            and str(task.get("task_id")) not in excluded_task_ids
        ]
        if args.debug_per_source is None:
            category_selected = select_category(tasks, per_category, weights, rng)
        else:
            category_selected = select_debug_per_source(tasks, args.debug_per_source, rng, debug_source_limits)
        selected.extend(annotate(task, args.profile, args.wave) for task in category_selected)
        print(f"[{category}] available={len(tasks)} selected={len(category_selected)}")

    summary = summarize(selected)
    summary["task_pool"] = {
        "unique_task_rows": len(selected),
        "per_category_requested": per_category,
        "target_sft_rows": args.target_sft_rows,
        "expected_yield": args.expected_yield,
        "debug_per_source": args.debug_per_source,
        "debug_source_limits": debug_source_limits,
        "candidate_models_in_execution_order": candidate_models,
        "excluded_source_datasets": sorted(excluded_source_datasets),
        "excluded_task_files": args.exclude_task_file,
        "excluded_task_ids": len(excluded_task_ids),
        "note": (
            "This is an unlabeled unique task pool. selected_model must be assigned only after "
            "candidate execution and verification; do not pre-expand this into router SFT labels."
        ),
    }
    summary["budget"] = {
        "candidate_count": len(candidate_models),
        "worst_case_executor_calls": len(selected) * len(candidate_models),
        "cheapest_first_note": (
            "Run candidates in ascending cost and stop once a task is solved if you want to minimize spend. "
            "Run all candidates only when you need full pairwise analysis."
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not args.dry_run:
        out_path = ROOT / args.out
        summary_path = ROOT / args.summary_out
        write_jsonl(out_path, selected)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"[wrote] {out_path.relative_to(ROOT)}")
        print(f"[wrote] {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
