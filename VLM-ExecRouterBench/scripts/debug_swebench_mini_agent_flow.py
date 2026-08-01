#!/usr/bin/env python3
"""Debug one SWE-bench task through mini-agent execution and official export.

This is intentionally a single-task harness. It mirrors the RouterSFT
``mini_agent`` SWE-bench path, but keeps every intermediate artifact under a
stable debug directory:

* executor input JSON consumed by ``run_swebench_mini_agent_executor.py``;
* executor output JSON from mini-swe-agent;
* one RouterSFT-style executor result row;
* SWE-bench official prediction export, or sb-cli JSON for Multimodal.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import generate_router_sft as router_sft  # noqa: E402
import run_swebench_mini_swe_agent_executor as mini_swe_agent  # noqa: E402


SWEBENCH_MULTIMODAL_DATASET = "princeton-nlp/SWE-bench_Multimodal"
DEFAULT_TASKS = "manifests/executor_task_pool_initial_100_per_source_no_livebrowsecomp_gaia50.jsonl"
DEFAULT_COMMAND = (
    "python3 scripts/run_swebench_mini_agent_executor.py "
    "--input {input} --output {output} --model {executor_model_ref} --model-ref {executor_model_ref}"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def load_tasks(path: Path) -> list[dict[str, Any]]:
    return list(router_sft.read_jsonl(path) or [])


def select_task(args: argparse.Namespace, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = tasks
    if args.task_id:
        candidates = [task for task in candidates if str(task.get("task_id")) == args.task_id]
    if args.source_dataset:
        candidates = [task for task in candidates if str(task.get("source_dataset")) == args.source_dataset]
    candidates = [task for task in candidates if router_sft.is_official_swebench_real_repo_task(task)]
    if not candidates:
        raise SystemExit("no matching official SWE-bench real-repo task found")
    return candidates[0]


def task_instance_id(task: dict[str, Any]) -> str:
    instance_id = router_sft.swebench_instance_id_for_task(task)
    if not instance_id:
        raise RuntimeError(f"Cannot resolve SWE-bench instance_id for task {task.get('task_id')!r}")
    return instance_id


def reference_for(task: dict[str, Any]) -> dict[str, Any]:
    return router_sft.swebench_reference(task)


def count_images(task: dict[str, Any]) -> int:
    count = 0
    executor_input = task.get("executor_input") if isinstance(task.get("executor_input"), dict) else {}
    for message in executor_input.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            count += sum(1 for item in content if isinstance(item, dict) and item.get("type") == "image_path")
    return count


def command_for(args: argparse.Namespace, input_path: Path, output_path: Path, model_ref: str, task: dict[str, Any]) -> str:
    template = args.command
    values = {
        "input": str(input_path),
        "output": str(output_path),
        "model": args.model,
        "provider": router_sft.provider_for_model(args.model),
        "task_id": str(task.get("task_id")),
        "category": str(task.get("category")),
        "timeout": str(args.timeout),
        "temperature": str(args.temperature),
        "max_tokens": str(args.max_tokens),
        "executor_model_ref": model_ref,
        "openclaw_model_ref": model_ref,
    }
    return template.format(**values)


def run_command(command: str, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        shlex.split(command),
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout * 2 + 300,
        check=False,
    )


def text_tail(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def print_failure_tail(stdout_path: Path, stderr_path: Path) -> None:
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    if stdout.strip():
        print("[executor_stdout_tail]")
        print(text_tail(stdout).rstrip())
    if stderr.strip():
        print("[executor_stderr_tail]")
        print(text_tail(stderr).rstrip())


def compact_text(value: Any, max_chars: int = 1200) -> str:
    text = str(value or "").strip()
    text = text.replace("\r", "")
    if len(text) <= max_chars:
        return text
    return text[: max_chars // 2].rstrip() + "\n...[truncated]...\n" + text[-max_chars // 2 :].lstrip()


def print_trajectory_tail(path: str | None, max_messages: int = 8) -> None:
    if not path:
        return
    traj_path = Path(path)
    if not traj_path.exists():
        return
    try:
        payload = read_json(traj_path)
    except Exception:
        return
    info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
    print("[trajectory_summary]")
    print(f"  exit_status: {info.get('exit_status')}")
    print(f"  submission_bytes: {len(str(info.get('submission') or '').encode('utf-8'))}")
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    for message in messages[-max_messages:]:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        extra = message.get("extra") if isinstance(message.get("extra"), dict) else {}
        marker = extra.get("exit_status") or extra.get("exception_str") or ""
        content = compact_text(message.get("content"))
        print(f"  [{role}] {marker}")
        if content:
            print(content)


def make_result_row(
    *,
    task: dict[str, Any],
    model: str,
    model_ref: str,
    command: str,
    response: dict[str, Any],
    latency_s: float,
) -> dict[str, Any]:
    message = router_sft.response_message(response)
    assistant_text = router_sft.message_text(message)
    openclaw_meta = response.get("openclaw") if isinstance(response.get("openclaw"), dict) else {}
    if openclaw_meta.get("swebench_real_repo") and not openclaw_meta.get("git_diff_present"):
        exit_status = str(openclaw_meta.get("trajectory_exit_status") or "").strip()
        if exit_status:
            reason_status = re.sub(r"[^a-z0-9]+", "_", exit_status.lower()).strip("_")
            passed, reason = False, f"model_wrong: mini_agent_{reason_status}_no_git_diff"
        else:
            passed, reason = False, "model_wrong: no_git_diff"
    else:
        passed, reason = False, "swebench_official_pending"
    status = "needs_official" if reason == "swebench_official_pending" else router_sft.status_for_verification(passed, reason)
    failure_type = None if status == "needs_official" else router_sft.failure_type_for_result(passed, reason)
    usage = response.get("usage")
    usage_tokens = router_sft.normalize_usage_tokens(usage)
    trajectory_tokens, last_call_tokens = router_sft.usage_token_variants(usage)
    judge_tokens = router_sft.normalize_usage_tokens(None)
    choices = response.get("choices") or []
    finish_reason = choices[0].get("finish_reason") if choices else None
    tool_summary = router_sft.openclaw_tool_summary(response)
    return {
        "status": status,
        "task_id": str(task.get("task_id")),
        "candidate_model": model,
        "provider": router_sft.provider_for_model(model),
        "executor_backend": "mini_agent",
        "executor_model_ref": model_ref,
        "executor_command_kind": "swebench_real_repo",
        "executor_command": command,
        "openclaw_model_ref": model_ref,
        "openclaw_command_kind": "swebench_real_repo",
        "openclaw_command": command,
        "category": task.get("category"),
        "difficulty_prior": task.get("difficulty_prior"),
        "source_dataset": task.get("source_dataset"),
        "source_id": task.get("source_id"),
        "reference_answer": router_sft.debug_reference_answer(task),
        "passed": passed,
        "failure_type": failure_type,
        "verify_reason": reason,
        "assistant_text": assistant_text,
        "tool_calls": router_sft.result_tool_calls_for_row(message, tool_summary),
        "finish_reason": finish_reason,
        "usage": usage,
        "raw_model_response": router_sft.raw_model_response_for_row(response),
        "openclaw_tool_summary": tool_summary,
        "browsecomp_plus": None,
        **usage_tokens,
        **router_sft.prefixed_token_fields("trajectory", trajectory_tokens),
        **router_sft.prefixed_token_fields("last_call", last_call_tokens),
        "openclaw": openclaw_meta or None,
        "judge_usage": [],
        **router_sft.prefixed_token_fields("judge", judge_tokens),
        "latency_s": round(latency_s, 3),
    }


def export_official_predictions(
    *,
    tasks_path: Path,
    results_path: Path,
    output_path: Path,
    task: dict[str, Any],
    model: str,
) -> Path:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "export_swebench_official_predictions.py"),
        "--results",
        str(results_path),
        "--tasks",
        str(tasks_path),
        "--output",
        str(output_path),
        "--source-dataset",
        str(task.get("source_dataset")),
        "--status",
        "needs_official",
        "--candidate-model",
        model,
        "--task-id",
        str(task.get("task_id")),
        "--dedupe-instance",
        "best",
    ]
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"official prediction export failed: {detail}")
    return output_path


def print_summary(
    paths: dict[str, Path],
    task: dict[str, Any],
    row: dict[str, Any] | None,
    command: str,
    model: str,
) -> None:
    reference = reference_for(task)
    mini_root = mini_swe_agent.default_mini_swe_agent_root()
    config_specs = mini_swe_agent.default_config_specs(mini_root, model)
    print("[swebench_debug]")
    print(f"  task_id: {task.get('task_id')}")
    print(f"  source_dataset: {task.get('source_dataset')}")
    print(f"  instance_id: {task_instance_id(task)}")
    print(f"  repo: {reference.get('repo')}")
    print(f"  base_commit: {reference.get('base_commit')}")
    print(f"  images: {count_images(task)}")
    print(f"  mini_swe_agent_root: {mini_root}")
    print(f"  mini_swe_agent_config: {config_specs}")
    print(f"  mini_swe_agent_model: {mini_swe_agent.mini_swe_agent_model_name(model, config_specs)}")
    print(f"  command: {command}")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    if row:
        meta = row.get("openclaw") if isinstance(row.get("openclaw"), dict) else {}
        print(f"  status: {row.get('status')}")
        print(f"  verify_reason: {row.get('verify_reason')}")
        print(f"  git_diff_present: {meta.get('git_diff_present')}")
        print(f"  git_diff_bytes: {meta.get('git_diff_bytes')}")
        print(f"  trajectory_exit_status: {meta.get('trajectory_exit_status')}")
        print(f"  trajectory_model_stats: {meta.get('trajectory_model_stats')}")
        print(f"  trajectory_path: {meta.get('trajectory_path')}")
        print_trajectory_tail(meta.get("trajectory_path"))


def inspect_failed_run(path: Path) -> None:
    debug_dir = path if path.is_absolute() else ROOT / path
    stdout_path = debug_dir / "executor_stdout.txt"
    stderr_path = debug_dir / "executor_stderr.txt"
    print(f"[swebench_debug_inspect] debug_dir: {debug_dir}")
    print(f"  executor_stdout: {stdout_path}")
    print(f"  executor_stderr: {stderr_path}")
    print_failure_tail(stdout_path, stderr_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--task-id", default="")
    parser.add_argument("--source-dataset", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--model-ref", default="")
    parser.add_argument("--model-ref-map", default=os.environ.get("OPENCLAW_MODEL_REF_MAP", ""))
    parser.add_argument("--command", default=os.environ.get("SWEBENCH_MINI_AGENT_COMMAND") or os.environ.get("SWEBENCH_MINI_SWE_AGENT_COMMAND") or DEFAULT_COMMAND)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--deepseek-thinking", choices=["disabled", "enabled"], default="disabled")
    parser.add_argument("--dry-run", action="store_true", help="Only write executor input and print command.")
    parser.add_argument("--skip-export", action="store_true", help="Run mini-agent but do not export official predictions.")
    parser.add_argument("--executor-output", type=Path, default=None, help="Use an existing executor output JSON instead of running mini-agent.")
    parser.add_argument("--inspect-run-dir", type=Path, default=None, help="Print stdout/stderr tails from an existing debug run directory and exit.")
    args = parser.parse_args()

    if args.inspect_run_dir:
        inspect_failed_run(args.inspect_run_dir)
        return
    if not args.model:
        raise SystemExit("--model is required unless --inspect-run-dir is used.")

    tasks_path = Path(args.tasks)
    if not tasks_path.is_absolute():
        tasks_path = ROOT / tasks_path
    tasks = load_tasks(tasks_path)
    task = select_task(args, tasks)
    model_ref_map = router_sft.load_openclaw_model_ref_map(args.model_ref_map)
    model_ref = args.model_ref or router_sft.executor_model_ref_for(args.model, model_ref_map)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "runs" / "swebench_mini_agent_debug" / f"{stamp}_{task.get('task_id')}"
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    input_path = output_dir / "executor_input.json"
    output_path = output_dir / "executor_output.json"
    results_path = output_dir / "executor_results.jsonl"
    predictions_path = output_dir / "official_predictions.jsonl"

    payload = router_sft.openclaw_executor_payload(
        task=task,
        model=args.model,
        openclaw_model_ref=model_ref,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        deepseek_thinking=args.deepseek_thinking,
    )
    write_json(input_path, payload)
    command = command_for(args, input_path, output_path, model_ref, task)

    paths: dict[str, Path] = {"debug_dir": output_dir, "executor_input": input_path}
    row: dict[str, Any] | None = None
    if args.dry_run:
        print_summary(paths, task, row, command, args.model)
        return

    started = time.time()
    if args.executor_output:
        existing_output = args.executor_output if args.executor_output.is_absolute() else ROOT / args.executor_output
        response = read_json(existing_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(existing_output.read_text(encoding="utf-8"), encoding="utf-8")
        latency_s = 0.0
    else:
        completed = run_command(command, args.timeout)
        stdout_path = output_dir / "executor_stdout.txt"
        stderr_path = output_dir / "executor_stderr.txt"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        paths["executor_stdout"] = stdout_path
        paths["executor_stderr"] = stderr_path
        if completed.returncode != 0:
            print_summary(paths, task, row, command, args.model)
            print_failure_tail(stdout_path, stderr_path)
            raise SystemExit(f"mini-agent command failed with exit {completed.returncode}; inspect stdout/stderr above")
        response = read_json(output_path)
        latency_s = time.time() - started

    paths["executor_output"] = output_path
    row = make_result_row(task=task, model=args.model, model_ref=model_ref, command=command, response=response, latency_s=latency_s)
    write_jsonl(results_path, [row])
    paths["executor_results"] = results_path

    if not args.skip_export and row.get("status") == "needs_official":
        export_official_predictions(
            tasks_path=tasks_path,
            results_path=results_path,
            output_path=predictions_path,
            task=task,
            model=args.model,
        )
        paths["official_predictions"] = predictions_path
        if str(task.get("source_dataset")) == SWEBENCH_MULTIMODAL_DATASET:
            rows = list(router_sft.read_jsonl(predictions_path) or [])
            sbcli_path = router_sft.sbcli_json_path_for_prediction_file(predictions_path)
            sbcli_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            paths["sbcli_predictions"] = sbcli_path

    print_summary(paths, task, row, command, args.model)


if __name__ == "__main__":
    main()
