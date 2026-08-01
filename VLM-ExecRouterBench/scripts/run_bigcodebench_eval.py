#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def read_first_jsonl(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    return value
    raise ValueError(f"No JSON object found in {path}")


def bool_from_status(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"pass", "passed", "success", "successful", "accepted", "ok", "true"}:
            return True
        if normalized in {"fail", "failed", "failure", "wrong", "wrong_answer", "error", "false"}:
            return False
    return None


def pass_from_node(node: Any, task_id: str | None) -> bool | None:
    if isinstance(node, dict):
        node_id = node.get("task_id") or node.get("id") or node.get("question_id")
        if task_id is None or node_id is None or str(node_id) == str(task_id):
            for key in ("passed", "pass", "is_passed", "success", "accepted", "correct"):
                if key in node:
                    value = bool_from_status(node.get(key))
                    if value is not None:
                        return value
            for key in ("status", "result", "verdict"):
                if key in node:
                    value = bool_from_status(node.get(key))
                    if value is not None:
                        return value
            samples = node.get("samples") or node.get("results")
            if isinstance(samples, list) and samples:
                value = pass_from_node(samples[0], task_id)
                if value is not None:
                    return value
            pass_at_1 = node.get("pass@1")
            if isinstance(pass_at_1, (int, float)):
                return pass_at_1 > 0
        for value in node.values():
            found = pass_from_node(value, task_id)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = pass_from_node(item, task_id)
            if found is not None:
                return found
    return None


def load_json_text(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def iter_candidate_json(workdir: Path) -> list[Path]:
    files = list(workdir.rglob("*.json")) + list(workdir.rglob("*.jsonl"))
    return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)


def load_candidate(path: Path) -> Any:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return None
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return rows
    return json.loads(text)


def format_command(template: str, values: dict[str, str]) -> str:
    command = template
    for key, value in values.items():
        command = command.replace("{" + key + "}", shlex.quote(value))
    return command


def run_single_task_official(sample: dict[str, Any], result_path: Path, timeout: int) -> int:
    task_id = str(sample.get("task_id") or "")
    solution = str(sample.get("solution") or sample.get("completion") or "")
    try:
        from bigcodebench.data import get_bigcodebench
        from bigcodebench.eval import PASS
        from bigcodebench.evaluate import check_correctness
    except Exception as exc:
        result_path.write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "passed": False,
                    "error": "bigcodebench_single_task_import_failed",
                    "exception": repr(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 2

    try:
        import matplotlib.pyplot  # noqa: F401
    except Exception as exc:
        result_path.write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "passed": False,
                    "error": "bigcodebench_missing_dependency",
                    "dependency": "matplotlib",
                    "exception": repr(exc),
                    "hint": "Install matplotlib in the same conda env used by scripts/run_bigcodebench_eval.py.",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 2

    subset = os.environ.get("BIGCODEBENCH_SUBSET", "full")
    try:
        problems = get_bigcodebench(subset=subset)
        problem = problems[task_id]
    except Exception as exc:
        result_path.write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "passed": False,
                    "error": "bigcodebench_problem_not_found",
                    "subset": subset,
                    "exception": repr(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 2

    try:
        result = check_correctness(
            0,
            problem,
            solution,
            int(os.environ.get("BIGCODEBENCH_MAX_AS_LIMIT", str(30 * 1024))),
            int(os.environ.get("BIGCODEBENCH_MAX_DATA_LIMIT", str(30 * 1024))),
            int(os.environ.get("BIGCODEBENCH_MAX_STACK_LIMIT", "10")),
            task_id,
            float(os.environ.get("BIGCODEBENCH_MIN_TIME_LIMIT", "1")),
            float(os.environ.get("BIGCODEBENCH_GT_TIME_LIMIT", str(max(timeout, 20)))),
        )
    except Exception as exc:
        result_path.write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "passed": False,
                    "error": "official_single_eval_failed",
                    "exception": repr(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 2

    status, details = result.get("base", ("", []))
    passed = status == PASS or str(status).lower() in {"pass", "passed", "success", "accepted"}
    result_path.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "passed": passed,
                "status": status,
                "details": details,
                "raw": result,
                "source": "bigcodebench.evaluate.check_correctness",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BigCodeBench official evaluator and normalize one task result.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("BIGCODEBENCH_EVAL_TIMEOUT", "600")))
    parser.add_argument("--command", default=os.environ.get("BIGCODEBENCH_OFFICIAL_COMMAND", ""))
    args = parser.parse_args()

    predictions = Path(args.predictions).resolve()
    result_path = Path(args.result).resolve()
    sample = read_first_jsonl(predictions)
    task_id = str(sample.get("task_id") or "")
    workdir = result_path.parent / "bigcodebench_work"
    workdir.mkdir(parents=True, exist_ok=True)

    use_full_evaluator = os.environ.get("BIGCODEBENCH_USE_FULL_EVALUATOR") == "1"
    command_template = args.command or (
        (
            "bigcodebench.evaluate --split instruct --subset full --samples {predictions} "
            + os.environ.get("BIGCODEBENCH_OFFICIAL_EXTRA_ARGS", "")
        )
        if use_full_evaluator
        else ""
    )
    if not command_template:
        return run_single_task_official(sample, result_path, timeout=args.timeout)
    values = {
        "predictions": str(predictions),
        "prediction": str(predictions),
        "result": str(result_path),
        "results": str(result_path),
        "workdir": str(workdir),
        "task_id": task_id,
    }
    command = format_command(command_template, values)
    proc = subprocess.run(command, shell=True, text=True, capture_output=True, cwd=workdir, timeout=args.timeout, check=False)
    if proc.returncode != 0:
        payload = {
            "task_id": task_id,
            "passed": False,
            "error": "official_eval_failed",
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return proc.returncode

    parsed = load_json_text(proc.stdout)
    candidates: list[tuple[str, Any]] = []
    if parsed is not None:
        candidates.append(("stdout", parsed))
    for path in iter_candidate_json(workdir):
        try:
            candidates.append((str(path), load_candidate(path)))
        except (json.JSONDecodeError, OSError):
            continue

    for source, node in candidates:
        passed = pass_from_node(node, task_id)
        if passed is not None:
            result_path.write_text(
                json.dumps({"task_id": task_id, "passed": passed, "source": source, "raw": node}, ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            return 0

    result_path.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "passed": False,
                "error": "official_eval_unparseable",
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
                "candidate_files": [str(path) for path in iter_candidate_json(workdir)[:20]],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
