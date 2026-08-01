#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def read_prediction(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    if isinstance(value, dict):
        return value
    raise ValueError(f"No prediction object found in {path}")


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


def pass_from_node(node: Any, question_id: str | None) -> bool | None:
    if isinstance(node, dict):
        node_id = node.get("question_id") or node.get("task_id") or node.get("id")
        if question_id is None or node_id is None or str(node_id) == str(question_id):
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
            graded_list = node.get("graded_list")
            if isinstance(graded_list, list) and graded_list:
                graded = [bool_from_status(item) for item in graded_list]
                if all(item is not None for item in graded):
                    return bool(graded[0])
            pass_at_1 = node.get("pass@1")
            if isinstance(pass_at_1, (int, float)):
                return pass_at_1 > 0
        for value in node.values():
            found = pass_from_node(value, question_id)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = pass_from_node(item, question_id)
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


def iter_candidate_json(*roots: Path) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(root.rglob("*_eval_all.json"))
        files.extend(root.rglob("*_eval.json"))
        files.extend(root.rglob("*.json"))
    return sorted(set(files), key=lambda path: path.stat().st_mtime, reverse=True)


def format_command(template: str, values: dict[str, str]) -> str:
    command = template
    for key, value in values.items():
        command = command.replace("{" + key + "}", shlex.quote(value))
    return command


def livecodebench_repo_root() -> Path | None:
    spec = importlib.util.find_spec("lcb_runner")
    if spec is None or not spec.submodule_search_locations:
        return None
    package_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
    root = package_dir.parent
    expected = root / "lcb_runner" / "prompts" / "few_shot_examples" / "generation" / "func.json"
    if expected.exists():
        return root
    return None


def default_command() -> str:
    if importlib.util.find_spec("lcb_runner.runner.custom_evaluator") is None:
        return ""
    return (
        f"{shlex.quote(sys.executable)} -m lcb_runner.runner.custom_evaluator "
        "--custom_output_file {predictions} --scenario codegeneration "
        "--custom_output_save_name {output_name} "
        + os.environ.get("LIVECODEBENCH_OFFICIAL_EXTRA_ARGS", "")
    )


def router_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def json_loads_maybe(value: Any, default: Any = None) -> Any:
    if not isinstance(value, str):
        return value if value is not None else default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return default


def raw_row_from_prediction(sample: dict[str, Any]) -> dict[str, Any] | None:
    source_ref = sample.get("source_ref") if isinstance(sample.get("source_ref"), dict) else {}
    raw_path_value = source_ref.get("raw_path")
    raw_line = source_ref.get("raw_line")
    root = router_repo_root()
    candidate_paths: list[Path] = []
    if raw_path_value:
        raw_path = Path(str(raw_path_value))
        candidate_paths.append(raw_path if raw_path.is_absolute() else root / raw_path)
    candidate_paths.append(root / "raw_hf" / "code" / "livecodebench__code_generation.jsonl")

    question_id = str(sample.get("question_id") or "")
    for path in candidate_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if raw_line and int(raw_line) != line_number:
                    continue
                row = json.loads(line)
                if raw_line or str(row.get("question_id") or "") == question_id:
                    return row
    return None


def python_function_name(starter_code: Any) -> str | None:
    match = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", str(starter_code or ""))
    if match:
        return match.group(1)
    return None


def livecodebench_sample_from_raw(row: dict[str, Any]) -> dict[str, Any]:
    public_tests = json_loads_maybe(row.get("public_test_cases"), default=[]) or []
    private_tests = json_loads_maybe(row.get("private_test_cases"), default=[]) or []
    tests = [test for test in list(public_tests) + list(private_tests) if isinstance(test, dict)]
    inputs = [str(test.get("input", "")) for test in tests if "input" in test and "output" in test]
    outputs = [str(test.get("output", "")) for test in tests if "input" in test and "output" in test]
    input_output: dict[str, Any] = {"inputs": inputs, "outputs": outputs}
    if any(test.get("testtype") == "functional" for test in tests):
        fn_name = python_function_name(row.get("starter_code"))
        if fn_name:
            input_output["fn_name"] = fn_name
    return {
        "question_id": row.get("question_id"),
        "input_output": json.dumps(input_output),
    }


def run_single_task_official(sample: dict[str, Any], result_path: Path, timeout: int) -> int:
    question_id = str(sample.get("question_id") or "")
    raw_row = raw_row_from_prediction(sample)
    if raw_row is None:
        result_path.write_text(
            json.dumps(
                {
                    "question_id": question_id,
                    "passed": False,
                    "error": "raw_livecodebench_row_not_found",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 2
    try:
        from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness
    except Exception as exc:
        result_path.write_text(
            json.dumps(
                {
                    "question_id": question_id,
                    "passed": False,
                    "error": "livecodebench_single_task_import_failed",
                    "exception": repr(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 2

    code_list = sample.get("code_list")
    code = str(code_list[0] if isinstance(code_list, list) and code_list else "")
    lcb_sample = livecodebench_sample_from_raw(raw_row)
    try:
        result, metadata = check_correctness(lcb_sample, code, timeout=timeout, debug=False)
    except Exception as exc:
        result_path.write_text(
            json.dumps(
                {
                    "question_id": question_id,
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
    passed = bool(result) and all(item is True for item in result)
    result_path.write_text(
        json.dumps(
            {
                "question_id": question_id,
                "passed": passed,
                "graded_list": [passed],
                "test_results": result,
                "metadata": metadata,
                "source": "lcb_runner.evaluation.compute_code_generation_metrics.check_correctness",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LiveCodeBench official custom evaluator and normalize one task result.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("LIVECODEBENCH_EVAL_TIMEOUT", "900")))
    parser.add_argument("--command", default=os.environ.get("LIVECODEBENCH_OFFICIAL_COMMAND", ""))
    args = parser.parse_args()

    predictions = Path(args.predictions).resolve()
    result_path = Path(args.result).resolve()
    sample = read_prediction(predictions)
    question_id = str(sample.get("question_id") or "")
    output_name = "vlm_exec_routerbench_eval_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", question_id or "one")
    workdir = result_path.parent / "livecodebench_work"
    workdir.mkdir(parents=True, exist_ok=True)

    use_custom_evaluator = os.environ.get("LIVECODEBENCH_USE_CUSTOM_EVALUATOR") == "1"
    command_template = args.command or (default_command() if use_custom_evaluator else "")
    lcb_root = Path(os.environ["LIVECODEBENCH_OFFICIAL_CWD"]).resolve() if os.environ.get("LIVECODEBENCH_OFFICIAL_CWD") else livecodebench_repo_root()
    if not command_template:
        return run_single_task_official(sample, result_path, timeout=args.timeout)
    if lcb_root is None and not args.command:
        payload = {
            "question_id": question_id,
            "passed": False,
            "error": "livecodebench_repo_root_not_found",
            "hint": "Install the official GitHub repo editable, or set LIVECODEBENCH_OFFICIAL_CWD.",
        }
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 2

    values = {
        "predictions": str(predictions),
        "prediction": str(predictions),
        "result": str(result_path),
        "results": str(result_path),
        "workdir": str(workdir),
        "question_id": question_id,
        "task_id": question_id,
        "output_name": output_name,
    }
    command = format_command(command_template, values)
    command_cwd = lcb_root or workdir
    try:
        proc = subprocess.run(command, shell=True, text=True, capture_output=True, cwd=command_cwd, timeout=args.timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        payload = {
            "question_id": question_id,
            "passed": False,
            "error": "official_eval_timeout",
            "timeout": args.timeout,
            "stdout": str(exc.stdout or "")[-4000:],
            "stderr": str(exc.stderr or "")[-4000:],
        }
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 124
    if proc.returncode != 0:
        payload = {
            "question_id": question_id,
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
    search_roots = [workdir]
    if lcb_root:
        search_roots.append(lcb_root / "output")
    for path in iter_candidate_json(*search_roots):
        try:
            candidates.append((str(path), json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, OSError):
            continue

    for source, node in candidates:
        passed = pass_from_node(node, question_id)
        if passed is not None:
            result_path.write_text(
                json.dumps({"question_id": question_id, "passed": passed, "source": source, "raw": node}, ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            return 0

    result_path.write_text(
        json.dumps(
            {
                "question_id": question_id,
                "passed": False,
                "error": "official_eval_unparseable",
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
                "candidate_files": [str(path) for path in iter_candidate_json(*search_roots)[:20]],
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
