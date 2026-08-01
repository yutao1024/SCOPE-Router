#!/usr/bin/env python3
"""Run one SWE-bench task through mini-swe-agent's official batch runner.

The wrapper keeps the RouterSFT executor contract:

* read one payload from ``--input``;
* write a chat-completion-like JSON object to ``--output``;
* expose a diff in ``assistant_text`` plus ``openclaw.git_diff_present`` so the
  existing SWE-bench official export stage can consume it unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MINI_SWE_AGENT_RELATIVE_CONFIG = Path("src") / "minisweagent" / "config" / "benchmarks" / "swebench.yaml"
MINI_SWE_AGENT_RELATIVE_XML_CONFIG = Path("src") / "minisweagent" / "config" / "benchmarks" / "swebench_xml.yaml"
MINI_SWE_AGENT_TRAJECTORY_FORMAT = "mini-swe-agent-1.1"
DEFAULT_MINI_SWE_AGENT_STEP_LIMIT = 40


def env_int(*names: str, default: int) -> int:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return int(value)
    return default


def mini_swe_agent_root_candidates() -> list[Path]:
    candidates = []
    configured = os.environ.get("MINI_SWE_AGENT_ROOT", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            ROOT / "external" / "mini-swe-agent",
            ROOT.parent / "external" / "mini-swe-agent",
        ]
    )
    return candidates


def default_mini_swe_agent_root() -> Path:
    for candidate in mini_swe_agent_root_candidates():
        if candidate.exists():
            return candidate
    return mini_swe_agent_root_candidates()[0]


def default_mini_swe_agent_config(mini_root: Path) -> Path:
    return mini_root / MINI_SWE_AGENT_RELATIVE_CONFIG


def default_mini_swe_agent_xml_config(mini_root: Path) -> Path:
    return mini_root / MINI_SWE_AGENT_RELATIVE_XML_CONFIG


def default_config_specs(mini_root: Path, model: str) -> list[str]:
    configured = os.environ.get("MINI_SWE_AGENT_CONFIG", "").strip()
    normalized_model = str(model or "").lower()
    use_openrouter_textbased = normalized_model.startswith("openrouter/") or "qwen" in normalized_model or "minimax" in normalized_model
    if configured:
        specs = [configured]
    else:
        specs = [str(default_mini_swe_agent_xml_config(mini_root))]
    if use_openrouter_textbased:
        specs.append("model.model_class=openrouter_textbased")
    return specs


def mini_swe_agent_model_name(model: str, config_specs: list[str]) -> str:
    """Map RouterSFT executor refs to the model ids expected by mini-swe-agent."""
    uses_openrouter_direct = any("openrouter" in str(spec).lower() for spec in config_specs)
    if uses_openrouter_direct and model.startswith("openrouter/"):
        return model.removeprefix("openrouter/")
    return model


def validate_mini_swe_agent_root(mini_root: Path, config_specs: list[str]) -> None:
    runner = mini_root / "src" / "minisweagent" / "run" / "benchmarks" / "swebench.py"
    if not runner.exists():
        searched = ", ".join(str(path) for path in mini_swe_agent_root_candidates())
        raise RuntimeError(
            "Cannot find mini-swe-agent benchmark runner. "
            f"Resolved root={mini_root}; searched={searched}. "
            "Set MINI_SWE_AGENT_ROOT or pass --mini-swe-agent-root."
        )
    missing_configs = [config for config in config_specs if Path(config).suffix in {".yaml", ".yml"} and not Path(config).exists()]
    if missing_configs:
        raise RuntimeError(f"mini-swe-agent config file not found: {', '.join(missing_configs)}")


def validate_mini_swe_agent_runtime(mini_root: Path) -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import litellm; import minisweagent",
        ],
        cwd=mini_root,
        text=True,
        capture_output=True,
        check=False,
        env=mini_swe_agent_env(mini_root),
    )
    if probe.returncode != 0:
        detail = subprocess_error_detail(probe.stdout, probe.stderr, f"exit {probe.returncode}")
        raise RuntimeError(
            "mini-swe-agent runtime dependencies are unavailable in the current Python environment. "
            "Install mini-swe-agent dependencies in this environment, for example: "
            f"{shlex.quote(sys.executable)} -m pip install -e {shlex.quote(str(mini_root))}. "
            f"Import probe failed: {detail}"
        )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def subprocess_error_detail(stdout: str, stderr: str, fallback: str, max_chars: int = 4000) -> str:
    detail = stderr.strip() or stdout.strip() or fallback
    if len(detail) <= max_chars:
        return detail
    half = max_chars // 2
    return detail[:half].rstrip() + "\n...[truncated middle]...\n" + detail[-half:].lstrip()


def verifier_reference(payload: dict[str, Any]) -> dict[str, Any]:
    verifier = payload.get("verifier") if isinstance(payload.get("verifier"), dict) else {}
    reference = verifier.get("reference") if isinstance(verifier.get("reference"), dict) else {}
    return reference or {}


def raw_row_from_source_ref(payload: dict[str, Any]) -> dict[str, Any]:
    source_ref = payload.get("source_ref") if isinstance(payload.get("source_ref"), dict) else {}
    raw_path = source_ref.get("raw_path")
    raw_line = source_ref.get("raw_line")
    if not raw_path or not isinstance(raw_line, int):
        return {}
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle, 1):
                if index == raw_line:
                    row = json.loads(line)
                    return row if isinstance(row, dict) else {}
    except Exception:
        return {}
    return {}


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
            elif item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif item.get("type") == "image_path":
                parts.append(f"[image: {item.get('image_path')}]")
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


def task_text(payload: dict[str, Any]) -> str:
    parts = []
    executor_input = payload.get("executor_input") if isinstance(payload.get("executor_input"), dict) else {}
    for message in executor_input.get("messages") or payload.get("messages") or []:
        if isinstance(message, dict):
            parts.append(text_from_content(message.get("content")))
    router_view = payload.get("router_view") if isinstance(payload.get("router_view"), dict) else {}
    if router_view.get("instruction"):
        parts.append(str(router_view["instruction"]))
    return "\n".join(part for part in parts if part)


def instance_id_for_payload(payload: dict[str, Any], reference: dict[str, Any], raw_row: dict[str, Any]) -> str:
    for value in (reference.get("instance_id"), payload.get("source_id"), raw_row.get("instance_id"), raw_row.get("id")):
        if value:
            return str(value)
    raise RuntimeError(f"Cannot resolve SWE-bench instance_id for task {payload.get('task_id')!r}")


def mini_dataset_row(payload: dict[str, Any]) -> dict[str, Any]:
    reference = verifier_reference(payload)
    raw_row = raw_row_from_source_ref(payload)
    row = dict(raw_row)
    instance_id = instance_id_for_payload(payload, reference, raw_row)
    row.update(
        {
            "instance_id": instance_id,
            "repo": reference.get("repo") or raw_row.get("repo"),
            "base_commit": reference.get("base_commit") or raw_row.get("base_commit"),
            "problem_statement": raw_row.get("problem_statement") or task_text(payload),
        }
    )
    if reference.get("image_name") and not row.get("image_name"):
        row["image_name"] = reference["image_name"]
    if reference.get("docker_image") and not row.get("docker_image"):
        row["docker_image"] = reference["docker_image"]
    if not row.get("repo") or not row.get("base_commit"):
        raise RuntimeError("mini-swe-agent SWE-bench dataset row requires repo and base_commit.")
    return row


def write_dataset_script(dataset_dir: Path, row: dict[str, Any]) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "test.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    (dataset_dir / "mini_router_swebench.py").write_text(
        textwrap.dedent(
            """
            import json
            import datasets


            _FEATURES = {
                "instance_id": datasets.Value("string"),
                "repo": datasets.Value("string"),
                "base_commit": datasets.Value("string"),
                "problem_statement": datasets.Value("string"),
                "patch": datasets.Value("string"),
                "test_patch": datasets.Value("string"),
                "image_name": datasets.Value("string"),
                "docker_image": datasets.Value("string"),
            }


            class MiniRouterSwebench(datasets.GeneratorBasedBuilder):
                VERSION = datasets.Version("1.0.0")

                def _info(self):
                    return datasets.DatasetInfo(features=datasets.Features(_FEATURES))

                def _split_generators(self, dl_manager):
                    return [datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"path": "test.jsonl"})]

                def _generate_examples(self, path):
                    with open(path, encoding="utf-8") as handle:
                        for index, line in enumerate(handle):
                            row = json.loads(line)
                            row = {key: "" if row.get(key) is None else str(row.get(key, "")) for key in _FEATURES}
                            yield index, row
            """
        ).lstrip(),
        encoding="utf-8",
    )


def load_prediction(preds_path: Path, instance_id: str) -> dict[str, Any]:
    if not preds_path.exists():
        raise RuntimeError(f"mini-swe-agent did not write {preds_path}.")
    data = json.loads(preds_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and instance_id in data and isinstance(data[instance_id], dict):
        return data[instance_id]
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and str(item.get("instance_id")) == instance_id:
                return item
    raise RuntimeError(f"mini-swe-agent preds.json has no prediction for {instance_id}.")


def path_listing(path: Path, max_items: int = 40) -> str:
    if not path.exists():
        return "<missing>"
    items = []
    for item in sorted(path.rglob("*"))[:max_items]:
        try:
            rel = item.relative_to(path)
        except ValueError:
            rel = item
        suffix = "/" if item.is_dir() else f" ({item.stat().st_size} bytes)"
        items.append(f"{rel}{suffix}")
    return ", ".join(items) if items else "<empty>"


def file_tail(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return "<missing>"
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:] if len(text) > max_chars else text


def require_prediction(
    preds_path: Path,
    instance_id: str,
    result: subprocess.CompletedProcess[str],
    cmd: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    try:
        return load_prediction(preds_path, instance_id)
    except RuntimeError as exc:
        detail = subprocess_error_detail(result.stdout, result.stderr, f"exit {result.returncode}")
        command = " ".join(shlex.quote(str(part)) for part in cmd)
        log_tail = file_tail(output_dir / "minisweagent.log")
        output_files = path_listing(output_dir)
        raise RuntimeError(
            f"{exc} mini-swe-agent returncode={result.returncode}; command={command}; "
            f"detail={detail}; output_files={output_files}; minisweagent_log_tail={log_tail}"
        ) from exc


def latest_traj_path(output_dir: Path, instance_id: str) -> Path | None:
    instance_dir = output_dir / instance_id
    candidates = sorted(instance_dir.glob("*.traj.json"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def trajectory_info(traj_path: Path | None) -> dict[str, Any]:
    if not traj_path or not traj_path.exists():
        return {}
    try:
        payload = json.loads(traj_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
    submission = str(info.get("submission") or "")
    return {
        "exit_status": info.get("exit_status"),
        "submission_bytes": len(submission.encode("utf-8")),
        "model_stats": info.get("model_stats") if isinstance(info.get("model_stats"), dict) else None,
    }


def usage_from_model_stats(model_stats: Any) -> dict[str, Any] | None:
    if not isinstance(model_stats, dict):
        return None
    usage = {
        "input_tokens": model_stats.get("input_tokens"),
        "output_tokens": model_stats.get("output_tokens"),
        "total_tokens": model_stats.get("total_tokens"),
    }
    usage = {key: value for key, value in usage.items() if isinstance(value, int)}
    return usage or None


def model_patch_from_prediction(prediction: dict[str, Any]) -> str:
    return str(prediction.get("model_patch") or "").strip()


def mini_swe_agent_failed_without_patch(
    prediction: dict[str, Any],
    result: subprocess.CompletedProcess[str],
    output_dir: Path,
) -> str | None:
    if model_patch_from_prediction(prediction):
        return None
    if result.returncode != 0:
        return subprocess_error_detail(result.stdout, result.stderr, f"exit {result.returncode}")

    log_tail = file_tail(output_dir / "minisweagent.log")
    detail = "\n".join(
        part
        for part in (result.stdout[-4000:], result.stderr[-4000:], log_tail)
        if part
    )
    normalized = detail.lower()
    failure_markers = (
        "error processing instance",
        "timeoutexpired",
        "traceback",
        "cannot connect to the docker daemon",
        "docker daemon",
        "no space left on device",
        "permission denied",
        "failed to start",
    )
    if any(marker in normalized for marker in failure_markers):
        return subprocess_error_detail(result.stdout, result.stderr, "mini-swe-agent internal failure")
    return None


def normalize_prediction(
    *,
    prediction: dict[str, Any],
    payload: dict[str, Any],
    result: subprocess.CompletedProcess[str],
    output_dir: Path,
    latency_s: float,
) -> dict[str, Any]:
    reference = verifier_reference(payload)
    instance_id = str(prediction.get("instance_id") or reference.get("instance_id") or payload.get("source_id") or "")
    patch = model_patch_from_prediction(prediction)
    content = f"```diff\n{patch}\n```" if patch else (result.stdout.strip() or result.stderr.strip())
    traj_path = latest_traj_path(output_dir, instance_id) if instance_id else None
    traj_info = trajectory_info(traj_path)
    model_stats = traj_info.get("model_stats")
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop" if result.returncode == 0 else "error",
            }
        ],
        "usage": usage_from_model_stats(model_stats),
        "mini_agent": {
            "backend": "mini_agent",
            "adapter": "mini_swe_agent",
            "mode": "swebench_real_repo",
            "swebench_real_repo": True,
            "repo": reference.get("repo"),
            "base_commit": reference.get("base_commit"),
            "instance_id": instance_id or None,
            "model": payload.get("model"),
            "model_ref": payload.get("executor_model_ref") or payload.get("openclaw_model_ref"),
            "prediction": prediction,
            "trajectory_path": str(traj_path) if traj_path else None,
            "trajectory_saved": bool(traj_path),
            "trajectory_format": MINI_SWE_AGENT_TRAJECTORY_FORMAT if traj_path else None,
            "trajectory_exit_status": traj_info.get("exit_status"),
            "trajectory_submission_bytes": traj_info.get("submission_bytes"),
            "trajectory_model_stats": model_stats,
            "output_dir": str(output_dir),
            "git_diff_bytes": len(patch.encode("utf-8")),
            "git_diff_present": bool(patch),
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
            "latency_s": round(latency_s, 3),
        },
        "openclaw": {
            "backend": "mini_agent",
            "adapter": "mini_swe_agent",
            "swebench_real_repo": True,
            "repo": reference.get("repo"),
            "base_commit": reference.get("base_commit"),
            "instance_id": instance_id or None,
            "model": payload.get("model"),
            "model_ref": payload.get("executor_model_ref") or payload.get("openclaw_model_ref"),
            "prediction": prediction,
            "trajectory_path": str(traj_path) if traj_path else None,
            "trajectory_saved": bool(traj_path),
            "trajectory_format": MINI_SWE_AGENT_TRAJECTORY_FORMAT if traj_path else None,
            "trajectory_exit_status": traj_info.get("exit_status"),
            "trajectory_submission_bytes": traj_info.get("submission_bytes"),
            "trajectory_model_stats": model_stats,
            "output_dir": str(output_dir),
            "git_diff_bytes": len(patch.encode("utf-8")),
            "git_diff_present": bool(patch),
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
            "latency_s": round(latency_s, 3),
        },
    }


def mini_swe_agent_env(mini_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    src = str(mini_root / "src")
    env["PYTHONPATH"] = os.pathsep.join([src, env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one SWE-bench task through mini-swe-agent batch mode.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="")
    parser.add_argument("--model-ref", default="")
    parser.add_argument("--mini-swe-agent-root", type=Path, default=default_mini_swe_agent_root())
    parser.add_argument("--config", action="append", default=None, help="mini-swe-agent -c/--config spec. Repeatable.")
    parser.add_argument("--environment-class", default=os.environ.get("MINI_SWE_AGENT_ENVIRONMENT_CLASS", "docker"))
    parser.add_argument("--work-root", type=Path, default=Path(os.environ.get("MINI_SWE_AGENT_WORK_ROOT", "/tmp/vlm-exec-routerbench-mini-swe-agent")).expanduser())
    parser.add_argument("--keep-output", action=argparse.BooleanOptionalAction, default=os.environ.get("MINI_SWE_AGENT_KEEP_OUTPUT", "1") != "0")
    parser.add_argument(
        "--step-limit",
        type=int,
        default=env_int(
            "MINI_SWE_AGENT_STEP_LIMIT",
            "SWEBENCH_MINI_AGENT_STEP_LIMIT",
            default=DEFAULT_MINI_SWE_AGENT_STEP_LIMIT,
        ),
        help="mini-swe-agent agent.step_limit override. Use 0 to keep the config file default.",
    )
    args = parser.parse_args()

    payload = read_json(args.input)
    model = args.model or str(payload.get("executor_model_ref") or payload.get("openclaw_model_ref") or payload.get("model") or "")
    timeout = int(float((payload.get("generation_config") or {}).get("timeout") or 600))
    row = mini_dataset_row(payload)
    instance_id = str(row["instance_id"])
    run_id = f"{re.sub(r'[^A-Za-z0-9_.-]+', '-', instance_id)}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
    run_root = args.work_root / run_id
    dataset_dir = run_root / "dataset"
    output_dir = run_root / "output"
    write_dataset_script(dataset_dir, row)
    output_dir.mkdir(parents=True, exist_ok=True)

    mini_root = args.mini_swe_agent_root.expanduser()
    config_specs = args.config or default_config_specs(mini_root, model)
    if args.step_limit > 0:
        config_specs.append(f"agent.step_limit={args.step_limit}")
    validate_mini_swe_agent_root(mini_root, config_specs)
    validate_mini_swe_agent_runtime(mini_root)
    agent_model = mini_swe_agent_model_name(model, config_specs)
    cmd = [
        sys.executable,
        str(mini_root / "src" / "minisweagent" / "run" / "benchmarks" / "swebench.py"),
        "--subset",
        str(dataset_dir),
        "--split",
        "test",
        "--output",
        str(output_dir),
        "--workers",
        "1",
        "--redo-existing",
        "--filter",
        f"^{re.escape(instance_id)}$",
        "--model",
        agent_model,
    ]
    for config in config_specs:
        cmd.extend(["--config", config])
    if args.environment_class:
        cmd.extend(["--environment-class", args.environment_class])

    started = time.time()
    result = subprocess.run(
        cmd,
        cwd=mini_root,
        text=True,
        capture_output=True,
        timeout=timeout * 2 + 300,
        check=False,
        env=mini_swe_agent_env(mini_root),
    )
    prediction = require_prediction(output_dir / "preds.json", instance_id, result, cmd, output_dir)
    failure_detail = mini_swe_agent_failed_without_patch(prediction, result, output_dir)
    if failure_detail:
        raise RuntimeError(f"mini-swe-agent execution failed without model_patch: {failure_detail}")
    normalized = normalize_prediction(
        prediction=prediction,
        payload={
            **payload,
            "executor_model_ref": args.model_ref or payload.get("executor_model_ref") or payload.get("openclaw_model_ref"),
            "openclaw_model_ref": args.model_ref or payload.get("openclaw_model_ref"),
            "model": agent_model,
        },
        result=result,
        output_dir=output_dir,
        latency_s=time.time() - started,
    )
    write_json(args.output, normalized)
    if not args.keep_output:
        shutil.rmtree(run_root, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[run_swebench_mini_swe_agent_executor_error] {exc}", file=sys.stderr)
        raise
