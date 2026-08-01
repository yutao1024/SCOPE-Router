#!/usr/bin/env python3
"""Debug one VQA task through the mini_agent multimodal tool loop."""

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
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import generate_router_sft as router_sft  # noqa: E402
import run_mini_agent_executor as mini_agent_executor  # noqa: E402


DEFAULT_TASKS = "manifests/executor_task_pool_initial_100_per_source_no_livebrowsecomp_gaia50.jsonl"
DEFAULT_SOURCE_DATASETS = (
    "echo840/OCRBench",
    "lmms-lab/DocVQA",
    "HuggingFaceM4/ChartQA",
    "AI4Math/MathVista",
    "MMMU/MMMU",
    "lmms-lab/ai2d",
    "xai-org/RealworldQA",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_tasks(path: Path) -> list[dict[str, Any]]:
    return list(router_sft.read_jsonl(path) or [])


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(parts)
    return ""


def task_text(task: dict[str, Any]) -> str:
    executor_input = task.get("executor_input") if isinstance(task.get("executor_input"), dict) else {}
    parts = []
    for message in executor_input.get("messages") or []:
        if isinstance(message, dict):
            parts.append(text_from_content(message.get("content")))
    router_view = task.get("router_view") if isinstance(task.get("router_view"), dict) else {}
    if router_view.get("instruction"):
        parts.append(str(router_view["instruction"]))
    return "\n".join(part for part in parts if part)


def select_task(args: argparse.Namespace, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [task for task in tasks if task.get("category") == "multimodal_doc_visual"]
    candidates = [task for task in candidates if router_sft.task_image_content_items(task)]
    if args.task_id:
        candidates = [task for task in candidates if str(task.get("task_id")) == args.task_id]
    if args.source_dataset:
        candidates = [task for task in candidates if str(task.get("source_dataset")) == args.source_dataset]
    if args.prefer_exact_text:
        exact_pattern = re.compile(
            r"\b(text|word|number|date|amount|price|label|table|document|form|invoice|serial|speed limit|servings)\b",
            re.IGNORECASE,
        )
        exact_candidates = [task for task in candidates if exact_pattern.search(task_text(task))]
        if exact_candidates:
            candidates = exact_candidates
    if not args.task_id and not args.source_dataset:
        preferred = []
        for source in DEFAULT_SOURCE_DATASETS:
            preferred.extend(task for task in candidates if str(task.get("source_dataset")) == source)
        if preferred:
            candidates = preferred
    if not candidates:
        raise SystemExit("no matching VQA task found")
    return candidates[0]


def image_paths_for_task(task: dict[str, Any]) -> list[Path]:
    return [router_sft.resolve_image_path(item["image_path"]) for item in router_sft.task_image_content_items(task)]


def command_for(args: argparse.Namespace, input_path: Path, output_path: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_mini_agent_executor.py"),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--model",
        args.model,
        "--temperature",
        str(args.temperature),
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
    ]


def run_command(cmd: list[str], timeout: int, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout * 2 + 120,
        check=False,
        env=env,
    )


def compact(value: Any, max_chars: int = 1200) -> str:
    text = str(value or "").strip().replace("\r", "")
    if len(text) <= max_chars:
        return text
    return text[: max_chars // 2].rstrip() + "\n...[truncated]...\n" + text[-max_chars // 2 :].lstrip()


def ocr_health(url: str, timeout: float) -> dict[str, Any]:
    health_url = url.rsplit("/", 1)[0] + "/health"
    try:
        with urlopen(health_url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}", "url": health_url}


def ocr_smoke(url: str, image_path: Path, timeout: float) -> dict[str, Any]:
    try:
        raw_body = json.dumps({"image_path": str(image_path), "lang": "ch"}, ensure_ascii=False).encode("utf-8")
        request = Request(
            url,
            data=raw_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if isinstance(payload, dict):
            text = str(payload.get("text") or "")
            return {
                "ok": bool(payload.get("ok")),
                "text_preview": re.sub(r"\s+", " ", text).strip()[:500],
                "text_chars": len(text),
                "block_count": payload.get("block_count"),
                "latency_s": payload.get("latency_s"),
                "error": payload.get("error"),
            }
        return {"ok": False, "error": "non_object_response"}
    except Exception as exc:
        return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}


def executor_ocr_smoke(url: str, backend: str, image_paths: list[Path]) -> dict[str, Any]:
    if not image_paths:
        return {"available": False, "error": "no_images"}
    old_backend = os.environ.get("MINI_AGENT_OCR_BACKEND")
    old_url = os.environ.get("MINI_AGENT_PADDLE_OCR_URL")
    try:
        os.environ["MINI_AGENT_OCR_BACKEND"] = backend
        os.environ["MINI_AGENT_PADDLE_OCR_URL"] = url
        result = mini_agent_executor.ocr_image_tool(
            {"image_index": 0, "lang": "ch"},
            image_paths,
            model="debug-vqa-ocr-smoke",
            timeout=30,
            retries=0,
            retry_sleep=0.0,
            http_transport="urllib",
            deepseek_thinking="disabled",
        )
        return {
            "available": bool(result.get("available")),
            "backend": result.get("backend"),
            "configured_backends": result.get("configured_backends"),
            "text_preview": result.get("text_preview"),
            "text_chars": result.get("text_chars") or len(str(result.get("text") or "")),
            "block_count": result.get("block_count"),
            "latency_s": result.get("latency_s"),
            "error": result.get("error"),
        }
    except Exception as exc:
        return {"available": False, "error": f"{exc.__class__.__name__}: {exc}"}
    finally:
        if old_backend is None:
            os.environ.pop("MINI_AGENT_OCR_BACKEND", None)
        else:
            os.environ["MINI_AGENT_OCR_BACKEND"] = old_backend
        if old_url is None:
            os.environ.pop("MINI_AGENT_PADDLE_OCR_URL", None)
        else:
            os.environ["MINI_AGENT_PADDLE_OCR_URL"] = old_url


def print_output_summary(output_path: Path) -> None:
    if not output_path.exists():
        print("[vqa_debug] executor output missing")
        return
    response = read_json(output_path)
    message = router_sft.response_message(response)
    print("[vqa_debug_response]")
    print(f"  finish_reason: {(response.get('choices') or [{}])[0].get('finish_reason')}")
    print(f"  assistant_text: {compact(router_sft.message_text(message), 800)!r}")
    multimodal = response.get("mini_agent_multimodal")
    if isinstance(multimodal, dict):
        calls = multimodal.get("calls") if isinstance(multimodal.get("calls"), list) else []
        print("[vqa_debug_tools]")
        print(f"  requested_calls: {multimodal.get('requested_calls')}")
        print(f"  executed_calls: {len(calls)}")
        print(f"  configured_ocr_backends: {(multimodal.get('ocr') or {}).get('configured_backends')}")
        print(f"  ocr_tool_executed: {any(call.get('tool') == 'ocr_image' for call in calls if isinstance(call, dict))}")
        for call in calls:
            if not isinstance(call, dict):
                continue
            summary = {
                "call_index": call.get("call_index"),
                "tool": call.get("tool"),
                "backend": call.get("backend"),
                "available": call.get("available"),
                "text_preview": call.get("text_preview"),
                "error": call.get("error"),
                "region": call.get("region"),
            }
            print("  " + json.dumps(summary, ensure_ascii=False))
    mini_agent = response.get("mini_agent") if isinstance(response.get("mini_agent"), dict) else {}
    if mini_agent.get("trajectory_path"):
        print(f"[vqa_debug] trajectory: {mini_agent.get('trajectory_path')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=Path(DEFAULT_TASKS))
    parser.add_argument("--task-id", default="")
    parser.add_argument("--source-dataset", default="")
    parser.add_argument("--model", default=os.environ.get("DEBUG_VQA_MODEL", ""))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--ocr-url", default=os.environ.get("MINI_AGENT_PADDLE_OCR_URL", "http://127.0.0.1:8766/ocr"))
    parser.add_argument("--ocr-backend", default=os.environ.get("MINI_AGENT_OCR_BACKEND", "paddle_http"))
    parser.add_argument("--ocr-smoke", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--executor-ocr-smoke", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefer-exact-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-run", action="store_true", help="Only select task and run local preflight checks.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--http-transport", choices=["curl", "urllib"], default=os.environ.get("MINI_AGENT_HTTP_TRANSPORT", "curl"))
    args = parser.parse_args()

    tasks_path = args.tasks if args.tasks.is_absolute() else ROOT / args.tasks
    task = select_task(args, load_tasks(tasks_path))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or ROOT / "runs" / "vqa_mini_agent_debug" / f"{stamp}_{task.get('task_id')}"
    input_path = output_dir / "executor_input.json"
    output_path = output_dir / "executor_output.json"
    stdout_path = output_dir / "executor_stdout.txt"
    stderr_path = output_dir / "executor_stderr.txt"
    metadata_path = output_dir / "debug_metadata.json"

    image_paths = image_paths_for_task(task)
    metadata = {
        "task_id": task.get("task_id"),
        "source_dataset": task.get("source_dataset"),
        "category": task.get("category"),
        "reference_answer": router_sft.debug_reference_answer(task),
        "question": task_text(task),
        "image_paths": [str(path) for path in image_paths],
        "image_exists": [path.exists() for path in image_paths],
        "ocr_url": args.ocr_url,
        "ocr_health": ocr_health(args.ocr_url, timeout=5),
    }
    if args.ocr_smoke and image_paths:
        metadata["ocr_smoke"] = ocr_smoke(args.ocr_url, image_paths[0], timeout=30)
    if args.executor_ocr_smoke:
        metadata["executor_ocr_smoke"] = executor_ocr_smoke(args.ocr_url, args.ocr_backend, image_paths)
    write_json(input_path, task)
    write_json(metadata_path, metadata)

    print("[vqa_debug]")
    print(f"  debug_dir: {output_dir}")
    print(f"  task_id: {task.get('task_id')}")
    print(f"  source_dataset: {task.get('source_dataset')}")
    print(f"  reference_answer: {metadata['reference_answer']!r}")
    print(f"  images: {metadata['image_exists']}")
    print(f"  ocr_health: {metadata['ocr_health']}")
    if "ocr_smoke" in metadata:
        print(f"  ocr_smoke: {metadata['ocr_smoke']}")
    if "executor_ocr_smoke" in metadata:
        print(f"  executor_ocr_smoke: {metadata['executor_ocr_smoke']}")
    print(f"  question: {compact(metadata['question'], 800)}")

    if args.no_run:
        return
    if not args.model:
        raise SystemExit("--model is required unless --no-run is set")

    env = os.environ.copy()
    env["MINI_AGENT_OCR_BACKEND"] = args.ocr_backend
    env["MINI_AGENT_PADDLE_OCR_URL"] = args.ocr_url
    cmd = command_for(args, input_path, output_path)
    print("[vqa_debug_command]")
    print("  " + " ".join(shlex.quote(part) for part in cmd))
    started = time.time()
    result = run_command(cmd, timeout=args.timeout, env=env)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    print("[vqa_debug_result]")
    print(f"  returncode: {result.returncode}")
    print(f"  latency_s: {round(time.time() - started, 3)}")
    if result.stdout.strip():
        print("[executor_stdout_tail]")
        print(compact(result.stdout, 2000))
    if result.stderr.strip():
        print("[executor_stderr_tail]")
        print(compact(result.stderr, 2000))
    print_output_summary(output_path)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
