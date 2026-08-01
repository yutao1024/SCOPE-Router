#!/usr/bin/env python3
"""Run one multimodal RouterSFT task through OpenClaw CLI capabilities.

This adapter keeps multimodal tasks inside the OpenClaw backend without using
the text-only `openclaw agent --message` path. It first asks OpenClaw's image
capability to describe the image, then asks OpenClaw's model capability to
answer the original question from that visual evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_image_path(path: str, payload: dict[str, Any]) -> str:
    image_path = Path(path)
    candidates = [image_path] if image_path.is_absolute() else [Path.cwd() / image_path]
    if image_path.is_absolute():
        candidates.extend([Path.cwd() / image_path.name, Path.cwd() / "raw_hf" / "mm" / "images" / image_path.name])
    for asset in payload.get("assets") or []:
        asset_path = Path(str(asset))
        if asset_path.name == image_path.name:
            candidates.append(asset_path if asset_path.is_absolute() else Path.cwd() / asset_path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Image not found: {path}")


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return "\n".join(part for part in parts if part)


def find_image_path(payload: dict[str, Any]) -> str:
    executor_input = payload.get("executor_input") or {}
    for message in executor_input.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_path":
                return resolve_image_path(str(item.get("image_path")), payload)
    raise RuntimeError("Payload does not contain an image_path in executor_input.messages.")


def task_question(payload: dict[str, Any]) -> str:
    executor_input = payload.get("executor_input") or {}
    pieces = []
    system = executor_input.get("system")
    if system:
        pieces.append(f"System instruction:\n{system}")
    for message in executor_input.get("messages") or []:
        if isinstance(message, dict):
            text = content_text(message.get("content"))
            if text:
                pieces.append(text)
    expected = payload.get("expected_output_format") or {}
    if expected:
        pieces.append("Expected output format:\n" + json.dumps(expected, ensure_ascii=False))
    return "\n\n".join(pieces).strip()


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r"\{", text):
        try:
            value = json.loads(text[match.start() :])
        except json.JSONDecodeError:
            continue
        return value if isinstance(value, dict) else None
    return None


def text_from_openclaw_json(value: dict[str, Any], fallback: str) -> str:
    inner = value.get("result", value)
    if isinstance(inner, dict):
        outputs = inner.get("outputs")
        if isinstance(outputs, list):
            parts = [str(item.get("text")) for item in outputs if isinstance(item, dict) and item.get("text")]
            if parts:
                return "\n".join(parts)
        payloads = inner.get("payloads")
        if isinstance(payloads, list):
            parts = [str(item.get("text")) for item in payloads if isinstance(item, dict) and item.get("text")]
            if parts:
                return "\n".join(parts)
        for key in ("description", "response", "text", "content", "output"):
            if inner.get(key):
                return str(inner[key])
    return fallback.strip()


def final_answer_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    if isinstance(parsed, dict):
        for key in ("answer", "final_answer", "description", "content", "text"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return stripped


def run_openclaw(
    args: list[str],
    timeout: int,
    *,
    retries: int,
    retry_sleep: float,
) -> tuple[str, dict[str, Any] | None]:
    last_error = None
    for attempt in range(retries + 1):
        try:
            result = subprocess.run(
                args,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            last_error = f"OpenClaw command timed out: {exc!r}"
        else:
            raw = result.stdout if result.stdout.strip() else result.stderr
            if result.returncode == 0:
                return raw, extract_json_object(raw)
            last_error = f"OpenClaw command failed rc={result.returncode}: {raw.strip()[:1000]}"
        if attempt < retries:
            time.sleep(retry_sleep * (attempt + 1))
    raise RuntimeError(last_error or "OpenClaw command failed")


def openclaw_cli_model_ref(payload: dict[str, Any], model_ref: str) -> str:
    if "/" in model_ref:
        return model_ref
    provider = str(payload.get("provider") or "").strip()
    model = str(payload.get("model") or model_ref).strip()
    if provider and model:
        return f"{provider}/{model}"
    raise RuntimeError(f"Cannot build OpenClaw CLI model ref from model_ref={model_ref!r}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenClaw multimodal executor for RouterSFT payloads.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-ref", required=True)
    parser.add_argument("--openclaw-bin", default="openclaw")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--local", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    payload = read_json(args.input)
    cli_model_ref = openclaw_cli_model_ref(payload, args.model_ref)
    image_path = find_image_path(payload)
    question = task_question(payload)

    describe_cmd = [
        args.openclaw_bin,
        "infer",
        "image",
        "describe",
        "--file",
        image_path,
        "--model",
        cli_model_ref,
        "--json",
    ]
    describe_raw, describe_json = run_openclaw(
        describe_cmd,
        timeout=args.timeout,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )
    image_description = text_from_openclaw_json(describe_json or {}, describe_raw)

    prompt = (
        "You are answering a visual QA benchmark item. Use the OpenClaw image description as visual evidence, "
        "then answer the original question. Output only the final requested answer: the number, choice label, "
        "short phrase, or entity. Do not include reasoning or Markdown.\n\n"
        f"OpenClaw image description:\n{image_description}\n\n"
        f"Original task:\n{question}"
    )
    model_cmd = [
        args.openclaw_bin,
        "infer",
        "model",
        "run",
        "--model",
        cli_model_ref,
        "--prompt",
        prompt,
        "--json",
    ]
    if args.local:
        model_cmd.append("--local")
    answer_raw, answer_json = run_openclaw(
        model_cmd,
        timeout=args.timeout,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )
    answer = final_answer_text(text_from_openclaw_json(answer_json or {}, answer_raw))

    write_json(
        args.output,
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": answer.strip()},
                    "finish_reason": "stop",
                }
            ],
            "usage": None,
            "openclaw": {
                "vision_mode": "image_describe_then_model_run",
                "model_ref": args.model_ref,
                "cli_model_ref": cli_model_ref,
                "image_path": image_path,
                "image_description": image_description,
                "describe_raw": describe_json or describe_raw,
                "answer_raw": answer_json or answer_raw,
            },
        },
    )


if __name__ == "__main__":
    main()
