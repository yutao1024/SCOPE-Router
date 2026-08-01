#!/usr/bin/env python3
"""Run one router executor task through the local OpenClaw CLI.

This script is intentionally small glue:
  1. read the JSON payload produced by generate_router_sft.py,
  2. send text tasks as a flattened prompt to `openclaw agent`, simulated
     tool-call tasks through `openclaw infer model run`, and image tasks through
     OpenClaw Gateway `agent` attachments when available, with command/API
     fallback modes for smoke tests,
  3. normalize OpenClaw's JSON/text output into a chat-completion-like object.

Run it from an environment where `openclaw` is installed, for example:

    conda run -n openclaw python scripts/run_openclaw_executor.py --input in.json --output out.json

If your OpenClaw config already points at the desired model, no model patching is
needed. If it does not, pass --model-ref-template, e.g. "api/{model}".
"""

from __future__ import annotations

import argparse
import ast
import base64
import http.client
import json
import mimetypes
import os
import re
import shutil
import shlex
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OPENAI_COMPATIBLE_PROVIDERS = {"bailian", "dubrify", "openrouter", "deepseek"}

OPENCLAW_PROVIDER_MODEL_PATCHES: dict[tuple[str, str], dict[str, Any]] = {
    (
        "openrouter",
        "qwen/qwen3-vl-8b-instruct",
    ): {
        "id": "qwen/qwen3-vl-8b-instruct",
        "name": "Qwen: Qwen3 VL 8B Instruct",
        "reasoning": False,
        "input": ["text", "image"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 256000,
        "maxTokens": 32768,
        "compat": {"supportsUsageInStreaming": True, "maxTokensField": "max_tokens"},
        "api": "openai-completions",
    },
    (
        "openrouter",
        "qwen/qwen3.5-35b-a3b",
    ): {
        "id": "qwen/qwen3.5-35b-a3b",
        "name": "Qwen: Qwen3.5-35B-A3B",
        "reasoning": True,
        "input": ["text", "image", "video"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 262144,
        "maxTokens": 65536,
        "compat": {"supportsUsageInStreaming": True, "maxTokensField": "max_tokens"},
        "api": "openai-completions",
    },
    (
        "openrouter",
        "mistralai/mistral-small-2603",
    ): {
        "id": "mistralai/mistral-small-2603",
        "name": "Mistral: Mistral Small 4",
        "reasoning": True,
        "input": ["text", "image"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 262144,
        "maxTokens": 65536,
        "compat": {"supportsUsageInStreaming": True, "maxTokensField": "max_tokens"},
        "api": "openai-completions",
    },
    (
        "openrouter",
        "openai/gpt-5.4-mini",
    ): {
        "id": "openai/gpt-5.4-mini",
        "name": "OpenAI: GPT-5.4 Mini",
        "reasoning": True,
        "input": ["text", "image"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 1050000,
        "maxTokens": 8192,
        "compat": {
            "supportsReasoningEffort": True,
            "supportsUsageInStreaming": True,
            "maxTokensField": "max_tokens",
        },
        "api": "openai-completions",
    },
    (
        "openrouter",
        "google/gemini-2.5-flash-lite",
    ): {
        "id": "google/gemini-2.5-flash-lite",
        "name": "Google: Gemini 2.5 Flash Lite",
        "reasoning": True,
        "input": ["text", "image", "audio", "video"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 1048576,
        "maxTokens": 65535,
        "compat": {"supportsUsageInStreaming": True, "maxTokensField": "max_tokens"},
        "api": "openai-completions",
    },
    (
        "openrouter",
        "minimax/minimax-m3",
    ): {
        "id": "minimax/minimax-m3",
        "name": "MiniMax: MiniMax M3",
        "reasoning": True,
        "input": ["text", "image", "video"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 1048576,
        "maxTokens": 65536,
        "compat": {"supportsUsageInStreaming": True, "maxTokensField": "max_tokens"},
        "api": "openai-completions",
    },
}
GATEWAY_CALL_SCRIPT = Path(__file__).with_name("openclaw_gateway_call.mjs")
OPENCLAW_BOOTSTRAP_FILES = (
    "AGENTS.md",
    "BOOTSTRAP.md",
    "HEARTBEAT.md",
    "IDENTITY.md",
    "SOUL.md",
    "TOOLS.md",
    "USER.md",
)
BROWSECOMP_PLUS_PLUGIN_ID = "browsecomp-plus-tool"
BROWSECOMP_PLUS_TOOL_NAME = "search_browsecomp_plus"
BROWSECOMP_PLUS_DEFAULT_TOP_K = 5
BROWSECOMP_PLUS_DEFAULT_MAX_DOC_CHARS = 2400
BROWSECOMP_PLUS_DEFAULT_MAX_TOOL_CALLS = 5
BROWSECOMP_PLUS_DEFAULT_PLUGIN_DIR = (
    Path(__file__).resolve().parents[1] / "external" / "openclaw_browsecomp_plus_tool"
)
NETWORK_TOOLS = {"web_search", "web_fetch"}
CODE_EXECUTION_TOOLS = {"read", "edit", "write", "apply_patch", "exec", "process"}
LOCAL_ANALYSIS_TOOLS = {"read", "exec", "process"}
WEB_AGENT_TOOLS = {"web_search", "web_fetch", "read", "exec", "process"}
PROMPT_ONLY_TOOL_DENY = {"*", "group:openclaw"}
NO_TOOL_NOISE_TOOLS = (
    CODE_EXECUTION_TOOLS
    | NETWORK_TOOLS
    | {
        "cron",
        "image",
        "image_generate",
        "music_generate",
        "video_generate",
        "memory_search",
        "memory_get",
        "sessions_list",
        "sessions_history",
        "sessions_send",
        "sessions_spawn",
        "sessions_yield",
        "subagents",
        "session_status",
        "skill_workshop",
        "get_goal",
        "create_goal",
        "update_goal",
        "update_plan",
    }
)


class BrowseCompPlusRecoverableError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        data: dict[str, Any],
        summary: dict[str, Any],
        fallback_reason: str,
    ) -> None:
        super().__init__(message)
        self.data = data
        self.summary = summary
        self.fallback_reason = fallback_reason


class BrowseCompPlusBudgetExceeded(BrowseCompPlusRecoverableError):
    def __init__(self, message: str, *, data: dict[str, Any], summary: dict[str, Any]) -> None:
        super().__init__(
            message,
            data=data,
            summary=summary,
            fallback_reason="browsecomp_plus_tool_call_budget_exceeded",
        )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(text)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def resolve_openclaw_bin(openclaw_bin: str) -> str:
    if os.path.sep in openclaw_bin:
        path = Path(openclaw_bin).expanduser()
        if path.exists():
            return str(path.resolve())
        raise FileNotFoundError(f"OpenClaw binary not found: {path}")
    resolved = shutil.which(openclaw_bin)
    if resolved:
        return resolved
    raise FileNotFoundError(
        f"OpenClaw binary {openclaw_bin!r} was not found in PATH. "
        "Set OPENCLAW_BIN=/absolute/path/to/openclaw, pass --openclaw-bin, "
        "or run this wrapper inside the environment where OpenClaw is installed."
    )


def openclaw_cmd(openclaw_bin: str, *args: str) -> list[str]:
    if Path(openclaw_bin).suffix == ".mjs":
        return [os.environ.get("OPENCLAW_NODE_BIN", "node"), openclaw_bin, *args]
    return [openclaw_bin, *args]


def candidate_bootstrap_workspaces(config_path: Path) -> list[Path]:
    configured = os.environ.get("OPENCLAW_BOOTSTRAP_WORKSPACE")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            config_path.parent / "workspace",
            Path.cwd() / ".openclaw" / "workspace",
            Path.cwd().parent / ".openclaw" / "workspace",
        ]
    )
    return candidates


def prepare_openclaw_workspace(workspace: Path, config_path: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    for source_dir in candidate_bootstrap_workspaces(config_path):
        if not source_dir.exists():
            continue
        copied = 0
        for name in OPENCLAW_BOOTSTRAP_FILES:
            source = source_dir / name
            target = workspace / name
            if not source.exists() or target.exists():
                continue
            try:
                shutil.copy2(source, target)
                copied += 1
            except OSError:
                pass
        if copied:
            return


def response_from_message(
    message: dict[str, Any],
    *,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }


def response_from_has_tool_calls(response: dict[str, Any]) -> bool:
    choices = response.get("choices") or []
    if not choices:
        return False
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    return bool(isinstance(message, dict) and message.get("tool_calls"))


def should_enforce_tool_policy_audit() -> bool:
    return os.environ.get("OPENCLAW_ENFORCE_TOOL_POLICY_AUDIT", "1") != "0"


def openclaw_system_prompt_report(data: dict[str, Any]) -> dict[str, Any] | None:
    candidates: list[Any] = [data]
    inner = data.get("result") if isinstance(data.get("result"), dict) else None
    if inner is not None:
        candidates.append(inner)
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        report = candidate.get("systemPromptReport")
        if isinstance(report, dict):
            return report
        meta = candidate.get("meta") if isinstance(candidate.get("meta"), dict) else {}
        report = meta.get("systemPromptReport")
        if isinstance(report, dict):
            return report
    return None


def openclaw_prompt_tool_names(data: dict[str, Any]) -> list[str]:
    report = openclaw_system_prompt_report(data)
    if not report:
        return []
    tools = report.get("tools") if isinstance(report.get("tools"), dict) else {}
    names = []
    for entry in tools.get("entries") or []:
        if isinstance(entry, dict) and entry.get("name"):
            names.append(str(entry["name"]))
    return names


def assert_openclaw_tool_policy_matches_prompt(response: dict[str, Any]) -> None:
    if not should_enforce_tool_policy_audit():
        return
    metadata = response.get("openclaw")
    if not isinstance(metadata, dict):
        return
    if metadata.get("tool_policy_audit_skipped_reason"):
        return
    raw_response = metadata.get("raw_response")
    if not isinstance(raw_response, dict):
        return
    actual_tools = set(openclaw_prompt_tool_names(raw_response))
    if not actual_tools:
        return
    allowed_value = metadata.get("tool_policy_allowed")
    denied = {str(item) for item in metadata.get("tool_policy_denied") or []}
    denied_present = sorted(actual_tools & denied)
    unexpected_allowed: list[str] = []
    if isinstance(allowed_value, list):
        allowed = {str(item) for item in allowed_value}
        unexpected_allowed = sorted(actual_tools - allowed)
    if not denied_present and not unexpected_allowed:
        return
    raise RuntimeError(
        "OpenClaw tool policy audit failed: actual system prompt tools do not match "
        "the declared benchmark policy. "
        f"declared_allowed={allowed_value!r}; declared_denied={sorted(denied)!r}; "
        f"actual_tools={sorted(actual_tools)!r}; denied_present={denied_present!r}; "
        f"unexpected_allowed={unexpected_allowed!r}. "
        "This usually means a Gateway call used a resident/main agent configuration "
        "instead of the runtime benchmark config."
    )


def attach_openclaw_metadata(response: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    response.setdefault("openclaw", {})
    if isinstance(response["openclaw"], dict):
        response["openclaw"].update(metadata)
    assert_openclaw_tool_policy_matches_prompt(response)
    return response


def normalize_executor_output(value: Any) -> dict[str, Any]:
    if isinstance(value, dict) and isinstance(value.get("choices"), list):
        return value

    if isinstance(value, dict) and isinstance(value.get("message"), dict):
        return response_from_message(
            value["message"],
            finish_reason=value.get("finish_reason"),
            usage=value.get("usage"),
        )

    if isinstance(value, dict) and "assistant_text" in value:
        message = {
            "role": "assistant",
            "content": str(value.get("assistant_text") or ""),
        }
        tool_calls = value.get("tool_calls")
        if tool_calls:
            message["tool_calls"] = tool_calls
        return response_from_message(
            message,
            finish_reason=value.get("finish_reason"),
            usage=value.get("usage"),
        )

    if isinstance(value, dict) and "content" in value:
        message = {
            "role": value.get("role", "assistant"),
            "content": value.get("content") or "",
        }
        tool_calls = value.get("tool_calls")
        if tool_calls:
            message["tool_calls"] = tool_calls
        return response_from_message(
            message,
            finish_reason=value.get("finish_reason"),
            usage=value.get("usage"),
        )

    if isinstance(value, str):
        return response_from_message({"role": "assistant", "content": value})

    raise RuntimeError("Executor output must be a chat completion, message, assistant_text object, or text.")


def read_executor_output(path: Path, stdout: str) -> dict[str, Any]:
    text = ""
    if path.exists() and path.stat().st_size:
        text = path.read_text(encoding="utf-8").strip()
    if not text:
        text = stdout.strip()
    if not text:
        raise RuntimeError("Executor command produced no output.")

    try:
        return normalize_executor_output(json.loads(text))
    except json.JSONDecodeError:
        pass

    jsonl_rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            jsonl_rows.append(json.loads(line))
        except json.JSONDecodeError:
            jsonl_rows = []
            break
    if jsonl_rows:
        for row in reversed(jsonl_rows):
            if isinstance(row, dict) and row.get("type") == "message":
                message = row.get("message")
                if isinstance(message, dict) and message.get("role") == "assistant":
                    usage = row.get("usage") if isinstance(row.get("usage"), dict) else None
                    return response_from_message(message, usage=usage)
            if isinstance(row, dict) and row.get("role") == "assistant":
                return response_from_message(row)

    return normalize_executor_output(text)


def resolve_image_path(path: str, payload: dict[str, Any] | None = None) -> str:
    image_path = Path(path)
    candidates = [image_path] if image_path.is_absolute() else [Path.cwd() / image_path]
    if image_path.is_absolute():
        candidates.extend([Path.cwd() / image_path.name, Path.cwd() / "raw_hf" / "mm" / "images" / image_path.name])
    if payload:
        for asset in payload.get("assets") or []:
            asset_path = Path(str(asset))
            if asset_path.name == image_path.name:
                candidates.append(asset_path if asset_path.is_absolute() else Path.cwd() / asset_path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(path)


def image_to_data_url(path: str, payload: dict[str, Any] | None = None) -> str:
    image_path = Path(resolve_image_path(path, payload))
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def convert_content_for_api(content: Any, payload: dict[str, Any] | None = None) -> Any:
    if not isinstance(content, list):
        return content

    converted = []
    for item in content:
        if not isinstance(item, dict):
            converted.append(item)
            continue
        if item.get("type") == "image_path":
            converted.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(str(item.get("image_path")), payload)},
                }
            )
        else:
            converted.append(item)
    return converted


def payload_has_images(payload: dict[str, Any]) -> bool:
    for message in payload_messages_for_api(payload):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"image_path", "image_url"}:
                return True
    return False


def payload_messages_for_api(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages = payload.get("messages")
    if isinstance(messages, list) and messages:
        return [dict(message) for message in messages if isinstance(message, dict)]

    executor_input = payload.get("executor_input") or {}
    output = []
    system = executor_input.get("system")
    if system:
        output.append({"role": "system", "content": str(system)})
    for message in executor_input.get("messages") or []:
        if not isinstance(message, dict):
            continue
        output.append(dict(message))
    return output


def build_api_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages = []
    for message in payload_messages_for_api(payload):
        item = dict(message)
        item["content"] = convert_content_for_api(item.get("content"), payload)
        messages.append(item)
    return messages


def text_from_content(
    content: Any,
    payload: dict[str, Any] | None = None,
    *,
    include_images: bool = True,
) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            item_type = item.get("type")
            if item_type == "text":
                parts.append(str(item.get("text", "")))
            elif item_type == "image_path":
                if include_images:
                    parts.append(f"[image: {resolve_image_path(str(item.get('image_path')), payload)}]")
            elif item_type == "image_url":
                if include_images:
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict):
                        image_url = image_url.get("url")
                    parts.append(f"[image_url: {image_url}]")
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


def expected_output_type(payload: dict[str, Any]) -> str:
    expected = payload.get("expected_output_format") or {}
    return str(expected.get("type") or "")


def payload_has_tool(payload: dict[str, Any], tool_name: str) -> bool:
    executor_input = payload.get("executor_input") if isinstance(payload.get("executor_input"), dict) else {}
    for tool in list(payload.get("tools") or []) + list(executor_input.get("tools") or []):
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = function.get("name") or tool.get("name")
        if str(name or "").strip() == tool_name:
            return True
    return False


def payload_declares_tools(payload: dict[str, Any]) -> bool:
    executor_input = payload.get("executor_input") if isinstance(payload.get("executor_input"), dict) else {}
    return bool(payload.get("tools") or executor_input.get("tools"))


def payload_needs_browsecomp_plus_plugin(payload: dict[str, Any]) -> bool:
    return payload_has_tool(payload, BROWSECOMP_PLUS_TOOL_NAME)


def model_parts(payload: dict[str, Any], model_ref: str) -> tuple[str, str]:
    if "/" in model_ref:
        provider, model = model_ref.split("/", 1)
        return provider, model
    provider = str(payload.get("provider") or "")
    if provider:
        return provider, model_ref
    model = str(payload.get("model") or model_ref)
    if "/" in model:
        provider, model_id = model.split("/", 1)
        return provider, model_id
    raise RuntimeError(f"Cannot infer provider for model ref {model_ref!r}")


def api_model_parts(payload: dict[str, Any], model_ref: str) -> tuple[str, str, str]:
    """Resolve a provider/model pair usable by the direct OpenAI-compatible API path.

    Vision tasks cannot be sent through the text-only OpenClaw CLI. They use the
    provider API directly, so local/plugin OpenClaw refs need to fall back to the
    original payload provider and model.
    """
    requested_ref = model_ref
    provider, model = model_parts(payload, model_ref)
    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        return provider, model, requested_ref

    payload_provider = str(payload.get("provider") or "")
    payload_model = str(payload.get("model") or "")
    if payload_provider in OPENAI_COMPATIBLE_PROVIDERS and payload_model:
        return payload_provider, payload_model, requested_ref

    raise RuntimeError(
        f"Vision API mode requires an OpenAI-compatible provider, got model_ref={requested_ref!r} "
        f"and payload provider={payload_provider!r}."
    )


def api_model_name(provider: str, model: str) -> str:
    if provider != "openrouter":
        return model
    if "/" in model:
        return model
    env_name = "OPENROUTER_" + re.sub(r"[^A-Za-z0-9]+", "_", model).upper().strip("_") + "_MODEL"
    return os.environ.get(env_name, f"openai/{model}")


def api_config(provider: str) -> tuple[str, str]:
    if provider == "bailian":
        base_url = os.environ.get("BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DASHSCOPE_API_KEY for Bailian/Qwen models.")
        return base_url.rstrip("/") + "/chat/completions", api_key
    if provider == "dubrify":
        base_url = os.environ.get("DUBRIFY_BASE_URL", "https://api.dubrify.com/v1")
        api_key = os.environ.get("DUBRIFY_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DUBRIFY_API_KEY for Dubrify/GPT models.")
        return base_url.rstrip("/") + "/chat/completions", api_key
    if provider == "openrouter":
        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY for OpenRouter/GPT models.")
        return base_url.rstrip("/") + "/chat/completions", api_key
    if provider == "deepseek":
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY for DeepSeek models.")
        return base_url.rstrip("/") + "/chat/completions", api_key
    raise RuntimeError(f"Unsupported OpenAI-compatible provider for vision API mode: {provider}")


def post_vision_chat_completion(
    payload: dict[str, Any],
    model_ref: str,
    timeout_s: int,
    retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    provider, model, requested_model_ref = api_model_parts(payload, model_ref)
    url, api_key = api_config(provider)
    model = api_model_name(provider, model)
    generation = payload.get("generation_config") or {}
    body = {
        "model": model,
        "messages": build_api_messages(payload),
        "temperature": generation.get("temperature", 0.0),
        "max_tokens": int(generation.get("max_tokens") or 1024),
    }
    request = Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last_error = None
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout_s) as response:
                parsed = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = f"vision API HTTP {exc.code}: {detail[:1000]}"
            if exc.code < 500 and exc.code not in {408, 429}:
                raise RuntimeError(last_error) from exc
        except (URLError, TimeoutError, ssl.SSLError, http.client.HTTPException, OSError) as exc:
            last_error = f"vision API request failed: {exc!r}"
        if attempt < retries:
            time.sleep(retry_sleep * (attempt + 1))
    else:
        raise RuntimeError(last_error or "vision API request failed")

    if not isinstance(parsed, dict):
        raise RuntimeError("vision API returned non-object JSON")
    if parsed.get("error"):
        raise RuntimeError(f"vision API error: {json.dumps(parsed.get('error'), ensure_ascii=False)[:1000]}")
    parsed.setdefault("openclaw", {})
    parsed["openclaw"]["vision_mode"] = "api"
    parsed["openclaw"]["model_ref"] = requested_model_ref
    parsed["openclaw"]["api_provider"] = provider
    parsed["openclaw"]["api_model"] = model
    return parsed


def run_vision_command(
    *,
    command_template: str,
    input_path: Path,
    output_path: Path,
    payload: dict[str, Any],
    model_ref: str,
    timeout_s: int,
    temperature: Any,
    max_tokens: int | None,
) -> dict[str, Any]:
    if not command_template:
        raise RuntimeError("--vision-command is required when --vision-mode=command.")

    format_values = {
        "input": str(input_path),
        "output": str(output_path),
        "model": str(payload.get("model") or ""),
        "provider": str(payload.get("provider") or ""),
        "model_ref": model_ref,
        "openclaw_model_ref": model_ref,
        "task_id": str(payload.get("task_id") or "task"),
        "category": str(payload.get("category") or ""),
        "timeout": str(timeout_s),
        "temperature": str(temperature if temperature is not None else 0.0),
        "max_tokens": str(max_tokens or ""),
    }
    command = command_template.format(**format_values)
    # The multimodal command can run multiple OpenClaw calls internally, for
    # example image describe + model answer, each with its own retry policy.
    command_timeout = timeout_s * 6 + 180
    result = subprocess.run(
        shlex.split(command),
        text=True,
        capture_output=True,
        timeout=command_timeout,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"OpenClaw vision command failed: {detail[:1000]}")

    normalized = read_executor_output(output_path, result.stdout)
    normalized.setdefault("openclaw", {})
    normalized["openclaw"]["vision_mode"] = "command"
    normalized["openclaw"]["model_ref"] = model_ref
    return normalized


def image_url_value(item: dict[str, Any]) -> str:
    image_url = item.get("image_url")
    if isinstance(image_url, dict):
        image_url = image_url.get("url")
    return str(image_url or "")


def payload_gateway_attachments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    attachments = []
    for message in payload_messages_for_api(payload):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "image_path":
                path = Path(resolve_image_path(str(item.get("image_path")), payload))
                mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
                attachments.append(
                    {
                        "type": "image",
                        "mimeType": mime_type,
                        "fileName": path.name,
                        "content": base64.b64encode(path.read_bytes()).decode("ascii"),
                    }
                )
            elif item_type == "image_url":
                url = image_url_value(item)
                match = re.match(r"^data:([^;,]+);base64,(.*)$", url, flags=re.DOTALL)
                if not match:
                    raise RuntimeError(
                        "OpenClaw Gateway media mode requires local image_path inputs or data:image/* image_url values."
                    )
                mime_type = match.group(1) or "image/png"
                attachments.append(
                    {
                        "type": "image",
                        "mimeType": mime_type,
                        "fileName": f"image-{len(attachments) + 1}{mimetypes.guess_extension(mime_type) or '.png'}",
                        "content": match.group(2).strip(),
                    }
                )
    return attachments


def normalize_openclaw_agent_raw(
    raw: str,
    *,
    metadata: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    parsed = extract_json_object(raw)
    if payload_needs_browsecomp_plus_plugin(payload):
        assert_browsecomp_plus_tool_used(parsed or {})
    nonfinal_error = openclaw_nonfinal_error_message(parsed or {})
    if nonfinal_error:
        raise_browsecomp_plus_recoverable_if_possible(
            payload=payload,
            data=parsed or {},
            message=f"OpenClaw returned non-final BrowseComp-Plus output after retrieval: {nonfinal_error[:1000]}",
            fallback_reason="browsecomp_plus_nonfinal_after_search",
        )
        raise RuntimeError(f"OpenClaw returned non-final output: {nonfinal_error[:1000]}")
    assistant_text, usage = extract_openclaw_text(parsed or {}, raw)
    assistant_text = normalize_assistant_text(assistant_text)
    error_message = openclaw_error_message(parsed or {}, assistant_text)
    if error_message:
        raise_browsecomp_plus_recoverable_if_possible(
            payload=payload,
            data=parsed or {},
            message=f"OpenClaw returned BrowseComp-Plus error output after retrieval: {error_message[:1000]}",
            fallback_reason="browsecomp_plus_agent_error_after_search",
        )
        raise RuntimeError(f"OpenClaw returned error output: {error_message[:1000]}")
    message = {"role": "assistant", "content": assistant_text}
    if expected_output_type(payload) == "openai_tool_calls_then_final_answer":
        message = tool_message_from_text(assistant_text) or message
    normalized = {
        "choices": [
            {
                "message": message,
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
        "openclaw": {
            **metadata,
            "raw_response": parsed,
        },
    }
    return normalized


def resolve_openclaw_package_dir(openclaw_bin: str) -> Path | None:
    configured = os.environ.get("OPENCLAW_PACKAGE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    binary = shutil.which(openclaw_bin) if not os.path.isabs(openclaw_bin) else openclaw_bin
    if not binary:
        return None

    resolved = Path(binary).resolve()
    candidates = []
    if resolved.name in {"openclaw", "openclaw.mjs"}:
        candidates.append(resolved.parent)
    if resolved.parent.name == "bin":
        env_root = resolved.parent.parent
        candidates.extend(
            [
                env_root / "lib" / "node_modules" / "openclaw",
                env_root / "lib" / "node_modules" / "@openclaw" / "openclaw",
            ]
        )
    try:
        launcher = resolved.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        launcher = ""
    for match in re.finditer(r"((?:\.\.?/|/)[^'\"\s]+(?:node_modules/openclaw|openclaw\.mjs|dist/index\.js))", launcher):
        target = Path(match.group(1))
        if not target.is_absolute():
            target = (resolved.parent / target).resolve()
        if target.name in {"openclaw.mjs", "index.js"}:
            candidates.append(target.parent.parent if target.parent.name == "dist" else target.parent)
        else:
            candidates.append(target)
    candidates.extend(resolved.parents)
    for candidate in candidates:
        package_json = candidate / "package.json"
        if not package_json.exists():
            continue
        try:
            package = json.loads(package_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if package.get("name") == "openclaw":
            return candidate
    return None


def resolve_openclaw_state_dir() -> Path:
    configured = os.environ.get("OPENCLAW_STATE_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path("~/.openclaw").expanduser()


def runtime_state_dir_for_config(config_path: Path) -> Path | None:
    configured = os.environ.get("OPENCLAW_RUNTIME_STATE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    name = config_path.name
    if ".vlm-exec-routerbench-" not in name:
        return None
    return (config_path.parent / "state" / config_path.stem).resolve()


def configure_runtime_state(config: dict[str, Any], state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    session = config.setdefault("session", {})
    if isinstance(session, dict):
        session["store"] = str(state_dir / "agents" / "{agentId}" / "sessions" / "sessions.json")


def resolve_openclaw_session_store_path(config_path: Path, agent: str) -> Path:
    config = read_json(config_path) if config_path.exists() else {}
    store = ((config.get("session") or {}).get("store") if isinstance(config, dict) else None)
    if isinstance(store, str) and store.strip():
        expanded = store.strip().replace("{agentId}", agent)
        return Path(expanded).expanduser().resolve()
    return (resolve_openclaw_state_dir() / "agents" / agent / "sessions" / "sessions.json").resolve()


def write_gateway_session_overrides(
    *,
    config_path: Path,
    agent: str,
    session_id: str,
    session_key: str,
    payload: dict[str, Any],
    model_ref: str,
    workspace: Path,
) -> None:
    provider = None
    model = None
    if model_ref:
        provider, model = model_parts(payload, model_ref)
    store_path = resolve_openclaw_session_store_path(config_path, agent)
    try:
        store = read_json(store_path) if store_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        store = {}
    if not isinstance(store, dict):
        store = {}
    now_ms = int(time.time() * 1000)
    entry = store.get(session_key)
    if not isinstance(entry, dict):
        entry = {}
    entry.update(
        {
            "sessionId": session_id,
            "updatedAt": now_ms,
            "workspaceDir": str(workspace),
            "workspace": str(workspace),
        }
    )
    if provider and model:
        entry.update(
            {
                "providerOverride": provider,
                "modelOverride": model,
            }
        )
    store[session_key] = entry
    write_json(store_path, store)


def is_retryable_gateway_transport_error(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    markers = (
        "gatewaytransporterror",
        "gateway closed (1006",
        "abnormal closure",
        "gateway not yet ready",
        "connection refused",
        "econnrefused",
        "socket hang up",
    )
    return any(marker in normalized for marker in markers)


def should_fallback_gateway_edit_on_isolation_failure() -> bool:
    return os.environ.get("OPENCLAW_GATEWAY_EDIT_FALLBACK_ON_ISOLATION_FAILURE", "1") != "0"


def run_gateway_helper_raw(
    *,
    openclaw_bin: str,
    method: str,
    params: dict[str, Any],
    timeout_s: int,
    workspace: Path,
    expect_final: bool,
    config_path: Path | None = None,
) -> str:
    if not GATEWAY_CALL_SCRIPT.exists():
        raise RuntimeError(f"OpenClaw Gateway helper not found: {GATEWAY_CALL_SCRIPT}")

    timeout_ms = max(10_000, (timeout_s + 60) * 1000)
    params_path = None
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=".json",
        prefix="openclaw-gateway-params-",
        dir=workspace,
        delete=False,
    ) as params_file:
        params_path = Path(params_file.name)
        json.dump(params, params_file, ensure_ascii=False)

    node_bin = os.environ.get("OPENCLAW_NODE_BIN", "node")
    package_dir = resolve_openclaw_package_dir(openclaw_bin)
    if package_dir is None:
        raise RuntimeError(
            "Cannot resolve OpenClaw package directory from openclaw binary. "
            "Set OPENCLAW_PACKAGE_DIR to the installed openclaw package root."
        )
    cmd = [
        node_bin,
        str(GATEWAY_CALL_SCRIPT),
        method,
        str(params_path),
        str(timeout_ms),
        "1" if expect_final else "0",
        str(package_dir),
    ]
    helper_env = os.environ.copy()
    if config_path is not None:
        resolved_config_path = config_path.expanduser().resolve()
        helper_env["OPENCLAW_CONFIG"] = str(resolved_config_path)
        helper_env["OPENCLAW_CONFIG_PATH"] = str(resolved_config_path)
        if should_isolate_gateway_runtime():
            state_dir = runtime_state_dir_for_config(resolved_config_path)
            if state_dir is not None:
                state_dir.mkdir(parents=True, exist_ok=True)
                helper_env["OPENCLAW_STATE_DIR"] = str(state_dir)
            helper_env.pop("OPENCLAW_GATEWAY_URL", None)
            helper_env.pop("OPENCLAW_GATEWAY_TOKEN", None)

    try:
        default_transport_retries = "8" if config_path is not None and should_isolate_gateway_runtime() else "2"
        transport_retries = max(0, int(os.environ.get("OPENCLAW_GATEWAY_TRANSPORT_RETRIES", default_transport_retries)))
        transport_retry_sleep = max(0.0, float(os.environ.get("OPENCLAW_GATEWAY_TRANSPORT_RETRY_SLEEP", "2.0")))
        result = None
        raw = ""
        for attempt in range(transport_retries + 1):
            result = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=timeout_s + 120,
                cwd=workspace,
                env=helper_env,
                check=False,
            )
            raw = result.stdout if result.stdout.strip() else result.stderr
            if result.returncode == 0:
                break
            if attempt >= transport_retries or not is_retryable_gateway_transport_error(raw):
                break
            if transport_retry_sleep:
                time.sleep(transport_retry_sleep * (attempt + 1))
    finally:
        if params_path is not None:
            try:
                params_path.unlink()
            except OSError:
                pass
    if result is None:
        raise RuntimeError(f"OpenClaw Gateway {method} failed before launching helper.")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        detail_limit = 6000 if is_retryable_gateway_transport_error(detail) else 2000
        raise RuntimeError(
            f"OpenClaw Gateway {method} failed rc={result.returncode}: "
            f"{detail[:detail_limit]}"
        )
    return raw


def apply_gateway_session_overrides(
    *,
    openclaw_bin: str,
    config_path: Path,
    agent: str,
    session_id: str,
    session_key: str,
    payload: dict[str, Any],
    model_ref: str,
    timeout_s: int,
    workspace: Path,
) -> None:
    write_gateway_session_overrides(
        config_path=config_path,
        agent=agent,
        session_id=session_id,
        session_key=session_key,
        payload=payload,
        model_ref=model_ref,
        workspace=workspace,
    )
    default_mode = "store" if should_isolate_gateway_runtime() else "patch"
    mode = os.environ.get("OPENCLAW_GATEWAY_MODEL_OVERRIDE_MODE", default_mode).strip().lower()
    if mode not in {"none", "patch", "store", "store-and-patch"}:
        raise RuntimeError(
            "OPENCLAW_GATEWAY_MODEL_OVERRIDE_MODE must be one of: none, patch, store, store-and-patch"
        )
    if mode in {"patch", "store-and-patch"}:
        patch_params = {"key": session_key}
        if model_ref:
            patch_params["model"] = model_ref
        # OpenClaw 2026.6.x rejects workspace/workspaceDir in sessions.patch.
        # Keep workspace in the session store override above, and only send it
        # to the gateway when explicitly requested for older runtimes.
        if os.environ.get("OPENCLAW_GATEWAY_PATCH_WORKSPACE", "0") == "1":
            patch_params["workspace"] = str(workspace)
            patch_params["workspaceDir"] = str(workspace)
        if len(patch_params) == 1:
            return
        raw = run_gateway_helper_raw(
            openclaw_bin=openclaw_bin,
            method="sessions.patch",
            params=patch_params,
            timeout_s=min(timeout_s, 30),
            workspace=workspace,
            expect_final=False,
            config_path=config_path,
        )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"OpenClaw Gateway sessions.patch returned non-JSON output: {raw[:1000]}") from exc
        resolved = parsed.get("result", parsed)
        if isinstance(resolved, dict):
            resolved_model = (resolved.get("resolved") or {}).get("model") if isinstance(resolved.get("resolved"), dict) else None
            resolved_provider = (
                (resolved.get("resolved") or {}).get("modelProvider")
                if isinstance(resolved.get("resolved"), dict)
                else None
            )
            if model_ref and resolved_provider and resolved_model:
                expected_provider, expected_model = model_parts(payload, model_ref)
                if resolved_provider != expected_provider or resolved_model != expected_model:
                    raise RuntimeError(
                        "OpenClaw Gateway sessions.patch resolved a different model: "
                        f"{resolved_provider}/{resolved_model} != {expected_provider}/{expected_model}"
                    )


def is_retryable_gateway_empty_reply_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "no_reply" in text or "no reply from agent" in text


def is_no_reply_agent_error(text: str) -> bool:
    lowered = text.lower()
    return (
        "openclaw returned error output: no_reply" in lowered
        or "openclaw returned error output: no reply from agent" in lowered
        or "agent couldn't generate a response" in lowered
    )


def is_reasoning_only_agent_error(text: str) -> bool:
    lowered = text.lower()
    return (
        "reasoning-only assistant turn detected" in lowered
        or "visible-answer continuation" in lowered
    )


def is_agent_request_schema_error(text: str) -> bool:
    lowered = text.lower()
    return (
        "provider rejected the request schema or tool payload" in lowered
        or "request schema or tool payload" in lowered
    )


def run_gateway_agent(
    *,
    openclaw_bin: str,
    agent: str,
    session_id: str,
    payload: dict[str, Any],
    model_ref: str,
    timeout_s: int,
    workspace: Path,
    config_path: Path,
    attachments: list[dict[str, Any]] | None = None,
    include_images: bool = True,
    message: str | None = None,
    param_overrides: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    session_key = f"agent:{agent}:{session_id}"
    apply_gateway_session_overrides(
        openclaw_bin=openclaw_bin,
        config_path=config_path,
        agent=agent,
        session_id=session_id,
        session_key=session_key,
        payload=payload,
        model_ref=model_ref,
        timeout_s=timeout_s,
        workspace=workspace,
    )
    params = {
        "message": message if message is not None else build_prompt(payload, include_images=include_images),
        "agentId": agent,
        "sessionId": session_id,
        "sessionKey": session_key,
        "timeout": timeout_s,
        "idempotencyKey": session_id,
    }
    # OpenClaw 2026.6.x rejects workspace/workspaceDir/cwd in gateway agent
    # params. The workspace is already written into the session store and the
    # helper process runs with cwd=workspace, so keep these params opt-in for
    # older runtimes that still expect them.
    if os.environ.get("OPENCLAW_GATEWAY_AGENT_WORKSPACE_PARAMS", "0") == "1":
        params.update(
            {
                "workspace": str(workspace),
                "workspaceDir": str(workspace),
                "cwd": str(workspace),
            }
        )
    if param_overrides:
        params.update(param_overrides)
    if attachments:
        params["attachments"] = attachments
    retries = max(0, int(os.environ.get("OPENCLAW_GATEWAY_RETRIES", "1")))
    retry_sleep = max(0.0, float(os.environ.get("OPENCLAW_GATEWAY_RETRY_SLEEP", "2.0")))
    idempotency_base = f"{session_id}-{uuid.uuid4().hex[:8]}"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        params["idempotencyKey"] = idempotency_base if attempt == 0 else f"{idempotency_base}-retry-{attempt}"
        try:
            raw = run_gateway_helper_raw(
                openclaw_bin=openclaw_bin,
                method="agent",
                params=params,
                timeout_s=timeout_s,
                workspace=workspace,
                expect_final=True,
                config_path=config_path,
            )
        except RuntimeError as exc:
            last_error = exc
            text = str(exc)
            if is_agent_request_schema_error(text):
                raise RuntimeError(
                    "OpenClaw Gateway agent request was rejected by the provider schema/tool "
                    f"validator for model {model_ref}: {text[:1000]}"
                ) from exc
            if attempt >= retries or not is_retryable_gateway_empty_reply_error(exc):
                raise
            if retry_sleep:
                time.sleep(retry_sleep)
            continue
        try:
            return normalize_openclaw_agent_raw(
                raw,
                metadata={
                    "session_id": session_id,
                    "model_ref": model_ref,
                    **(metadata or {}),
                    **({"gateway_retry_attempt": attempt} if attempt else {}),
                },
                payload=payload,
            )
        except RuntimeError as exc:
            if attempt >= retries or not is_retryable_gateway_empty_reply_error(exc):
                raise
            last_error = exc
            if retry_sleep:
                time.sleep(retry_sleep)
    raise RuntimeError(f"OpenClaw Gateway agent failed after retries: {last_error}")


def build_gateway_media_prompt(payload: dict[str, Any]) -> str:
    lines = []
    executor_input = payload.get("executor_input") or {}
    system = executor_input.get("system")
    if system:
        lines.append(str(system))

    messages = executor_input.get("messages") or payload.get("messages") or []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = text_from_content(message.get("content"), payload, include_images=False)
        if content:
            lines.append(content)

    expected = payload.get("expected_output_format") or {}
    expected_description = str(expected.get("description") or "").strip() if isinstance(expected, dict) else ""
    if expected_description:
        lines.append(f"Answer format hint: {expected_description}")
    if expected_output_type(payload) == "workspace_patch_then_final_answer":
        lines.append(
            "This is a multimodal software issue-resolution task. Use the attached image(s) as visual evidence "
            "for the expected/current behavior. Do not stop because repository files are absent. Infer the "
            "smallest correct change from the issue text and visual evidence, and return a concrete unified "
            "diff or precise patch-style code change followed by a concise summary."
        )

    lines.append(
        "The image is already attached to this message. Answer from the attached image(s) and "
        "the text above only. Do not call image, read, web_fetch, shell, or any other tool. "
        "Do not inspect workspace files or invent local paths such as 1.png. Return only the "
        "direct final answer text, with no JSON wrapper."
    )
    return "\n\n".join(line for line in lines if line).strip()


def build_gateway_media_edit_prompt(payload: dict[str, Any]) -> str:
    lines = []
    executor_input = payload.get("executor_input") or {}
    system = executor_input.get("system")
    if system:
        lines.append(f"System instruction:\n{system}")

    messages = executor_input.get("messages") or payload.get("messages") or []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        content = text_from_content(message.get("content"), payload, include_images=False)
        if content:
            lines.append(f"{role.capitalize()} message:\n{content}")

    expected = payload.get("expected_output_format") or {}
    expected_description = str(expected.get("description") or "").strip() if isinstance(expected, dict) else ""
    if expected_description and expected_output_type(payload) != "workspace_patch_then_final_answer":
        lines.append(f"Task output intent:\n{expected_description}")

    if should_deny_gateway_edit_apply_patch():
        patch_instruction = (
            "apply_patch is disabled for this debug run. Use repository inspection plus edit/write "
            "tool calls only."
        )
    else:
        patch_instruction = (
            "You may use apply_patch, but only with OpenClaw's patch envelope: "
            "*** Begin Patch, *** Update File: path, @@ hunks, and *** End Patch. "
            "Never pass standard git diff headers such as --- or +++ to apply_patch. "
            "If apply_patch fails once for syntax or hunk format, stop using it and switch "
            "to read plus edit/write."
        )

    lines.append(
        "This is a real repository editing task. The target repository is checked out in the "
        "workspace, and the image evidence is attached to this message. Make actual file edits "
        "in the workspace; do not return a patch, unified diff, or patch-style text as the main "
        "result. Use OpenClaw tools to inspect files and edit the repository. Do not call "
        "image/web/network tools or read local image paths. Locate the exact case-sensitive file "
        "path before editing, then read the relevant range. For edit calls, copy exact old text "
        "from the latest read result, including whitespace and indentation. If an edit oldText "
        "match fails, reread the smallest relevant range and make a smaller anchored edit. "
        f"{patch_instruction} The final assistant text should be only a concise summary of the "
        "actual edits; the harness will capture git diff from the workspace."
    )
    return "\n\n".join(line for line in lines if line).strip()


def build_gateway_media_edit_extra_prompt() -> str:
    if should_deny_gateway_edit_apply_patch():
        patch_instruction = (
            "apply_patch is disabled for this debug run; use read plus edit/write instead."
        )
    else:
        patch_instruction = (
            "If using apply_patch, use OpenClaw's *** Begin Patch / *** Update File / @@ envelope, "
            "never standard git diff headers. After one patch syntax or hunk-format failure, switch "
            "to read plus edit/write instead of retrying apply_patch."
        )
    return (
        "This is a repository editing task with attached image evidence. Use the user message, "
        "attached media, and checked-out workspace. Do not call image/web/network tools or read "
        "local image paths. Leave the fix as actual file edits; do not return unified diff text "
        "as the main result. Use read/search before editing, and prefer exact edit/write operations "
        f"when possible. {patch_instruction}"
    )


def run_gateway_media_agent(
    *,
    openclaw_bin: str,
    agent: str,
    session_id: str,
    payload: dict[str, Any],
    model_ref: str,
    timeout_s: int,
    workspace: Path,
    config_path: Path,
) -> dict[str, Any]:
    attachments = payload_gateway_attachments(payload)
    if not attachments:
        raise RuntimeError("OpenClaw Gateway media mode found no image attachments.")
    gateway_timeout_s = int(float(os.environ.get("OPENCLAW_GATEWAY_MEDIA_TIMEOUT", str(timeout_s))))
    gateway_timeout_s = max(10, min(timeout_s, gateway_timeout_s))
    return run_gateway_agent(
        openclaw_bin=openclaw_bin,
        agent=agent,
        session_id=session_id,
        payload=payload,
        model_ref=model_ref,
        timeout_s=gateway_timeout_s,
        workspace=workspace,
        config_path=config_path,
        attachments=attachments,
        include_images=False,
        message=build_gateway_media_prompt(payload),
        param_overrides={
            "thinking": "off",
            "deliver": False,
            "bootstrapContextMode": "lightweight",
            "extraSystemPrompt": (
                "This is a benchmark multimodal inference call. The image is already attached. "
                "Do not use tools, web, shell, or workspace files. Never call the image tool or "
                "read local image paths. Use only the user message and attached image(s). Return "
                "only the direct final answer text, with no JSON wrapper."
            ),
        },
        metadata={
            "vision_mode": "gateway",
            "attachment_count": len(attachments),
            "session_model_override": model_ref,
        },
    )


def run_gateway_media_edit_agent(
    *,
    openclaw_bin: str,
    agent: str,
    session_id: str,
    payload: dict[str, Any],
    model_ref: str,
    timeout_s: int,
    workspace: Path,
    config_path: Path,
) -> dict[str, Any]:
    attachments = payload_gateway_attachments(payload)
    if not attachments:
        raise RuntimeError("OpenClaw Gateway media edit mode found no image attachments.")
    return run_gateway_agent(
        openclaw_bin=openclaw_bin,
        agent=agent,
        session_id=session_id,
        payload=payload,
        model_ref=model_ref,
        timeout_s=timeout_s,
        workspace=workspace,
        config_path=config_path,
        attachments=attachments,
        include_images=False,
        message=build_gateway_media_edit_prompt(payload),
        param_overrides={
            "thinking": "off",
            "deliver": False,
            "extraSystemPrompt": build_gateway_media_edit_extra_prompt(),
        },
        metadata={
            "vision_mode": "gateway-edit",
            "attachment_count": len(attachments),
            "session_model_override": model_ref,
        },
    )


def run_gateway_text_agent(
    *,
    openclaw_bin: str,
    agent: str,
    session_id: str,
    payload: dict[str, Any],
    model_ref: str,
    timeout_s: int,
    workspace: Path,
    config_path: Path,
) -> dict[str, Any]:
    return run_gateway_agent(
        openclaw_bin=openclaw_bin,
        agent=agent,
        session_id=session_id,
        payload=payload,
        model_ref=model_ref,
        timeout_s=timeout_s,
        workspace=workspace,
        config_path=config_path,
        metadata={"text_mode": "gateway"},
    )


def post_text_chat_completion(
    payload: dict[str, Any],
    model_ref: str,
    timeout_s: int,
) -> dict[str, Any]:
    return post_text_prompt_chat_completion(
        payload=payload,
        model_ref=model_ref,
        prompt=build_prompt(payload),
        timeout_s=timeout_s,
        metadata={"text_mode": "api_fallback"},
    )


def post_text_prompt_chat_completion(
    *,
    payload: dict[str, Any],
    model_ref: str,
    prompt: str,
    timeout_s: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provider, model = model_parts(payload, model_ref)
    url, api_key = api_config(provider)
    model = api_model_name(provider, model)
    generation = payload.get("generation_config") or {}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": generation.get("temperature", 0.0),
        "max_tokens": int(generation.get("max_tokens") or 1024),
    }
    request = Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"text API HTTP {exc.code}: {detail[:1000]}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"text API request failed: {exc!r}") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError("text API returned non-object JSON")
    if parsed.get("error"):
        raise RuntimeError(f"text API error: {json.dumps(parsed.get('error'), ensure_ascii=False)[:1000]}")
    parsed.setdefault("openclaw", {})
    parsed["openclaw"].update(metadata or {})
    parsed["openclaw"]["model_ref"] = model_ref
    return parsed


TOKEN_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "total_tokens",
)


def int_usage_token(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def first_usage_token(usage: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        item: Any = usage
        for part in key.split("."):
            if not isinstance(item, dict):
                item = None
                break
            item = item.get(part)
        value = int_usage_token(item)
        if value is not None:
            return value
    return None


def normalize_usage_for_accounting(usage: Any) -> dict[str, int | None]:
    if not isinstance(usage, dict):
        return {field: None for field in TOKEN_USAGE_FIELDS}

    trajectory = {
        field: int_usage_token(usage.get("trajectory_" + field))
        for field in TOKEN_USAGE_FIELDS
    }
    if any(value is not None for value in trajectory.values()):
        return trajectory

    if isinstance(usage.get("agent_usage"), dict):
        return normalize_usage_for_accounting(usage["agent_usage"])

    input_tokens = first_usage_token(
        usage,
        "input_tokens",
        "prompt_tokens",
        "totalInput",
        "input",
    )
    output_tokens = first_usage_token(
        usage,
        "output_tokens",
        "completion_tokens",
        "output",
    )
    reasoning_tokens = first_usage_token(
        usage,
        "reasoning_tokens",
        "reasoningTokens",
        "reasoning",
        "completion_tokens_details.reasoning_tokens",
    )
    cache_read_tokens = first_usage_token(
        usage,
        "cache_read_tokens",
        "totalCacheRead",
        "cacheRead",
        "prompt_tokens_details.cached_tokens",
    )
    total_tokens = first_usage_token(usage, "total_tokens", "totalTokens", "total")
    billable_parts = [input_tokens, cache_read_tokens, output_tokens]
    if any(part is not None for part in billable_parts):
        billable_total = sum(part or 0 for part in billable_parts)
        if total_tokens is None or total_tokens < billable_total:
            total_tokens = billable_total

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cache_read_tokens": cache_read_tokens,
        "total_tokens": total_tokens,
    }


def sum_usage_tokens(*usages: Any) -> dict[str, int | None]:
    totals = {field: 0 for field in TOKEN_USAGE_FIELDS}
    seen = {field: False for field in TOKEN_USAGE_FIELDS}
    for usage in usages:
        normalized = normalize_usage_for_accounting(usage)
        for field in TOKEN_USAGE_FIELDS:
            value = normalized.get(field)
            if value is None:
                continue
            totals[field] += value
            seen[field] = True
    return {
        field: totals[field] if seen[field] else None
        for field in TOKEN_USAGE_FIELDS
    }


def extract_openclaw_agent_usage(data: dict[str, Any]) -> dict[str, Any]:
    inner = data.get("result", data)
    if not isinstance(inner, dict):
        return {}
    meta = inner.get("meta") if isinstance(inner.get("meta"), dict) else {}
    agent_meta = meta.get("agentMeta") if isinstance(meta.get("agentMeta"), dict) else {}
    return normalize_openclaw_agent_usage(agent_meta, meta) if agent_meta or meta else {}


def combine_forced_final_usage(
    *,
    agent_usage: dict[str, Any],
    api_usage: Any,
) -> dict[str, Any]:
    combined = sum_usage_tokens(agent_usage, api_usage)
    return {
        **combined,
        "usage_segments": [
            {
                "kind": "browsecomp_plus_agent_trajectory",
                "usage": agent_usage,
                **normalize_usage_for_accounting(agent_usage),
            },
            {
                "kind": "forced_final_api",
                "usage": api_usage if isinstance(api_usage, dict) else None,
                **normalize_usage_for_accounting(api_usage),
            },
        ],
    }


def should_preflight_browsecomp_plus_plugin() -> bool:
    return os.environ.get("OPENCLAW_BROWSECOMP_PLUS_PREFLIGHT", "1") != "0"


def should_restrict_browsecomp_plus_tools() -> bool:
    return os.environ.get("OPENCLAW_BROWSECOMP_PLUS_STRICT_TOOLS", "1") != "0"


def should_force_browsecomp_plus_final_on_budget() -> bool:
    return os.environ.get("OPENCLAW_BROWSECOMP_PLUS_FORCE_FINAL_ON_BUDGET", "1") != "0"


def should_deny_gateway_edit_apply_patch() -> bool:
    return os.environ.get("OPENCLAW_GATEWAY_EDIT_DENY_APPLY_PATCH", "0") == "1"


def should_isolate_runtime_config() -> bool:
    return os.environ.get("OPENCLAW_ROUTER_ISOLATE_CONFIG", "1") != "0"


def should_use_empty_workspace_for_prompt_only() -> bool:
    return os.environ.get("OPENCLAW_PROMPT_ONLY_EMPTY_WORKSPACE", "1") != "0"


def should_gateway_no_tool_vision() -> bool:
    return os.environ.get("OPENCLAW_GATEWAY_NO_TOOL_VISION", "0") == "1"


def is_code_execution_payload(payload: dict[str, Any]) -> bool:
    category = str(payload.get("category") or "")
    output_type = expected_output_type(payload)
    verifier = payload.get("verifier") if isinstance(payload.get("verifier"), dict) else {}
    verifier_type = str(verifier.get("type") or "")
    source_dataset = str(payload.get("source_dataset") or "")
    return (
        category == "code_debug_edit"
        or output_type
        in {
            "code_answer",
            "code_answer_or_patch",
            "workspace_patch_then_final_answer",
            "real_repo_edit_then_final_answer",
        }
        or verifier_type
        in {
            "unit_tests",
            "python_unittest_tests",
            "official_code_benchmark",
            "input_output_tests",
        }
        or source_dataset.startswith(("Muennighoff/mbpp", "codeparrot/apps", "princeton-nlp/SWE-bench"))
    )


def is_output_only_code_payload(payload: dict[str, Any]) -> bool:
    output_type = expected_output_type(payload)
    verifier = payload.get("verifier") if isinstance(payload.get("verifier"), dict) else {}
    verifier_type = str(verifier.get("type") or "")
    source_dataset = str(payload.get("source_dataset") or "")
    return (
        output_type in {"code_answer", "code_answer_or_patch"}
        and verifier_type
        in {
            "unit_tests",
            "python_unittest_tests",
            "input_output_tests",
            "official_code_benchmark",
        }
    ) or (
        source_dataset.startswith(("Muennighoff/mbpp", "codeparrot/apps"))
        and verifier_type
        in {
            "unit_tests",
            "python_unittest_tests",
            "input_output_tests",
        }
    )


def is_prompt_only_payload(payload: dict[str, Any]) -> bool:
    if payload_has_images(payload):
        return False
    if payload_needs_browsecomp_plus_plugin(payload) or is_gaia_or_livebrowse_payload(payload):
        return False
    output_type = expected_output_type(payload)
    if output_type in {"workspace_patch_then_final_answer", "real_repo_edit_then_final_answer"}:
        return False
    if expected_output_type(payload) == "openai_tool_calls_then_final_answer":
        return False
    return is_output_only_code_payload(payload) or is_no_tool_vision_payload(payload)


def is_gaia_or_livebrowse_payload(payload: dict[str, Any]) -> bool:
    output_type = expected_output_type(payload)
    source_dataset = str(payload.get("source_dataset") or "")
    style = str(payload.get("clawbench_style") or "")
    return (
        output_type == "gaia_agent_final_answer"
        or style in {"gaia_agent_final_answer", "livebrowsecomp_agent_final_answer"}
        or source_dataset in {"gaia-benchmark/GAIA", "Forival/LiveBrowseComp"}
    )


def gaia_payload_requires_web(payload: dict[str, Any]) -> bool:
    source_dataset = str(payload.get("source_dataset") or "")
    if source_dataset == "Forival/LiveBrowseComp":
        return True
    verifier = payload.get("verifier") if isinstance(payload.get("verifier"), dict) else {}
    reference = verifier.get("reference") if isinstance(verifier.get("reference"), dict) else {}
    return bool(reference.get("requires_websearch"))


def is_no_tool_vision_payload(payload: dict[str, Any]) -> bool:
    return expected_output_type(payload) == "vision_natural_language_answer" and not payload_declares_tools(payload)


def tool_allowlist_for_payload(payload: dict[str, Any]) -> set[str] | None:
    if os.environ.get("OPENCLAW_AUTO_TOOL_POLICY", "1") == "0":
        return None
    if payload_needs_browsecomp_plus_plugin(payload):
        return {BROWSECOMP_PLUS_TOOL_NAME}
    if (
        is_prompt_only_payload(payload)
        and os.environ.get("OPENCLAW_RESTRICT_PROMPT_ONLY_TOOLS", "1") != "0"
    ):
        return set()
    if (
        is_output_only_code_payload(payload)
        and os.environ.get("OPENCLAW_RESTRICT_OUTPUT_CODE_TOOLS", "1") != "0"
    ):
        return set()
    if (
        is_no_tool_vision_payload(payload)
        and os.environ.get("OPENCLAW_DENOISE_VISION_TOOLS", "1") != "0"
    ):
        return set()
    if is_code_execution_payload(payload):
        if os.environ.get("OPENCLAW_RESTRICT_CODE_TOOLS", "1") == "0":
            return None
        return set(CODE_EXECUTION_TOOLS)
    if is_gaia_or_livebrowse_payload(payload):
        if os.environ.get("OPENCLAW_RESTRICT_GAIA_TOOLS", "1") == "0":
            return None
        return set(WEB_AGENT_TOOLS if gaia_payload_requires_web(payload) else LOCAL_ANALYSIS_TOOLS)
    return None


def tool_denylist_for_payload(payload: dict[str, Any]) -> set[str]:
    denied: set[str] = set()
    if (
        is_output_only_code_payload(payload)
        and os.environ.get("OPENCLAW_RESTRICT_OUTPUT_CODE_TOOLS", "1") != "0"
    ):
        denied.update(CODE_EXECUTION_TOOLS)
    if (
        is_prompt_only_payload(payload)
        and os.environ.get("OPENCLAW_RESTRICT_PROMPT_ONLY_TOOLS", "1") != "0"
    ):
        denied.update(PROMPT_ONLY_TOOL_DENY)
        denied.update(NO_TOOL_NOISE_TOOLS)
    if os.environ.get("OPENCLAW_DENY_NETWORK_TOOLS_FOR_CODE", "1") != "0" and is_code_execution_payload(payload):
        denied.update(NETWORK_TOOLS)
    if os.environ.get("OPENCLAW_DENOISE_VISION_TOOLS", "1") != "0" and is_no_tool_vision_payload(payload):
        denied.update(PROMPT_ONLY_TOOL_DENY)
        denied.update(NO_TOOL_NOISE_TOOLS)
    if payload_needs_browsecomp_plus_plugin(payload):
        denied.difference_update({BROWSECOMP_PLUS_TOOL_NAME})
    return denied


def should_deny_network_tools_for_payload(payload: dict[str, Any]) -> bool:
    if os.environ.get("OPENCLAW_DENY_NETWORK_TOOLS_FOR_CODE", "1") == "0":
        return False
    if payload_needs_browsecomp_plus_plugin(payload):
        return False
    return is_code_execution_payload(payload)


def should_restrict_code_tools_for_payload(payload: dict[str, Any]) -> bool:
    if os.environ.get("OPENCLAW_RESTRICT_CODE_TOOLS", "1") == "0":
        return False
    if payload_needs_browsecomp_plus_plugin(payload):
        return False
    return is_code_execution_payload(payload)


def make_runtime_config(source_config_path: Path, workspace: Path, task_id: str, label: str) -> Path:
    task_slug = re.sub(r"[^A-Za-z0-9_.:-]+", "-", str(task_id or "task"))
    runtime_dir = Path(os.environ.get("OPENCLAW_RUNTIME_CONFIG_DIR") or workspace).expanduser()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = runtime_dir / (
        f".vlm-exec-routerbench-{label}-{task_slug}-{os.getpid()}-{uuid.uuid4().hex[:8]}.json"
    )
    write_text_atomic(runtime_config, source_config_path.read_text(encoding="utf-8"))
    return runtime_config.resolve()


def assert_browsecomp_plus_plugin_registered(
    *,
    openclaw_bin: str,
    timeout_s: int,
    workspace: Path,
) -> None:
    cmd = openclaw_cmd(openclaw_bin, "plugins", "inspect", BROWSECOMP_PLUS_PLUGIN_ID, "--json")
    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=max(30, min(timeout_s, 60)),
        cwd=workspace,
        check=False,
    )
    raw = result.stdout if result.stdout.strip() else result.stderr
    detail = raw.strip()[:2000]
    if result.returncode != 0:
        raise RuntimeError(
            "BrowseComp-Plus OpenClaw plugin preflight failed while inspecting the plugin: "
            f"rc={result.returncode}: {detail}"
        )

    parsed = extract_json_object(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"BrowseComp-Plus plugin inspect returned non-object output: {detail}")
    plugin = parsed.get("plugin") if isinstance(parsed.get("plugin"), dict) else {}

    def inspect_summary() -> str:
        summary = {
            "source": plugin.get("source"),
            "rootDir": plugin.get("rootDir"),
            "status": plugin.get("status"),
            "imported": plugin.get("imported"),
            "pluginToolNames": plugin.get("toolNames"),
            "runtimeTools": tool_names,
            "contractTools": contract_tools,
            "compat": plugin.get("compat") or plugin.get("compatibility"),
            "diagnostics": parsed.get("diagnostics"),
        }
        return json.dumps(summary, ensure_ascii=False)[:2000]

    if plugin.get("id") != BROWSECOMP_PLUS_PLUGIN_ID:
        raise RuntimeError(
            "BrowseComp-Plus OpenClaw plugin inspect returned the wrong plugin. "
            f"Expected plugin id {BROWSECOMP_PLUS_PLUGIN_ID!r}. Raw inspect output: {detail}"
        )
    if plugin.get("status") != "loaded":
        raise RuntimeError(
            "BrowseComp-Plus OpenClaw plugin was discovered but is not loaded. "
            f"Status={plugin.get('status')!r}. Raw inspect output: {detail}"
        )
    expected_root = browsecomp_plus_plugin_dir()
    actual_root_text = str(plugin.get("rootDir") or plugin.get("source") or "").strip()
    actual_root = Path(actual_root_text).expanduser() if actual_root_text else None
    if actual_root is not None:
        try:
            actual_root = actual_root.resolve()
        except OSError:
            pass
        if actual_root != expected_root:
            raise RuntimeError(
                "BrowseComp-Plus OpenClaw plugin loaded from an unexpected directory. "
                f"Expected {expected_root}, got {actual_root}. Remove stale plugins.load.paths entries "
                "or unset OPENCLAW_BROWSECOMP_PLUS_PLUGIN_DIR. "
                f"Raw inspect output: {detail}"
            )

    tool_names = []
    for tool in parsed.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        names = tool.get("names")
        if isinstance(names, list):
            tool_names.extend(str(name) for name in names)
        elif tool.get("name"):
            tool_names.append(str(tool.get("name")))
    contract_tools = []
    contracts = plugin.get("contracts") if isinstance(plugin.get("contracts"), dict) else {}
    if isinstance(contracts.get("tools"), list):
        contract_tools.extend(str(name) for name in contracts["tools"])
    if BROWSECOMP_PLUS_TOOL_NAME in tool_names:
        return
    if BROWSECOMP_PLUS_TOOL_NAME in contract_tools and plugin.get("imported") is False:
        # Some OpenClaw builds keep `plugins inspect` on the manifest-inventory
        # path and do not import plugin modules there. `browsecomp_plus_plugin_dir`
        # already imports the same entry in a Node smoke test and verifies that
        # registerTool exposes the expected name, so let execution continue. The
        # post-run tool-summary guard still fails if the real tool is not used.
        return
    if BROWSECOMP_PLUS_TOOL_NAME not in tool_names:
        raise RuntimeError(
            "BrowseComp-Plus OpenClaw plugin loaded, but its runtime tool registry did not include "
            f"{BROWSECOMP_PLUS_TOOL_NAME!r}. Inspect summary: {inspect_summary()}. "
            "Sync external/openclaw_browsecomp_plus_tool/index.js and openclaw.plugin.json on the runner, "
            "and verify the plugin entry imports cleanly."
        )


def run_infer_model(
    *,
    openclaw_bin: str,
    payload: dict[str, Any],
    model_ref: str,
    timeout_s: int,
    local: bool,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt = build_prompt(payload, tool_prompt_mode="simulated")
    cmd = openclaw_cmd(
        openclaw_bin,
        "infer",
        "model",
        "run",
        "--model",
        model_ref,
        "--prompt",
        prompt,
        "--json",
    )
    if local:
        cmd.append("--local")
    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=timeout_s + 60,
        check=False,
    )
    raw = result.stdout if result.stdout.strip() else result.stderr
    if result.returncode != 0:
        raise RuntimeError(
            f"openclaw infer model run failed rc={result.returncode}: "
            f"{(result.stderr or result.stdout).strip()[:1000]}"
        )

    parsed = extract_json_object(raw)
    assistant_text, usage = extract_openclaw_text(parsed or {}, raw)
    assistant_text = normalize_assistant_text(assistant_text)
    error_message = openclaw_error_message(parsed or {}, assistant_text)
    if error_message:
        raise RuntimeError(f"OpenClaw infer returned error output: {error_message[:1000]}")
    message = {"role": "assistant", "content": assistant_text}
    if expected_output_type(payload) == "openai_tool_calls_then_final_answer":
        message = tool_message_from_text(assistant_text) or message
    return {
        "choices": [
            {
                "message": message,
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
        "openclaw": {
            "model_ref": model_ref,
            "text_mode": "infer",
            "raw_response": parsed,
            **(metadata or {}),
        },
    }


def run_cli_agent(
    *,
    openclaw_bin: str,
    agent: str,
    session_id: str,
    payload: dict[str, Any],
    model_ref: str,
    timeout_s: int,
    openclaw_timeout: int,
    workspace: Path,
    text_thinking: str,
    local: bool,
    pass_generation_args: bool,
    temperature: Any,
    max_tokens: int | None,
) -> dict[str, Any]:
    prompt = build_prompt(payload, tool_prompt_mode="simulated")

    cmd = openclaw_cmd(
        openclaw_bin,
        "agent",
        "--agent",
        agent,
        "--session-id",
        session_id,
        "--message",
        prompt,
        "--json",
        "--timeout",
        str(openclaw_timeout),
    )
    if model_ref:
        cmd.extend(["--model", model_ref])
    if text_thinking:
        cmd.extend(["--thinking", text_thinking])
    if local:
        cmd.append("--local")
    if pass_generation_args and temperature is not None:
        cmd.extend(["--temperature", str(temperature)])
    if pass_generation_args and max_tokens is not None:
        cmd.extend(["--max-tokens", str(max_tokens)])

    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=timeout_s + 60,
        cwd=workspace,
        check=False,
    )
    raw = result.stdout if result.stdout.strip() else result.stderr
    if result.returncode != 0:
        raise RuntimeError(
            f"openclaw agent failed rc={result.returncode}: "
            f"{(result.stderr or result.stdout).strip()[:1000]}"
        )

    parsed = extract_json_object(raw)
    if payload_needs_browsecomp_plus_plugin(payload):
        assert_browsecomp_plus_tool_used(parsed or {})
    nonfinal_error = openclaw_nonfinal_error_message(parsed or {})
    if nonfinal_error:
        raise_browsecomp_plus_recoverable_if_possible(
            payload=payload,
            data=parsed or {},
            message=f"OpenClaw returned non-final BrowseComp-Plus output after retrieval: {nonfinal_error[:1000]}",
            fallback_reason="browsecomp_plus_nonfinal_after_search",
        )
        raise RuntimeError(f"OpenClaw returned non-final output: {nonfinal_error[:1000]}")
    assistant_text, usage = extract_openclaw_text(parsed or {}, raw)
    error_message = openclaw_error_message(parsed or {}, assistant_text)
    if error_message:
        raise_browsecomp_plus_recoverable_if_possible(
            payload=payload,
            data=parsed or {},
            message=f"OpenClaw returned BrowseComp-Plus error output after retrieval: {error_message[:1000]}",
            fallback_reason="browsecomp_plus_agent_error_after_search",
        )
        raise RuntimeError(f"OpenClaw returned error output: {error_message[:1000]}")
    if payload_needs_browsecomp_plus_plugin(payload) and is_browsecomp_plus_pseudo_tool_intent(assistant_text):
        raise RuntimeError(
            "OpenClaw returned a text-only BrowseComp-Plus tool intent instead of executing the "
            f"real {BROWSECOMP_PLUS_TOOL_NAME} tool. Check plugin registration and tool availability."
        )
    if payload_needs_browsecomp_plus_plugin(payload) and is_browsecomp_plus_unavailable_text(assistant_text):
        raise RuntimeError(
            "OpenClaw reported that the real BrowseComp-Plus tool is unavailable during execution. "
            "Check that the plugin manifest includes kind=tool, the plugin is enabled in config, "
            f"and {BROWSECOMP_PLUS_TOOL_NAME} appears in the agent runtime tool list."
        )
    message = {"role": "assistant", "content": assistant_text}
    if expected_output_type(payload) == "openai_tool_calls_then_final_answer":
        message = tool_message_from_text(assistant_text) or message
    return {
        "choices": [
            {
                "message": message,
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
        "openclaw": {
            "session_id": session_id,
            "model_ref": model_ref,
            "text_mode": "cli",
            "raw_response": parsed,
        },
    }


def build_prompt(payload: dict[str, Any], *, include_images: bool = True, tool_prompt_mode: str = "simulated") -> str:
    executor_input = payload.get("executor_input") or {}
    lines = []
    output_type = expected_output_type(payload)
    is_tool_call_task = output_type == "openai_tool_calls_then_final_answer"
    verifier = payload.get("verifier") if isinstance(payload.get("verifier"), dict) else {}
    verifier_reference = verifier.get("reference") if isinstance(verifier.get("reference"), dict) else {}
    is_official_code_benchmark = verifier.get("type") == "official_code_benchmark"
    is_output_only_code = is_output_only_code_payload(payload)
    system = executor_input.get("system")
    if is_tool_call_task and tool_prompt_mode == "simulated":
        lines.append(
            "System instruction:\n"
            "This is a benchmark tool-call selection task. The listed functions below are simulated "
            "OpenAI-compatible tool schemas, not real OpenClaw local tools. Do not try to execute, "
            "invoke, inspect, verify, or finish any tool. Do not call OpenClaw's Finish/operator tool. "
            "Your job is only to choose the required tool calls and return them as plain assistant text JSON."
        )
    elif system:
        lines.append(f"System instruction:\n{system}")

    if is_output_only_code:
        benchmark = str(verifier_reference.get("benchmark") or "official code benchmark")
        test_kind = str(verifier_reference.get("test_kind") or "")
        guard = (
            "OpenClaw execution guard for this benchmark:\n"
            "Do not use tools, shell, file edit, file read, web, workspace inspection, or OpenClaw Finish/operator actions. "
            "Do not create, modify, or mention local workspace files. "
            "Solve from the prompt text only and return solution code as the final assistant text. "
            "No apology, no analysis, no test transcript, and no workspace-edit output."
        )
        if is_official_code_benchmark:
            guard += " Return exactly one fenced Python code block and no Markdown outside the code fence."
        if is_official_code_benchmark and benchmark == "livecodebench" and test_kind == "stdin":
            guard += " This is a stdin/stdout task; provide a complete program that reads stdin and prints stdout."
        elif is_official_code_benchmark and benchmark == "livecodebench" and test_kind == "functional":
            fn_name = str(verifier_reference.get("fn_name") or "").strip()
            if fn_name:
                guard += f" This is a function task; define `{fn_name}` exactly and do not add stdin/stdout handling."
        elif is_official_code_benchmark and benchmark == "bigcodebench":
            entry_point = str(verifier_reference.get("entry_point") or "").strip()
            if entry_point:
                guard += f" Define `{entry_point}` exactly."
        lines.append(guard)

    messages = executor_input.get("messages") or payload.get("messages") or []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        content = text_from_content(message.get("content"), payload, include_images=include_images)
        if content:
            lines.append(f"{role.capitalize()} message:\n{content}")

    tools = payload.get("tools") or executor_input.get("tools") or []
    if tools:
        if is_tool_call_task and tool_prompt_mode == "simulated":
            lines.append(
                "Available OpenAI-compatible tool schemas:\n"
                + json.dumps(tools, ensure_ascii=False, indent=2)
                + "\nReturn only JSON in this exact shape, with no Markdown, no prose, and no code fence:\n"
                '{\"tool_calls\":[{\"name\":\"tool_name\",\"arguments\":{\"schema_field\":\"value\"}}],'
                '\"final_answer\":\"brief final answer after the tool calls\"}\n'
                "Use the tool names exactly as provided in the schemas. Include every tool call needed "
                "to complete all user-requested actions. Fill arguments only with fields supported by "
                "the provided schema; if a schema has no properties, use an empty arguments object. "
                "Do not say a tool is unavailable. Do not ask clarifying questions or stop because real-world "
                "information is missing; these are simulated benchmark tasks, so select the closest applicable "
                "tool for each explicit action the user requested. Preserve the logical workflow order, where "
                "later calls depend on earlier calls. Do not call, invoke, or include OpenClaw's Finish tool; "
                "only include a tool named Finish if it is explicitly listed in the JSON schemas above."
            )
        else:
            if payload_needs_browsecomp_plus_plugin(payload):
                lines.append(
                    f"{BROWSECOMP_PLUS_TOOL_NAME} is a real OpenClaw tool for this task. "
                    "Call it through OpenClaw's tool-use mechanism before answering. "
                    "Pass top_k=5 unless you have a task-specific reason to request fewer results. "
                    "Choose a concise evidence-seeking query from the user question. If the first result set does "
                    "not identify the answer, make at most four refined follow-up searches. Do not make more than five "
                    "BrowseComp-Plus tool calls total. Use only the returned evidence, not internal knowledge, to "
                    f"infer the answer. Do not describe the tool call or return JSON with type/name "
                    f"{BROWSECOMP_PLUS_TOOL_NAME}."
                )
            lines.append(
                "Available tool schemas:\n"
                + json.dumps(tools, ensure_ascii=False, indent=2)
                + "\nIf tool calls are required, use OpenClaw's normal tool-use flow."
            )

    expected = payload.get("expected_output_format") or {}
    if expected:
        if output_type == "workspace_patch_then_final_answer":
            lines.append(
                "This benchmark task may describe a repository issue without mounting the real repository. "
                "Do not stop because local files are absent. Infer the smallest correct change from the "
                "issue text and return a concrete unified diff or precise patch-style code change, followed "
                "by a concise summary. Prefer file paths and changed lines that match the issue description."
            )
        elif output_type in {"code_answer", "code_answer_or_patch"}:
            if is_official_code_benchmark:
                lines.append(
                    "Return only the actual solution code in one fenced Python code block. Do not include "
                    "explanations, patch text, schema text, JSON wrappers, or workspace-edit output."
                )
            else:
                lines.append(
                    "Return the actual solution code or patch content directly. Keep explanations brief and "
                    "ensure the code block contains the complete runnable or replaceable answer. Do not echo "
                    "the expected-output schema or return a JSON wrapper."
                )
        elif output_type == "gaia_agent_final_answer":
            lines.append(
                "This is an actual agent task. Use available tools when needed to solve it, "
                "but keep browsing focused: treat 8 total web/search/fetch tool calls as the "
                "browsing budget. Answer as soon as the evidence is sufficient. If you reach "
                "the budget, do not keep browsing; provide the best final answer from the "
                "evidence already gathered. "
                "then return only the final answer string requested by the user. Do not include reasoning, "
                "Markdown, JSON, schema text, citations, or explanatory prose in the final response."
            )
        elif output_type == "vision_natural_language_answer":
            lines.append(
                "Return only the final requested answer. Do not include reasoning, Markdown, JSON, "
                "schema text, or explanatory prose."
            )
        elif payload_needs_browsecomp_plus_plugin(payload):
            lines.append(
                "Return only the final answer string from the retrieved evidence. Do not include reasoning, "
                "Markdown, JSON, schema text, citations, or explanatory prose."
            )
        else:
            lines.append("Expected final answer format:\n" + json.dumps(expected, ensure_ascii=False, indent=2))

    if is_tool_call_task and tool_prompt_mode == "simulated":
        lines.append(
            "Return the JSON object now as the assistant's final text. Do not use OpenClaw tools, "
            "do not call Finish, and do not wait for real tool results."
        )
    elif is_official_code_benchmark:
        lines.append("Return the single fenced Python code block now. Do not use tools or edit files.")
    elif output_type == "gaia_agent_final_answer":
        lines.append(
            "Return only the final answer. For GAIA browsing tasks, do not exceed 8 total "
            "web/search/fetch tool calls; once the budget is reached, answer using the best "
            "evidence already gathered."
        )
    elif payload_needs_browsecomp_plus_plugin(payload):
        lines.append(
            f"Use {BROWSECOMP_PLUS_TOOL_NAME} before answering, then return only the final answer string. "
            "If the first retrieved documents do not settle the question, search up to four more times with better evidence queries. "
            "Pass top_k=5 on each call and do not make more than five BrowseComp-Plus tool calls total. "
            "Each tool result tells you the call number. When the tool result says it is the final allowed search, "
            "do not call the tool again; you must provide the best final answer from the retrieved evidence. "
            "Prefer retrieved evidence over any remembered facts."
        )
    elif is_prompt_only_payload(payload):
        lines.append("Solve from the prompt text only and provide the final answer.")
    else:
        lines.append("Solve the task through OpenClaw and provide the final answer.")
    return "\n\n".join(lines).strip()


def normalize_tool_calls(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    calls = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        function = item.get("function") if isinstance(item.get("function"), dict) else None
        name = item.get("name") or (function or {}).get("name")
        arguments = item.get("arguments", (function or {}).get("arguments", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"_raw": arguments}
        if not name:
            continue
        calls.append(
            {
                "id": str(item.get("id") or f"call_{index}"),
                "type": "function",
                "function": {
                    "name": str(name),
                    "arguments": json.dumps(arguments or {}, ensure_ascii=False),
                },
            }
        )
    return calls


def tool_message_from_text(text: str) -> dict[str, Any] | None:
    parsed = extract_json_object(text)
    if not isinstance(parsed, dict):
        return None
    tool_calls = normalize_tool_calls(parsed.get("tool_calls"))
    if not tool_calls:
        return None
    final_answer = parsed.get("final_answer") or parsed.get("content") or ""
    return {
        "role": "assistant",
        "content": str(final_answer),
        "tool_calls": tool_calls,
    }


def is_browsecomp_plus_pseudo_tool_intent(text: str) -> bool:
    parsed = extract_json_object(text)
    if not isinstance(parsed, dict):
        return False
    name = parsed.get("type") or parsed.get("name")
    function = parsed.get("function") if isinstance(parsed.get("function"), dict) else {}
    name = name or function.get("name")
    return str(name or "").strip() == BROWSECOMP_PLUS_TOOL_NAME


def is_browsecomp_plus_unavailable_text(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    return (
        BROWSECOMP_PLUS_TOOL_NAME in normalized
        and bool(re.search(r"\b(?:not available|unavailable|isn't available|is not available)\b", normalized))
    )


def openclaw_tool_summary(data: dict[str, Any]) -> dict[str, Any] | None:
    inner = data.get("result", data)
    if not isinstance(inner, dict):
        return None
    meta = inner.get("meta") if isinstance(inner.get("meta"), dict) else {}
    summary = meta.get("toolSummary")
    return summary if isinstance(summary, dict) else None


def browsecomp_plus_has_retrieval_evidence(data: dict[str, Any]) -> bool:
    summary = openclaw_tool_summary(data)
    if isinstance(summary, dict):
        tool_names = [str(name) for name in summary.get("tools") or []]
        try:
            calls_count = int(summary.get("calls") or 0)
        except (TypeError, ValueError):
            calls_count = 0
        if BROWSECOMP_PLUS_TOOL_NAME in tool_names or calls_count > 0:
            return True
    return bool(browsecomp_plus_evidence_chunks(data, max_chunks=1))


def raise_browsecomp_plus_recoverable_if_possible(
    *,
    payload: dict[str, Any],
    data: dict[str, Any],
    message: str,
    fallback_reason: str,
) -> None:
    if not payload_needs_browsecomp_plus_plugin(payload):
        return
    if not browsecomp_plus_has_retrieval_evidence(data):
        return
    raise BrowseCompPlusRecoverableError(
        message,
        data=data,
        summary=openclaw_tool_summary(data) or {},
        fallback_reason=fallback_reason,
    )


def browsecomp_plus_failure_detail(data: Any) -> str:
    details: list[str] = []
    seen = 0

    def visit(value: Any, key_hint: str = "") -> None:
        nonlocal seen
        if len(details) >= 5 or seen > 500:
            return
        seen += 1
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, str(key))
            return
        if isinstance(value, list):
            for item in value:
                visit(item, key_hint)
            return
        if not isinstance(value, str):
            return
        text = " ".join(value.strip().split())
        if not text:
            return
        lowered = text.lower()
        key_lower = key_hint.lower()
        if text.startswith("System instruction:") or "Available tool schemas:" in text:
            return
        interesting_key = key_lower in {"error", "errors", "message", "summary", "stderr"}
        interesting_text = any(
            marker in lowered
            for marker in (
                BROWSECOMP_PLUS_TOOL_NAME,
                "browsecomp-plus",
                "search browsecomp plus",
                "bm25",
                "pyserini",
                "retriever",
                "query required",
                "index not found",
                " failed",
            )
        )
        if interesting_key or interesting_text:
            clipped = text[:500]
            if clipped not in details:
                details.append(clipped)

    visit(data)
    return " | ".join(details)[:1500]


def browsecomp_plus_budget_max_calls() -> int:
    return max(
        1,
        int(os.environ.get("OPENCLAW_BROWSECOMP_PLUS_MAX_TOOL_CALLS", str(BROWSECOMP_PLUS_DEFAULT_MAX_TOOL_CALLS))),
    )


def clip_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def render_browsecomp_plus_results(results: Any) -> str:
    if not isinstance(results, list):
        return ""
    chunks = []
    for doc in results:
        if not isinstance(doc, dict):
            continue
        header = (
            f"[{doc.get('rank', '?')}] docid={doc.get('docid', '')} "
            f"score={doc.get('score', '')} url={doc.get('url', '')}"
        ).strip()
        title = str(doc.get("title") or "").strip()
        body = str(doc.get("text") or doc.get("contents") or "").strip()
        chunks.append("\n".join(part for part in (header, title, body) if part))
    return "\n\n".join(chunks)


def browsecomp_plus_evidence_chunks(value: Any, *, max_chunks: int = 5) -> list[str]:
    chunks: list[str] = []
    seen: set[str] = set()
    visited = 0

    def add(text: str) -> None:
        text = clip_text(text, 16000)
        if text.startswith("System instruction:") or "Available tool schemas:" in text:
            return
        if not text or text in seen or len(chunks) >= max_chunks:
            return
        seen.add(text)
        chunks.append(text)

    def visit(item: Any, key_hint: str = "") -> None:
        nonlocal visited
        if len(chunks) >= max_chunks or visited > 2000:
            return
        visited += 1
        if isinstance(item, dict):
            details = item.get("details") if isinstance(item.get("details"), dict) else {}
            tool_name = str(
                item.get("toolName")
                or item.get("name")
                or item.get("tool")
                or details.get("tool")
                or ""
            )
            if tool_name == BROWSECOMP_PLUS_TOOL_NAME:
                rendered = render_browsecomp_plus_results(details.get("results") or item.get("results"))
                if rendered:
                    add(rendered)
                else:
                    content = item.get("content")
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                add(str(part.get("text") or ""))
                    elif isinstance(content, str):
                        add(content)
                return
            for key, child in item.items():
                visit(child, str(key))
            return
        if isinstance(item, list):
            for child in item:
                visit(child, key_hint)
            return
        if isinstance(item, str):
            text = item.strip()
            if "BrowseComp-Plus search call" in text:
                add(text)

    visit(value)
    return chunks


def openclaw_session_file_from_raw(data: dict[str, Any]) -> Path | None:
    inner = data.get("result", data)
    if not isinstance(inner, dict):
        return None
    meta = inner.get("meta") if isinstance(inner.get("meta"), dict) else {}
    agent_meta = meta.get("agentMeta") if isinstance(meta.get("agentMeta"), dict) else {}
    for value in (
        agent_meta.get("sessionFile"),
        meta.get("sessionFile"),
        inner.get("sessionFile"),
    ):
        if isinstance(value, str) and value.strip():
            path = Path(value).expanduser()
            if path.exists():
                return path
    return None


def openclaw_session_file_from_store(config_path: Path, agent: str, session_id: str) -> Path | None:
    store_path = resolve_openclaw_session_store_path(config_path, agent)
    try:
        store = read_json(store_path) if store_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(store, dict):
        return None
    entry = store.get(f"agent:{agent}:{session_id}")
    if not isinstance(entry, dict):
        return None
    value = entry.get("sessionFile")
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    return path if path.exists() else None


def read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return rows
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def extract_browsecomp_plus_evidence(
    data: dict[str, Any],
    *,
    config_path: Path,
    agent: str,
    session_id: str,
) -> list[str]:
    max_calls = browsecomp_plus_budget_max_calls()
    chunks = browsecomp_plus_evidence_chunks(data, max_chunks=max_calls)
    if len(chunks) >= max_calls:
        return chunks[:max_calls]

    session_file = openclaw_session_file_from_raw(data) or openclaw_session_file_from_store(config_path, agent, session_id)
    if session_file is None:
        return chunks
    for chunk in browsecomp_plus_evidence_chunks(read_jsonl_objects(session_file), max_chunks=max_calls):
        if chunk not in chunks:
            chunks.append(chunk)
        if len(chunks) >= max_calls:
            break
    return chunks[:max_calls]


def build_browsecomp_plus_forced_answer_prompt(payload: dict[str, Any], evidence_chunks: list[str]) -> str:
    lines = []
    executor_input = payload.get("executor_input") or {}
    system = executor_input.get("system")
    if system:
        lines.append(f"System instruction:\n{system}")

    messages = executor_input.get("messages") or payload.get("messages") or []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = text_from_content(message.get("content"), payload, include_images=False)
        if content:
            role = str(message.get("role") or "user").capitalize()
            lines.append(f"{role} message:\n{content}")

    evidence_text = "\n\n---\n\n".join(
        f"BrowseComp-Plus evidence chunk {index}:\n{chunk}"
        for index, chunk in enumerate(evidence_chunks, start=1)
    )
    lines.append(
        "Retrieved BrowseComp-Plus evidence from the completed search budget:\n"
        + evidence_text
    )
    lines.append(
        "Tools are disabled. Do not request or describe any tool call. "
        "Use only the evidence above and the user question. "
        "Return only the final answer string, with no reasoning, citations, Markdown, JSON, or extra prose. "
        "If the evidence is incomplete, give the best answer supported by the retrieved evidence."
    )
    return "\n\n".join(line for line in lines if line).strip()


def recover_browsecomp_plus_budget_answer(
    *,
    payload: dict[str, Any],
    model_ref: str,
    timeout_s: int,
    error: BrowseCompPlusRecoverableError,
    config_path: Path,
    agent: str,
    session_id: str,
) -> dict[str, Any]:
    evidence_chunks = extract_browsecomp_plus_evidence(
        error.data,
        config_path=config_path,
        agent=agent,
        session_id=session_id,
    )
    if not evidence_chunks:
        raise error
    prompt = build_browsecomp_plus_forced_answer_prompt(payload, evidence_chunks)
    agent_usage = extract_openclaw_agent_usage(error.data)
    normalized = post_text_prompt_chat_completion(
        payload=payload,
        model_ref=model_ref,
        prompt=prompt,
        timeout_s=timeout_s,
        metadata={
            "text_mode": "cli-then-api-forced-final",
            "fallback_reason": error.fallback_reason,
            "agent_attempt": {
                "error": str(error)[:4000],
                "session_id": session_id,
                "model_ref": model_ref,
                "tool_summary": error.summary,
                "evidence_chunks": len(evidence_chunks),
                "usage": agent_usage,
            },
        },
    )
    api_usage = normalized.get("usage")
    normalized["usage"] = combine_forced_final_usage(
        agent_usage=agent_usage,
        api_usage=api_usage,
    )
    normalized.setdefault("openclaw", {})
    normalized["openclaw"]["usage_accounting"] = "browsecomp_plus_agent_plus_forced_final_api"
    return normalized


def assert_browsecomp_plus_tool_used(data: dict[str, Any]) -> None:
    summary = openclaw_tool_summary(data)
    if not summary:
        evidence = browsecomp_plus_evidence_chunks(data, max_chunks=1)
        if evidence:
            return
        raise RuntimeError(
            "OpenClaw completed a BrowseComp-Plus task without evidence that "
            f"{BROWSECOMP_PLUS_TOOL_NAME} executed: missing toolSummary and retrieved evidence."
        )
    tool_names = [str(name) for name in summary.get("tools") or []]
    calls_count = int(summary.get("calls") or 0)
    max_calls = browsecomp_plus_budget_max_calls()
    if calls_count > max_calls:
        summary_detail = json.dumps(summary, ensure_ascii=False)[:2000]
        raise BrowseCompPlusBudgetExceeded(
            "OpenClaw exceeded the BrowseComp-Plus tool-call budget: "
            f"{calls_count} call(s), max {max_calls}. "
            "The wrapper prompt allows only bounded query refinement to avoid tool loops and context blowups. "
            f"Tool summary: {summary_detail}",
            data=data,
            summary=summary,
        )
    failures = int(summary.get("failures") or 0)
    if failures and failures >= calls_count:
        summary_detail = json.dumps(summary, ensure_ascii=False)[:2000]
        failure_detail = browsecomp_plus_failure_detail(data)
        raise RuntimeError(
            "OpenClaw executed "
            f"{BROWSECOMP_PLUS_TOOL_NAME}, but all {failures} tool call(s) failed. "
            "Check the BrowseComp-Plus retriever runtime, including python, pyserini, BM25 index path, "
            "tool arguments, and plugin config. "
            + (f"Failure detail: {failure_detail}. " if failure_detail else "")
            + f"Tool summary: {summary_detail}"
        )
    if BROWSECOMP_PLUS_TOOL_NAME in tool_names:
        return
    calls = summary.get("calls")
    if calls:
        raise RuntimeError(
            "OpenClaw completed a BrowseComp-Plus task without executing "
            f"{BROWSECOMP_PLUS_TOOL_NAME}. Runtime tools used: {', '.join(tool_names) or 'none'}."
        )


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    def parse_object(value_text: str) -> dict[str, Any] | None:
        try:
            value = json.loads(value_text)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(value_text)
            except (SyntaxError, ValueError):
                return None
        return value if isinstance(value, dict) else None

    parsed = parse_object(text)
    if parsed is not None:
        return parsed

    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                value = parse_object(text[start : index + 1])
                if value is None:
                    break
                return value
    # OpenClaw can emit logs before JSON. Try every plausible suffix start.
    for match in re.finditer(r"\{", text):
        value = parse_object(text[match.start() :])
        if value is not None:
            return value
    return None


def normalize_openclaw_agent_usage(agent_meta: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    usage = agent_meta.get("usage") if isinstance(agent_meta.get("usage"), dict) else {}
    last_call_usage = agent_meta.get("lastCallUsage") if isinstance(agent_meta.get("lastCallUsage"), dict) else {}
    primary = usage or last_call_usage
    def token_total(value: dict[str, Any]) -> Any:
        total = value.get("total") or value.get("totalTokens")
        billable_parts = (
            value.get("totalInput") or value.get("input"),
            value.get("totalCacheRead") or value.get("cacheRead"),
            value.get("output"),
        )
        if any(part is not None for part in billable_parts):
            try:
                billable_total = sum(int(part or 0) for part in billable_parts)
            except (TypeError, ValueError):
                return total
            try:
                if total is None or int(total) < billable_total:
                    return billable_total
            except (TypeError, ValueError):
                return billable_total
        return total

    last_call_tokens = {
        "last_call_input_tokens": last_call_usage.get("totalInput") or last_call_usage.get("input"),
        "last_call_output_tokens": last_call_usage.get("output"),
        "last_call_reasoning_tokens": last_call_usage.get("reasoningTokens") or last_call_usage.get("reasoning"),
        "last_call_cache_read_tokens": last_call_usage.get("totalCacheRead") or last_call_usage.get("cacheRead"),
        "last_call_total_tokens": token_total(last_call_usage),
    }
    trajectory_tokens = {
        "trajectory_input_tokens": usage.get("totalInput") or usage.get("input"),
        "trajectory_output_tokens": usage.get("output"),
        "trajectory_reasoning_tokens": usage.get("reasoningTokens") or usage.get("reasoning"),
        "trajectory_cache_read_tokens": usage.get("totalCacheRead") or usage.get("cacheRead"),
        "trajectory_total_tokens": token_total(usage),
    }
    return {
        "input_tokens": primary.get("totalInput") or primary.get("input"),
        "output_tokens": primary.get("output"),
        "reasoning_tokens": primary.get("reasoningTokens") or primary.get("reasoning"),
        "cache_read_tokens": primary.get("totalCacheRead") or primary.get("cacheRead"),
        "total_tokens": token_total(primary),
        "model": agent_meta.get("model"),
        "duration_ms": meta.get("durationMs"),
        "agent_usage": usage,
        "last_call_usage": last_call_usage,
        **trajectory_tokens,
        **last_call_tokens,
    }


def extract_openclaw_text(data: dict[str, Any], fallback: str) -> tuple[str, dict[str, Any]]:
    inner = data.get("result", data)
    outputs = inner.get("outputs") if isinstance(inner, dict) else None
    if isinstance(outputs, list):
        parts = []
        for output in outputs:
            if isinstance(output, dict) and output.get("text"):
                parts.append(str(output["text"]))
        if parts:
            return "\n".join(parts), {}
    payloads = inner.get("payloads") if isinstance(inner, dict) else None
    if isinstance(payloads, list):
        parts = []
        for payload in payloads:
            if isinstance(payload, dict) and payload.get("text"):
                parts.append(str(payload["text"]))
        if parts:
            meta = inner.get("meta") if isinstance(inner.get("meta"), dict) else {}
            agent_meta = meta.get("agentMeta") if isinstance(meta.get("agentMeta"), dict) else {}
            return "\n".join(parts), normalize_openclaw_agent_usage(agent_meta, meta)
        meta = inner.get("meta") if isinstance(inner.get("meta"), dict) else {}
        agent_meta = meta.get("agentMeta") if isinstance(meta.get("agentMeta"), dict) else {}
        for key in ("finalAssistantVisibleText", "finalAssistantRawText"):
            value = meta.get(key)
            if value:
                return str(value), normalize_openclaw_agent_usage(agent_meta, meta)

    for key in ("response", "text", "content", "error"):
        value = inner.get(key) if isinstance(inner, dict) else None
        if value:
            return str(value), {}
    return fallback.strip(), {}


def normalize_assistant_text(text: str) -> str:
    parsed = extract_json_object(text)
    if isinstance(parsed, dict):
        if parsed.get("type") in {"agent_final_answer", "gaia_agent_final_answer", "vision_natural_language_answer"} and parsed.get("description") is not None:
            return str(parsed.get("description") or "").strip()
        result = parsed.get("result")
        if isinstance(result, dict):
            meta = result.get("meta")
            if isinstance(meta, dict):
                for key in ("finalAssistantVisibleText", "finalAssistantRawText"):
                    value = meta.get(key)
                    if value:
                        return str(value).strip()
    return text.strip()


def openclaw_nonfinal_error_message(data: dict[str, Any]) -> str | None:
    inner = data.get("result", data)
    if not isinstance(inner, dict):
        return None
    meta = inner.get("meta") if isinstance(inner.get("meta"), dict) else {}
    payloads = inner.get("payloads")
    payload_texts = []
    if isinstance(payloads, list):
        for payload in payloads:
            if isinstance(payload, dict) and str(payload.get("text") or "").strip():
                payload_texts.append(str(payload.get("text")).strip())
    has_visible_text = any(str(meta.get(key) or "").strip() for key in ("finalAssistantVisibleText", "finalAssistantRawText"))
    if payload_texts or has_visible_text:
        return None

    stop_reason = str(meta.get("stopReason") or "")
    liveness_state = str(meta.get("livenessState") or "")
    aborted = bool(meta.get("aborted"))
    if aborted or stop_reason == "toolUse" or liveness_state in {"blocked", "waiting", "working"}:
        return (
            f"no final assistant text; aborted={aborted}, "
            f"stopReason={stop_reason or 'unknown'}, livenessState={liveness_state or 'unknown'}"
        )
    if isinstance(payloads, list) and not payloads:
        return "no final assistant text; empty payloads"
    return None


def openclaw_error_message(data: dict[str, Any], assistant_text: str) -> str | None:
    inner = data.get("result", data)
    if isinstance(inner, dict):
        if inner.get("status") == "error":
            return str(inner.get("summary") or inner.get("error") or inner)
        error = inner.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("kind") or error)
        if error:
            return str(error)

    text = assistant_text.strip()
    lower_text = text.lower()
    if re.match(r"^HTTP\s+[45]\d\d\b", text, flags=re.IGNORECASE):
        return text
    if lower_text.startswith("context overflow"):
        return text
    if "agent couldn't generate a response" in lower_text:
        return text
    if lower_text in {"no_reply", "no reply from agent."}:
        return text
    if "image failed" in lower_text:
        return text
    if re.search(r"\bimage (?:is )?not (?:accessible|available|provided|uploaded)\b", lower_text):
        return text
    if "please provide a valid image" in lower_text or "upload the image again" in lower_text:
        return text
    if lower_text.startswith("llm request failed"):
        return text
    if "finish failed" in lower_text:
        return text
    if "request timed out before a response was generated" in lower_text:
        return text
    if re.search(
        r"\b(network connection error|connection error|provider error|api error|"
        r"rate limit|quota exhausted|unauthorized|authentication failed)\b",
        lower_text,
    ):
        return text
    return None


def patch_openclaw_model(
    config_path: Path,
    model_ref: str,
    workspace: Path | None = None,
    max_tokens: int | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    if not config_path.exists():
        raise FileNotFoundError(f"OpenClaw config not found: {config_path}")
    config = read_json(config_path)
    defaults = config.setdefault("agents", {}).setdefault("defaults", {})
    allowed_models = defaults.setdefault("models", {})
    if isinstance(allowed_models, dict) and model_ref not in allowed_models:
        allowed_models[model_ref] = {"alias": model_ref}
    model = defaults.setdefault("model", {})
    model["primary"] = model_ref
    model["fallbacks"] = []
    # Router tasks already provide the full prompt. Letting OpenClaw bootstrap
    # the large RouterSFT workspace can consume the model context before the
    # actual task starts, especially for smaller models.
    defaults["skipBootstrap"] = True
    if workspace is not None:
        defaults["workspace"] = str(workspace)
    if isinstance(payload, dict):
        generation = payload.get("generation_config") if isinstance(payload.get("generation_config"), dict) else {}
        timeout_value = generation.get("timeout")
        if timeout_value is not None:
            timeout_s = int(float(timeout_value))
            timeout_buffer_s = max(0, int(os.environ.get("OPENCLAW_AGENT_TIMEOUT_BUFFER", str(timeout_s))))
            defaults["timeoutSeconds"] = timeout_s + timeout_buffer_s
    if payload is not None and payload_needs_browsecomp_plus_plugin(payload):
        patch_browsecomp_plus_plugin_config(config, payload=payload)
    if "/" in model_ref:
        provider_name, model_id = model_ref.split("/", 1)
        providers = config.setdefault("models", {}).setdefault("providers", {})
        provider = providers.setdefault(provider_name, {})
        models = provider.setdefault("models", [])
        matched = False
        if isinstance(models, list):
            for entry in models:
                if isinstance(entry, dict) and entry.get("id") == model_id:
                    if max_tokens is not None:
                        entry["maxTokens"] = max_tokens
                    matched = True
                    break
            if not matched:
                patch = OPENCLAW_PROVIDER_MODEL_PATCHES.get((provider_name, model_id))
                if patch:
                    entry = dict(patch)
                    if max_tokens is not None:
                        entry["maxTokens"] = max_tokens
                    models.append(entry)
        # Some OpenClaw configs resolve providers through plugins or env-level defaults
        # without listing every model here. Updating provider metadata is best-effort.
    backup = config_path.with_suffix(config_path.suffix + ".router_sft_backup")
    if not backup.exists():
        write_text_atomic(backup, config_path.read_text(encoding="utf-8"))
    patch_openclaw_config_compat(config)
    write_json(config_path, config)


def append_unique(values: list[Any], value: str) -> None:
    if value not in [str(item) for item in values]:
        values.append(value)


def remove_browsecomp_plus_load_paths(values: list[Any]) -> None:
    values[:] = [
        value
        for value in values
        if Path(str(value)).name != BROWSECOMP_PLUS_DEFAULT_PLUGIN_DIR.name
    ]


def remove_tool_policy_values(values: list[Any], names: set[str]) -> None:
    values[:] = [value for value in values if str(value) not in names]


def patch_denied_tools(config: dict[str, Any], denied_tools: set[str]) -> None:
    tools_config = config.setdefault("tools", {})
    for key in ("allow", "alsoAllow"):
        if isinstance(tools_config.get(key), list):
            remove_tool_policy_values(tools_config[key], denied_tools)
    if not isinstance(tools_config.get("deny"), list):
        tools_config["deny"] = []
    for tool_name in sorted(denied_tools):
        append_unique(tools_config["deny"], tool_name)

    sandbox_tool_policy = tools_config.setdefault("sandbox", {}).setdefault("tools", {})
    for key in ("allow", "alsoAllow"):
        if isinstance(sandbox_tool_policy.get(key), list):
            remove_tool_policy_values(sandbox_tool_policy[key], denied_tools)
    if not isinstance(sandbox_tool_policy.get("deny"), list):
        sandbox_tool_policy["deny"] = []
    for tool_name in sorted(denied_tools):
        append_unique(sandbox_tool_policy["deny"], tool_name)


def patch_allowed_tools(config: dict[str, Any], allowed_tools: set[str]) -> None:
    allowed = sorted(allowed_tools)
    tools_config = config.setdefault("tools", {})
    tools_config["allow"] = allowed
    tools_config.pop("alsoAllow", None)
    # A broad profile can add high-noise tools before policy filtering. The
    # allowlist is the complete policy for benchmark code execution.
    tools_config.pop("profile", None)
    if isinstance(tools_config.get("deny"), list):
        remove_tool_policy_values(tools_config["deny"], allowed_tools)

    sandbox_tool_policy = tools_config.setdefault("sandbox", {}).setdefault("tools", {})
    sandbox_tool_policy["allow"] = allowed
    sandbox_tool_policy.pop("alsoAllow", None)
    sandbox_tool_policy.pop("profile", None)
    if isinstance(sandbox_tool_policy.get("deny"), list):
        remove_tool_policy_values(sandbox_tool_policy["deny"], allowed_tools)


def agent_tool_configs(config: dict[str, Any]) -> Iterable[dict[str, Any]]:
    agents = config.get("agents")
    if not isinstance(agents, dict):
        return []
    configs: list[dict[str, Any]] = []
    defaults = agents.get("defaults")
    if isinstance(defaults, dict):
        tools = defaults.setdefault("tools", {})
        if isinstance(tools, dict):
            configs.append(tools)
    entries = agents.get("list")
    if isinstance(entries, list):
        iterable_entries = entries
    elif isinstance(entries, dict):
        iterable_entries = list(entries.values())
    else:
        iterable_entries = []
    for entry in iterable_entries:
        if isinstance(entry, dict):
            tools = entry.setdefault("tools", {})
            if isinstance(tools, dict):
                configs.append(tools)
    return configs


def patch_prompt_only_tool_config(tools_config: dict[str, Any]) -> None:
    tools_config["allow"] = []
    tools_config.pop("alsoAllow", None)
    tools_config.pop("profile", None)
    if not isinstance(tools_config.get("deny"), list):
        tools_config["deny"] = []
    for tool_name in sorted(set(PROMPT_ONLY_TOOL_DENY) | set(NO_TOOL_NOISE_TOOLS)):
        append_unique(tools_config["deny"], tool_name)

    sandbox_tool_policy = tools_config.setdefault("sandbox", {}).setdefault("tools", {})
    sandbox_tool_policy["allow"] = []
    sandbox_tool_policy.pop("alsoAllow", None)
    sandbox_tool_policy.pop("profile", None)
    if not isinstance(sandbox_tool_policy.get("deny"), list):
        sandbox_tool_policy["deny"] = []
    for tool_name in sorted(set(PROMPT_ONLY_TOOL_DENY) | set(NO_TOOL_NOISE_TOOLS)):
        append_unique(sandbox_tool_policy["deny"], tool_name)


def patch_prompt_only_tools(config: dict[str, Any]) -> None:
    patch_prompt_only_tool_config(config.setdefault("tools", {}))
    for tools_config in agent_tool_configs(config):
        patch_prompt_only_tool_config(tools_config)


def patch_openclaw_config_compat(config: dict[str, Any]) -> None:
    plugins = config.get("plugins")
    if isinstance(plugins, dict) and isinstance(plugins.get("allow"), list) and plugins["allow"]:
        plugins.setdefault("bundledDiscovery", "compat")

    agents = config.get("agents")
    defaults = agents.get("defaults") if isinstance(agents, dict) else None
    if isinstance(defaults, dict) and isinstance(defaults.get("tools"), dict):
        # Current OpenClaw builds reject tools under agents.defaults. Keep the
        # benchmark policy in the top-level tools config instead.
        defaults.pop("tools", None)


def applied_tool_policy_metadata(payload: dict[str, Any], allowed_tools: set[str] | None, denied_tools: set[str]) -> dict[str, Any]:
    if os.environ.get("OPENCLAW_AUTO_TOOL_POLICY", "1") == "0":
        mode = "disabled"
    elif is_prompt_only_payload(payload) and os.environ.get("OPENCLAW_RESTRICT_PROMPT_ONLY_TOOLS", "1") != "0":
        mode = "prompt_only_no_tools"
    elif payload_needs_browsecomp_plus_plugin(payload):
        mode = "browsecomp_plus_only"
    elif is_no_tool_vision_payload(payload) and allowed_tools == set():
        mode = "vision_no_tools"
    elif allowed_tools is not None:
        mode = "allowlist"
    elif denied_tools:
        mode = "denylist"
    else:
        mode = "default"
    return {
        "tool_policy_mode": mode,
        "tool_policy_allowed": sorted(allowed_tools) if allowed_tools is not None else None,
        "tool_policy_denied": sorted(denied_tools),
    }


def patch_gateway_edit_tool_config(config: dict[str, Any]) -> None:
    patch_denied_tools(config, {"apply_patch"})


def should_isolate_gateway_runtime() -> bool:
    return os.environ.get("OPENCLAW_GATEWAY_ISOLATE_RUNTIME", "0") != "0"


def find_free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def patch_gateway_runtime_isolation(config: dict[str, Any], port: int, state_dir: Path | None = None) -> None:
    gateway = config.setdefault("gateway", {})
    if not isinstance(gateway, dict):
        gateway = {}
        config["gateway"] = gateway
    gateway["mode"] = "local"
    gateway["bind"] = "loopback"
    gateway["port"] = port
    # A runtime benchmark gateway must not proxy to a resident remote gateway.
    gateway.pop("remote", None)
    if state_dir is not None:
        configure_runtime_state(config, state_dir)


def validate_browsecomp_plus_plugin_dir(plugin_dir: Path) -> None:
    index_path = plugin_dir / "index.js"
    manifest_path = plugin_dir / "openclaw.plugin.json"
    if not index_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"BrowseComp-Plus OpenClaw plugin directory is incomplete: {plugin_dir}. "
            "Expected index.js and openclaw.plugin.json."
        )
    index_text = index_path.read_text(encoding="utf-8", errors="replace")
    if BROWSECOMP_PLUS_TOOL_NAME not in index_text or "registerTool" not in index_text:
        raise RuntimeError(
            f"BrowseComp-Plus OpenClaw plugin entry does not register {BROWSECOMP_PLUS_TOOL_NAME}: {index_path}"
        )
    if "export default" not in index_text:
        raise RuntimeError(
            f"BrowseComp-Plus OpenClaw plugin entry looks stale and lacks a default export: {index_path}. "
            "Unset OPENCLAW_BROWSECOMP_PLUS_PLUGIN_DIR or point it at this repository's updated "
            "external/openclaw_browsecomp_plus_tool directory."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"BrowseComp-Plus OpenClaw plugin manifest is invalid JSON: {manifest_path}") from exc
    activation = manifest.get("activation") if isinstance(manifest, dict) else None
    if isinstance(activation, dict) and activation.get("onCapabilities"):
        raise RuntimeError(
            "BrowseComp-Plus OpenClaw plugin manifest contains activation.onCapabilities, "
            "which can leave current OpenClaw builds with a loaded plugin but an empty runtime tool registry. "
            f"Sync the updated manifest without activation.onCapabilities: {manifest_path}"
        )
    node_bin = os.environ.get("OPENCLAW_NODE_BIN", "node")
    smoke_script = (
        "const mod = await import(process.argv[1]);"
        "const entry = mod.default ?? mod;"
        "const register = entry.register ?? mod.register;"
        "const names = [];"
        "const api = {"
        "  config: {},"
        "  pluginConfig: { repoRoot: process.cwd(), python: process.env.PYTHON || 'python3' },"
        "  registerTool(tool, opts) {"
        "    if (opts?.name) names.push(opts.name);"
        "    if (Array.isArray(opts?.names)) names.push(...opts.names);"
        "    if (tool && typeof tool !== 'function' && tool.name) names.push(tool.name);"
        "  }"
        "};"
        "if (typeof register !== 'function') throw new Error('missing register function');"
        "register(api);"
        f"if (!names.includes('{BROWSECOMP_PLUS_TOOL_NAME}')) "
        f"throw new Error('register did not expose {BROWSECOMP_PLUS_TOOL_NAME}; names=' + JSON.stringify(names));"
    )
    smoke = subprocess.run(
        [node_bin, "--input-type=module", "-e", smoke_script, str(index_path)],
        text=True,
        capture_output=True,
        timeout=20,
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )
    if smoke.returncode != 0:
        detail = (smoke.stderr or smoke.stdout).strip()
        raise RuntimeError(f"BrowseComp-Plus OpenClaw plugin registration smoke test failed: {detail[:1000]}")


def browsecomp_plus_plugin_dir() -> Path:
    configured = os.environ.get("OPENCLAW_BROWSECOMP_PLUS_PLUGIN_DIR")
    if configured:
        plugin_dir = Path(configured).expanduser().resolve()
        validate_browsecomp_plus_plugin_dir(plugin_dir)
        return plugin_dir

    candidates = [
        BROWSECOMP_PLUS_DEFAULT_PLUGIN_DIR,
        Path(__file__).resolve().parents[2] / "external" / "openclaw_browsecomp_plus_tool",
    ]
    last_error: Exception | None = None
    for plugin_dir in candidates:
        try:
            plugin_dir = plugin_dir.resolve()
            validate_browsecomp_plus_plugin_dir(plugin_dir)
            return plugin_dir
        except (FileNotFoundError, RuntimeError) as exc:
            last_error = exc
    raise FileNotFoundError(
        "BrowseComp-Plus OpenClaw plugin not found in any default location: "
        + ", ".join(str(path) for path in candidates)
        + (f". Last error: {last_error}" if last_error else "")
    )


def patch_browsecomp_plus_plugin_config(config: dict[str, Any], payload: dict[str, Any] | None = None) -> None:
    plugin_dir = browsecomp_plus_plugin_dir()
    if not plugin_dir.exists():
        raise FileNotFoundError(
            f"BrowseComp-Plus OpenClaw plugin not found: {plugin_dir}. "
            "Copy external/openclaw_browsecomp_plus_tool into this repo on the server, "
            "or set OPENCLAW_BROWSECOMP_PLUS_PLUGIN_DIR to the plugin directory."
        )
    plugins = config.setdefault("plugins", {})
    plugins["enabled"] = True
    plugins.pop("bundledDiscovery", None)
    # Only extend an existing non-empty allowlist. Creating plugins.allow here
    # would switch OpenClaw into allowlist mode and can hide bundled providers.
    if isinstance(plugins.get("allow"), list) and plugins["allow"]:
        for plugin_id, entry in list((plugins.get("entries") or {}).items()):
            if isinstance(entry, dict) and entry.get("enabled") is True:
                append_unique(plugins["allow"], str(plugin_id))
        append_unique(plugins["allow"], BROWSECOMP_PLUS_PLUGIN_ID)

    # OpenClaw normalizes config.plugins.load.paths into the loader's internal
    # loadPaths field. Keeping a raw plugins.loadPaths key makes config invalid
    # on current builds, so remove any value written by older wrapper versions.
    plugins.pop("loadPaths", None)
    if not isinstance(plugins.get("load"), dict):
        plugins["load"] = {}
    load = plugins["load"]
    if not isinstance(load.get("paths"), list):
        load["paths"] = []
    remove_browsecomp_plus_load_paths(load["paths"])
    append_unique(load["paths"], str(plugin_dir))

    if not isinstance(plugins.get("entries"), dict):
        plugins["entries"] = {}
    entries = plugins["entries"]
    entry = entries.setdefault(BROWSECOMP_PLUS_PLUGIN_ID, {})
    if not isinstance(entry, dict):
        entry = {}
        entries[BROWSECOMP_PLUS_PLUGIN_ID] = entry
    entry["enabled"] = True
    if not isinstance(entry.get("config"), dict):
        entry["config"] = {}
    plugin_config = entry["config"]
    # Runtime configs may be copied from a previous wrapper run. Do not preserve
    # stale repo/python values here; the retriever must run in the active env.
    plugin_config["repoRoot"] = str(Path(__file__).resolve().parents[1])
    plugin_config["python"] = (
        os.environ.get("OPENCLAW_BROWSECOMP_PLUS_PYTHON")
        or os.environ.get("BROWSECOMP_PLUS_PYTHON")
        or sys.executable
    )
    plugin_config["retriever"] = (
        os.environ.get("OPENCLAW_BROWSECOMP_PLUS_RETRIEVER")
        or os.environ.get("BROWSECOMP_PLUS_RETRIEVER")
        or "bm25"
    )
    retriever_server_url = (
        os.environ.get("OPENCLAW_BROWSECOMP_PLUS_RETRIEVER_SERVER_URL")
        or os.environ.get("BROWSECOMP_PLUS_RETRIEVER_SERVER_URL")
        or ""
    ).strip()
    if retriever_server_url:
        plugin_config["retrieverServerUrl"] = retriever_server_url
    else:
        plugin_config.pop("retrieverServerUrl", None)
    plugin_config.pop("queryOverride", None)
    plugin_config["maxDocChars"] = int(
        os.environ.get("OPENCLAW_BROWSECOMP_PLUS_MAX_DOC_CHARS", str(BROWSECOMP_PLUS_DEFAULT_MAX_DOC_CHARS))
    )
    plugin_config["maxCalls"] = int(
        os.environ.get("OPENCLAW_BROWSECOMP_PLUS_MAX_TOOL_CALLS", str(BROWSECOMP_PLUS_DEFAULT_MAX_TOOL_CALLS))
    )
    tools_config = config.setdefault("tools", {})
    sandbox_tool_policy = tools_config.setdefault("sandbox", {}).setdefault("tools", {})
    if should_restrict_browsecomp_plus_tools():
        tools_config["allow"] = [BROWSECOMP_PLUS_TOOL_NAME]
        tools_config["deny"] = ["image"]
        tools_config.pop("alsoAllow", None)
        tools_config.pop("profile", None)
        sandbox_tool_policy["allow"] = [BROWSECOMP_PLUS_TOOL_NAME]
        # OpenClaw adds image to explicit allowlists unless it is explicitly denied.
        sandbox_tool_policy["deny"] = ["image"]
        sandbox_tool_policy.pop("alsoAllow", None)
    else:
        if isinstance(tools_config.get("allow"), list) and tools_config["allow"]:
            append_unique(tools_config["allow"], BROWSECOMP_PLUS_TOOL_NAME)
        elif not isinstance(tools_config.get("alsoAllow"), list):
            tools_config["alsoAllow"] = []
        if not isinstance(tools_config.get("allow"), list) or not tools_config["allow"]:
            append_unique(tools_config["alsoAllow"], BROWSECOMP_PLUS_TOOL_NAME)

        if isinstance(sandbox_tool_policy.get("allow"), list) and sandbox_tool_policy["allow"]:
            append_unique(sandbox_tool_policy["allow"], BROWSECOMP_PLUS_TOOL_NAME)
        elif not isinstance(sandbox_tool_policy.get("alsoAllow"), list):
            sandbox_tool_policy["alsoAllow"] = []
        if not isinstance(sandbox_tool_policy.get("allow"), list) or not sandbox_tool_policy["allow"]:
            append_unique(sandbox_tool_policy["alsoAllow"], BROWSECOMP_PLUS_TOOL_NAME)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenClaw CLI adapter for generate_router_sft.py.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--openclaw-bin", default=os.environ.get("OPENCLAW_BIN", "openclaw"))
    parser.add_argument("--agent", default=os.environ.get("OPENCLAW_AGENT", "main"))
    parser.add_argument("--session-prefix", default="router-sft")
    parser.add_argument(
        "--session-id",
        default="",
        help="Reuse a specific OpenClaw session id. Useful for outer harness repair turns.",
    )
    parser.add_argument(
        "--model-ref-template",
        default=os.environ.get("OPENCLAW_MODEL_REF_TEMPLATE", ""),
        help=(
            "Optional model ref template. If the input payload already contains openclaw_model_ref, "
            "that value is preserved unless this template explicitly references {openclaw_model_ref}."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("OPENCLAW_CONFIG", "~/.openclaw/openclaw.json")).expanduser(),
    )
    parser.add_argument(
        "--pass-generation-args",
        action="store_true",
        help="Pass --temperature and --max-tokens through to `openclaw agent` if this OpenClaw version supports them.",
    )
    parser.add_argument(
        "--timeout-unit",
        choices=["seconds", "milliseconds"],
        default=os.environ.get("OPENCLAW_TIMEOUT_UNIT", "seconds"),
        help="Unit expected by `openclaw agent --timeout`. This OpenClaw CLI usually expects seconds.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Small workspace directory for OpenClaw. Defaults to ~/.openclaw/router_sft_workspace.",
    )
    parser.add_argument(
        "--vision-mode",
        choices=["auto", "gateway", "gateway-edit", "command", "api", "cli"],
        default=os.environ.get("OPENCLAW_VISION_MODE", "auto"),
        help=(
            "How to handle image inputs. auto first uses OpenClaw Gateway attachments, "
            "then --vision-command when configured, then the OpenAI-compatible provider API; "
            "gateway-edit uses Gateway attachments while preserving workspace editing; "
            "cli preserves the old flattened-text behavior."
        ),
    )
    parser.add_argument(
        "--vision-command",
        default=os.environ.get("OPENCLAW_VISION_COMMAND", ""),
        help=(
            "Command template for multimodal OpenClaw execution. Placeholders: "
            "{input}, {output}, {model}, {provider}, {model_ref}, {openclaw_model_ref}, "
            "{task_id}, {category}, {timeout}, {temperature}, {max_tokens}."
        ),
    )
    parser.add_argument(
        "--text-mode",
        choices=["gateway", "cli", "infer"],
        default=os.environ.get("OPENCLAW_TEXT_MODE", "cli"),
        help=(
            "How to handle text-only inputs. gateway calls OpenClaw Gateway agent "
            "directly; infer uses `openclaw infer model run`; cli preserves the "
            "legacy `openclaw agent --message` path."
        ),
    )
    parser.add_argument(
        "--tool-mode",
        choices=["infer", "agent", "agent-then-infer"],
        default=os.environ.get("OPENCLAW_TOOL_MODE", "infer"),
        help=(
            "How to handle simulated tool-call benchmark tasks. infer uses "
            "`openclaw infer model run` to avoid real OpenClaw tool execution; "
            "agent uses the original `openclaw agent` path; agent-then-infer "
            "captures an agent attempt, then falls back to infer if no tool_calls are produced."
        ),
    )
    parser.add_argument(
        "--text-thinking",
        default=os.environ.get("OPENCLAW_TEXT_THINKING", "off"),
        help="Thinking level passed to text-mode `openclaw agent --thinking`. Set empty to omit.",
    )
    parser.add_argument("--vision-retries", type=int, default=int(os.environ.get("OPENCLAW_VISION_RETRIES", "2")))
    parser.add_argument(
        "--vision-retry-sleep",
        type=float,
        default=float(os.environ.get("OPENCLAW_VISION_RETRY_SLEEP", "2.0")),
    )
    parser.add_argument(
        "--api-fallback-on-empty",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("OPENCLAW_API_FALLBACK_ON_EMPTY", "1") != "0",
        help="If OpenClaw CLI returns an empty/fallback assistant response, retry the same prompt through the provider API.",
    )
    parser.add_argument(
        "--reasoning-only-fallback",
        choices=["infer", "api", "off"],
        default=os.environ.get("OPENCLAW_REASONING_ONLY_FALLBACK", "infer"),
        help=(
            "Fallback for text-mode cli failures where OpenClaw agent reports repeated "
            "reasoning-only assistant turns. infer keeps the call inside OpenClaw CLI "
            "via `openclaw infer model run`; api calls the provider API directly; off "
            "surfaces the original agent failure."
        ),
    )
    parser.add_argument("--no-local", action="store_true", help="Do not pass --local to `openclaw agent`.")
    args = parser.parse_args()
    args.config = args.config.expanduser().resolve()
    source_config_path = args.config
    os.environ["OPENCLAW_CONFIG"] = str(args.config)
    os.environ["OPENCLAW_CONFIG_PATH"] = str(args.config)
    args.openclaw_bin = resolve_openclaw_bin(args.openclaw_bin)

    payload = read_json(args.input)
    model = str(payload.get("model") or "")
    provider = str(payload.get("provider") or "")
    payload_model_ref = str(payload.get("openclaw_model_ref") or "")
    model_ref = payload_model_ref
    generation = payload.get("generation_config") or {}
    timeout_s = int(float(generation.get("timeout") or 180))
    agent_timeout_buffer_s = max(0, int(os.environ.get("OPENCLAW_AGENT_TIMEOUT_BUFFER", str(timeout_s))))
    openclaw_agent_timeout_s = timeout_s + agent_timeout_buffer_s
    openclaw_timeout = (
        openclaw_agent_timeout_s * 1000
        if args.timeout_unit == "milliseconds"
        else openclaw_agent_timeout_s
    )
    temperature = generation.get("temperature")
    max_tokens = int(generation.get("max_tokens") or 0) or None

    workspace_was_explicit = args.workspace is not None
    workspace = args.workspace
    if workspace is None:
        workspace = args.config.parent / "router_sft_workspace"
        if is_prompt_only_payload(payload) and should_use_empty_workspace_for_prompt_only():
            task_slug = re.sub(r"[^A-Za-z0-9_.:-]+", "-", str(payload.get("task_id") or "task"))
            workspace = (
                args.config.parent
                / "router_sft_prompt_only_workspaces"
                / f"{task_slug}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            )
    workspace = workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    if (
        should_isolate_runtime_config()
        or (payload_needs_browsecomp_plus_plugin(payload) and should_restrict_browsecomp_plus_tools())
    ):
        args.config = make_runtime_config(
            source_config_path,
            workspace,
            str(payload.get("task_id", "task")),
            "runtime",
        )
        os.environ["OPENCLAW_CONFIG"] = str(args.config)
        os.environ["OPENCLAW_CONFIG_PATH"] = str(args.config)

    allowed_tools = tool_allowlist_for_payload(payload)
    denied_tools = tool_denylist_for_payload(payload)
    tool_policy_meta = applied_tool_policy_metadata(payload, allowed_tools, denied_tools)
    if allowed_tools is not None or denied_tools:
        config = read_json(args.config)
        if tool_policy_meta["tool_policy_mode"] == "prompt_only_no_tools":
            patch_prompt_only_tools(config)
        elif allowed_tools is not None:
            patch_allowed_tools(config, allowed_tools)
        if denied_tools and tool_policy_meta["tool_policy_mode"] != "prompt_only_no_tools":
            patch_denied_tools(config, denied_tools)
        patch_openclaw_config_compat(config)
        write_json(args.config, config)

    if args.model_ref_template and (not payload_model_ref or "{openclaw_model_ref}" in args.model_ref_template):
        model_ref = args.model_ref_template.format(
            model=model,
            provider=provider,
            openclaw_model_ref=payload_model_ref,
        )

    if not (is_prompt_only_payload(payload) and should_use_empty_workspace_for_prompt_only()):
        prepare_openclaw_workspace(workspace, source_config_path)

    session_id = str(args.session_id or "").strip()
    if not session_id:
        session_id = (
            f"{args.session_prefix}-{payload.get('task_id', 'task')}-{model}-"
            f"{os.getpid()}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        )
    session_id = re.sub(r"[^A-Za-z0-9_.:-]+", "-", session_id)

    has_images = payload_has_images(payload)
    if has_images and args.vision_mode == "gateway-edit":
        if not workspace_was_explicit:
            raise RuntimeError(
                "gateway-edit requires a real repository workspace. Pass --workspace /path/to/checkout "
                "or use scripts/run_swebench_openclaw_executor.py, which prepares the checkout and "
                "delegates with --workspace. Refusing to edit the default router_sft_workspace."
            )
        if should_deny_gateway_edit_apply_patch():
            runtime_config = make_runtime_config(
                source_config_path,
                workspace,
                str(payload.get("task_id", "task")),
                "gateway-edit",
            )
            config = read_json(source_config_path)
            allowed_tools = tool_allowlist_for_payload(payload)
            denied_tools = tool_denylist_for_payload(payload)
            if tool_policy_meta["tool_policy_mode"] == "prompt_only_no_tools":
                patch_prompt_only_tools(config)
            elif allowed_tools is not None:
                patch_allowed_tools(config, allowed_tools)
            if denied_tools and tool_policy_meta["tool_policy_mode"] != "prompt_only_no_tools":
                patch_denied_tools(config, denied_tools)
            patch_gateway_edit_tool_config(config)
            patch_openclaw_config_compat(config)
            write_json(runtime_config, config)
            args.config = runtime_config
            os.environ["OPENCLAW_CONFIG"] = str(args.config)
            os.environ["OPENCLAW_CONFIG_PATH"] = str(args.config)
        gateway_runtime_isolated = False
        if should_isolate_gateway_runtime():
            gateway_port = find_free_loopback_port()
            gateway_state_dir = runtime_state_dir_for_config(args.config)
            config = read_json(args.config)
            patch_gateway_runtime_isolation(config, gateway_port, gateway_state_dir)
            write_json(args.config, config)
            if gateway_state_dir is not None:
                os.environ["OPENCLAW_STATE_DIR"] = str(gateway_state_dir)
            tool_policy_meta["gateway_runtime_isolated"] = True
            tool_policy_meta["gateway_runtime_port"] = gateway_port
            if gateway_state_dir is not None:
                tool_policy_meta["gateway_runtime_state_dir"] = str(gateway_state_dir)
            gateway_runtime_isolated = True
        else:
            tool_policy_meta["gateway_runtime_isolated"] = False
            tool_policy_meta["tool_policy_audit_skipped_reason"] = "gateway_edit_resident_gateway_restore"
        if not model_ref:
            model_ref = model
        if model_ref:
            patch_openclaw_model(args.config, model_ref, workspace=workspace, max_tokens=max_tokens, payload=payload)
        try:
            normalized = run_gateway_media_edit_agent(
                openclaw_bin=args.openclaw_bin,
                agent=args.agent,
                session_id=session_id,
                payload=payload,
                model_ref=model_ref,
                timeout_s=timeout_s,
                workspace=workspace,
                config_path=args.config,
            )
        except RuntimeError as exc:
            if (
                not gateway_runtime_isolated
                or not should_fallback_gateway_edit_on_isolation_failure()
                or not is_retryable_gateway_transport_error(str(exc))
            ):
                raise
            fallback_config = make_runtime_config(
                source_config_path,
                workspace,
                str(payload.get("task_id", "task")),
                "gateway-edit-fallback",
            )
            config = read_json(fallback_config)
            allowed_tools = tool_allowlist_for_payload(payload)
            denied_tools = tool_denylist_for_payload(payload)
            if tool_policy_meta["tool_policy_mode"] == "prompt_only_no_tools":
                patch_prompt_only_tools(config)
            elif allowed_tools is not None:
                patch_allowed_tools(config, allowed_tools)
            if denied_tools and tool_policy_meta["tool_policy_mode"] != "prompt_only_no_tools":
                patch_denied_tools(config, denied_tools)
            patch_openclaw_config_compat(config)
            write_json(fallback_config, config)
            if model_ref:
                patch_openclaw_model(fallback_config, model_ref, workspace=workspace, max_tokens=max_tokens, payload=payload)

            saved_env = {
                "OPENCLAW_GATEWAY_ISOLATE_RUNTIME": os.environ.get("OPENCLAW_GATEWAY_ISOLATE_RUNTIME"),
                "OPENCLAW_ENFORCE_TOOL_POLICY_AUDIT": os.environ.get("OPENCLAW_ENFORCE_TOOL_POLICY_AUDIT"),
                "OPENCLAW_STATE_DIR": os.environ.get("OPENCLAW_STATE_DIR"),
                "OPENCLAW_CONFIG": os.environ.get("OPENCLAW_CONFIG"),
                "OPENCLAW_CONFIG_PATH": os.environ.get("OPENCLAW_CONFIG_PATH"),
            }
            try:
                os.environ["OPENCLAW_GATEWAY_ISOLATE_RUNTIME"] = "0"
                os.environ["OPENCLAW_ENFORCE_TOOL_POLICY_AUDIT"] = "0"
                os.environ.pop("OPENCLAW_STATE_DIR", None)
                os.environ["OPENCLAW_CONFIG"] = str(fallback_config)
                os.environ["OPENCLAW_CONFIG_PATH"] = str(fallback_config)
                normalized = run_gateway_media_edit_agent(
                    openclaw_bin=args.openclaw_bin,
                    agent=args.agent,
                    session_id=f"{session_id}-resident-fallback",
                    payload=payload,
                    model_ref=model_ref,
                    timeout_s=timeout_s,
                    workspace=workspace,
                    config_path=fallback_config,
                )
            finally:
                for key, value in saved_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            normalized.setdefault("openclaw", {})
            normalized["openclaw"]["gateway_edit_fallback_reason"] = "isolated_gateway_transport_error"
            normalized["openclaw"]["gateway_edit_isolation_error"] = str(exc)[:4000]
            normalized["openclaw"]["gateway_edit_fallback_config"] = str(fallback_config)
            tool_policy_meta["tool_policy_audit_skipped_reason"] = "gateway_edit_isolation_fallback_to_resident"
        attach_openclaw_metadata(normalized, tool_policy_meta)
        write_json(args.output, normalized)
        return

    skip_gateway_for_no_tool_vision = (
        has_images
        and args.vision_mode == "auto"
        and is_no_tool_vision_payload(payload)
        and not should_gateway_no_tool_vision()
    )
    if has_images and args.vision_mode in {"auto", "gateway"} and not skip_gateway_for_no_tool_vision:
        if not model_ref:
            model_ref = model
        if model_ref:
            patch_openclaw_model(args.config, model_ref, workspace=workspace, max_tokens=max_tokens, payload=payload)
        try:
            normalized = run_gateway_media_agent(
                openclaw_bin=args.openclaw_bin,
                agent=args.agent,
                session_id=session_id,
                payload=payload,
                model_ref=model_ref,
                timeout_s=timeout_s,
                workspace=workspace,
                config_path=args.config,
            )
            attach_openclaw_metadata(normalized, tool_policy_meta)
            write_json(args.output, normalized)
            return
        except Exception:
            if args.vision_mode == "gateway":
                raise

    use_vision_command = has_images and args.vision_mode == "command"
    if has_images and args.vision_mode == "auto" and args.vision_command:
        use_vision_command = True
    if use_vision_command:
        if not model_ref:
            model_ref = model
        normalized = run_vision_command(
            command_template=args.vision_command,
            input_path=args.input,
            output_path=args.output,
            payload=payload,
            model_ref=model_ref,
            timeout_s=timeout_s,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        attach_openclaw_metadata(normalized, tool_policy_meta)
        write_json(args.output, normalized)
        return

    use_vision_api = has_images and (args.vision_mode == "api" or args.vision_mode == "auto")
    if use_vision_api:
        if not model_ref:
            model_ref = model
        normalized = post_vision_chat_completion(
            payload,
            model_ref,
            timeout_s=timeout_s,
            retries=args.vision_retries,
            retry_sleep=args.vision_retry_sleep,
        )
        if skip_gateway_for_no_tool_vision:
            normalized.setdefault("openclaw", {})
            normalized["openclaw"]["vision_gateway_skipped_reason"] = "no_tool_vision_payload"
        attach_openclaw_metadata(normalized, tool_policy_meta)
        write_json(args.output, normalized)
        return

    if model_ref:
        patch_openclaw_model(args.config, model_ref, workspace=workspace, max_tokens=max_tokens, payload=payload)
    if (
        not has_images
        and payload_needs_browsecomp_plus_plugin(payload)
        and should_preflight_browsecomp_plus_plugin()
    ):
        assert_browsecomp_plus_plugin_registered(
            openclaw_bin=args.openclaw_bin,
            timeout_s=timeout_s,
            workspace=workspace,
        )

    if not has_images and args.text_mode == "gateway":
        normalized = run_gateway_text_agent(
            openclaw_bin=args.openclaw_bin,
            agent=args.agent,
            session_id=session_id,
            payload=payload,
            model_ref=model_ref or model,
            timeout_s=timeout_s,
            workspace=workspace,
            config_path=args.config,
        )
        attach_openclaw_metadata(normalized, tool_policy_meta)
        write_json(args.output, normalized)
        return

    if not has_images and args.text_mode == "infer":
        normalized = run_infer_model(
            openclaw_bin=args.openclaw_bin,
            payload=payload,
            model_ref=model_ref or model,
            timeout_s=timeout_s,
            local=not args.no_local,
            metadata={"text_mode": "infer"},
        )
        attach_openclaw_metadata(normalized, tool_policy_meta)
        write_json(args.output, normalized)
        return

    is_tool_call_task = not has_images and expected_output_type(payload) == "openai_tool_calls_then_final_answer"

    if is_tool_call_task and args.tool_mode == "infer":
        normalized = run_infer_model(
            openclaw_bin=args.openclaw_bin,
            payload=payload,
            model_ref=model_ref or model,
            timeout_s=timeout_s,
            local=not args.no_local,
            metadata={"tool_mode": "infer"},
        )
        attach_openclaw_metadata(normalized, tool_policy_meta)
        write_json(args.output, normalized)
        return

    agent_error_attempt = None
    cli_no_reply_retries = max(0, int(os.environ.get("OPENCLAW_CLI_NO_REPLY_RETRIES", "2")))
    try:
        normalized = None
        last_no_reply_error: RuntimeError | None = None
        for cli_attempt in range(cli_no_reply_retries + 1):
            attempt_session_id = session_id if cli_attempt == 0 else f"{session_id}-retry-{cli_attempt}"
            try:
                normalized = run_cli_agent(
                    openclaw_bin=args.openclaw_bin,
                    agent=args.agent,
                    session_id=attempt_session_id,
                    payload=payload,
                    model_ref=model_ref or model,
                    timeout_s=openclaw_agent_timeout_s,
                    openclaw_timeout=openclaw_timeout,
                    workspace=workspace,
                    text_thinking=args.text_thinking,
                    local=not args.no_local,
                    pass_generation_args=args.pass_generation_args,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if cli_attempt:
                    normalized.setdefault("openclaw", {})
                    normalized["openclaw"]["cli_retry_attempt"] = cli_attempt
                    normalized["openclaw"]["original_session_id"] = session_id
                break
            except RuntimeError as exc:
                if cli_attempt >= cli_no_reply_retries or not is_no_reply_agent_error(str(exc)):
                    raise
                last_no_reply_error = exc
                time.sleep(min(2.0 * (cli_attempt + 1), 8.0))
        if normalized is None and last_no_reply_error is not None:
            raise last_no_reply_error
    except RuntimeError as exc:
        error_text = str(exc)
        if (
            isinstance(exc, BrowseCompPlusRecoverableError)
            and not is_tool_call_task
            and should_force_browsecomp_plus_final_on_budget()
        ):
            if not model_ref:
                model_ref = model
            try:
                normalized = recover_browsecomp_plus_budget_answer(
                    payload=payload,
                    model_ref=model_ref,
                    timeout_s=timeout_s,
                    error=exc,
                    config_path=args.config,
                    agent=args.agent,
                    session_id=session_id,
                )
                attach_openclaw_metadata(normalized, tool_policy_meta)
                write_json(args.output, normalized)
                return
            except RuntimeError:
                raise exc
        if (
            not is_tool_call_task
            and args.reasoning_only_fallback != "off"
            and (is_reasoning_only_agent_error(error_text) or is_agent_request_schema_error(error_text))
        ):
            if not model_ref:
                model_ref = model
            fallback_reason = (
                "agent_request_schema_error"
                if is_agent_request_schema_error(error_text)
                else "reasoning_only_agent_error"
            )
            agent_attempt = {
                "error": error_text[:4000],
                "session_id": session_id,
                "model_ref": model_ref or model,
                "text_mode": "cli",
                "fallback_reason": fallback_reason,
            }
            if args.reasoning_only_fallback == "api":
                normalized = post_text_chat_completion(payload, model_ref, timeout_s=timeout_s)
                normalized.setdefault("openclaw", {})
                normalized["openclaw"]["text_mode"] = "cli-then-api"
                normalized["openclaw"]["agent_attempt"] = agent_attempt
            else:
                normalized = run_infer_model(
                    openclaw_bin=args.openclaw_bin,
                    payload=payload,
                    model_ref=model_ref or model,
                    timeout_s=timeout_s,
                    local=not args.no_local,
                    metadata={
                        "text_mode": "cli-then-infer",
                        "fallback_reason": fallback_reason,
                        "agent_attempt": agent_attempt,
                    },
                )
            attach_openclaw_metadata(normalized, tool_policy_meta)
            write_json(args.output, normalized)
            return
        if args.api_fallback_on_empty and "agent couldn't generate a response" in error_text.lower():
            if not model_ref:
                model_ref = model
            normalized = post_text_chat_completion(payload, model_ref, timeout_s=timeout_s)
            attach_openclaw_metadata(normalized, tool_policy_meta)
            write_json(args.output, normalized)
            return
        if not (is_tool_call_task and args.tool_mode == "agent-then-infer"):
            raise exc
        agent_error_attempt = {
            "error": error_text[:4000],
            "session_id": session_id,
            "model_ref": model_ref or model,
            "text_mode": "cli",
        }
        normalized = None

    if is_tool_call_task and args.tool_mode == "agent-then-infer" and (
        normalized is None or not response_from_has_tool_calls(normalized)
    ):
        if agent_error_attempt is None:
            agent_attempt = {
                "usage": normalized.get("usage"),
                "message": (normalized.get("choices") or [{}])[0].get("message") if normalized.get("choices") else None,
                **(normalized.get("openclaw") or {}),
            }
        else:
            agent_attempt = agent_error_attempt
        normalized = run_infer_model(
            openclaw_bin=args.openclaw_bin,
            payload=payload,
            model_ref=model_ref or model,
            timeout_s=timeout_s,
            local=not args.no_local,
            metadata={"tool_mode": "agent-then-infer", "agent_attempt": agent_attempt},
        )
    if normalized is not None:
        attach_openclaw_metadata(normalized, tool_policy_meta)
    write_json(args.output, normalized)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[run_openclaw_executor_error] {exc}", file=sys.stderr)
        raise
