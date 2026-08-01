import argparse
import ast
import base64
import hashlib
import json
import mimetypes
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from browsecomp_plus_retriever import search_browsecomp_plus


ROOT = Path(__file__).resolve().parents[1]
PROMPT_TEMPLATE_DIR = ROOT / "prompts" / "router_sft"
BROWSECOMP_PLUS_DATASET = "Tevatron/browsecomp-plus"
BROWSECOMP_PLUS_DEFAULT_TOP_K = 5
BROWSECOMP_PLUS_DEFAULT_MAX_DOC_CHARS = 2400
BROWSECOMP_PLUS_DEFAULT_MAX_TOOL_CALLS = 5
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_OPENROUTER_PRICE_CACHE = "runs/openrouter_model_prices.json"
SWEBENCH_LITE_DATASET = "princeton-nlp/SWE-bench_Lite"
SWEBENCH_MULTILINGUAL_DATASET = "SWE-bench/SWE-bench_Multilingual"
SWEBENCH_MULTIMODAL_DATASET = "princeton-nlp/SWE-bench_Multimodal"
SWEBENCH_SOURCE_DATASETS = {
    SWEBENCH_LITE_DATASET,
    SWEBENCH_MULTILINGUAL_DATASET,
    SWEBENCH_MULTIMODAL_DATASET,
}
SWEBENCH_MULTIMODAL_SBCLI_SUBSET = "swe-bench-m"
DEFAULT_MINI_AGENT_CONDA_ENV = os.environ.get("MINI_AGENT_CONDA_ENV", "swe-agent")
DEFAULT_SWEBENCH_CONDA_ENV = os.environ.get("SWEBENCH_CONDA_ENV", "swebench")
DEFAULT_OPENCLAW_COMMAND = "python3 scripts/run_openclaw_executor.py --input {input} --output {output}"
DEFAULT_SWEBENCH_OPENCLAW_COMMAND = "python3 scripts/run_swebench_openclaw_executor.py --input {input} --output {output}"
DEFAULT_MINI_AGENT_COMMAND = (
    "python3 scripts/run_mini_agent_executor.py --input {input} --output {output} "
    "--model {executor_model_ref} --temperature {temperature} --max-tokens {max_tokens} --timeout {timeout}"
)
DEFAULT_SWEBENCH_MINI_AGENT_COMMAND = (
    f"conda run -n {shlex.quote(DEFAULT_MINI_AGENT_CONDA_ENV)} python scripts/run_swebench_mini_agent_executor.py "
    "--input {input} --output {output} --model {executor_model_ref} --model-ref {executor_model_ref}"
)
DEFAULT_SWEBENCH_OFFICIAL_PYTHON = (
    os.environ.get("SWEBENCH_PYTHON")
    or f"conda run -n {shlex.quote(DEFAULT_SWEBENCH_CONDA_ENV)} python"
)

PROMPT_TEMPLATE_SPECS = {
    "common.available_tools": ("common.md", "available_tools"),
    "browsecomp_plus.system": ("browsecomp_plus.md", "system"),
    "browsecomp_plus.tool_policy": ("browsecomp_plus.md", "tool_policy"),
    "browsecomp_plus.refine_search": ("browsecomp_plus.md", "refine_search"),
    "browsecomp_plus.search_note": ("browsecomp_plus.md", "search_note"),
    "browsecomp_plus.final_search_note": ("browsecomp_plus.md", "final_search_note"),
}

PROMPT_TEMPLATE_ALIASES = {
    "available_tools.md": "common.available_tools",
    "browsecomp_plus_system.md": "browsecomp_plus.system",
    "browsecomp_plus_tool_policy.md": "browsecomp_plus.tool_policy",
    "browsecomp_plus_refine_search.md": "browsecomp_plus.refine_search",
    "browsecomp_plus_search_note.md": "browsecomp_plus.search_note",
    "browsecomp_plus_final_search_note.md": "browsecomp_plus.final_search_note",
}

PROMPT_TEMPLATE_FALLBACKS = {
    "common.available_tools": (
        "Available tools:\n"
        "The API tool schema is provided separately in the tools field. Use the tools when they are needed; "
        "do not invent unavailable tools or arguments.\n"
        "{tool_lines}"
    ),
    "browsecomp_plus.system": (
        "For BrowseComp-Plus tasks, use search_browsecomp_plus as an evidence tool. "
        "Do not answer from general knowledge or a merely similar document. If retrieved evidence does not "
        "explicitly identify the requested entity, use another search call with a narrower or different query. "
        "Use the available search budget before guessing. When you have enough evidence, output only the final "
        "answer string with no explanation, Markdown, citations, or prefix."
    ),
    "browsecomp_plus.tool_policy": (
        "BrowseComp-Plus tool policy:\n"
        "- You may run up to {max_loop_turns} search round(s). A round may include multiple query variants.\n"
        "- If retrieved documents do not explicitly answer the question, refine the query and call the tool again.\n"
        "- Do not answer from partial clues, general knowledge, or a similar but different document.\n"
        "- Final response must be only the answer string, with no explanation, Markdown, citation, or prefix."
    ),
    "browsecomp_plus.refine_search": (
        "The current retrieved evidence was insufficient, but search budget remains. "
        "Do not give a no-answer response yet. Call search_browsecomp_plus again with a "
        "different, narrower query based on the missing clue."
    ),
    "browsecomp_plus.search_note": (
        "BrowseComp-Plus search round {loop_turn}/{max_loop_turns}. You have {remaining_loop_turns} search round(s) remaining. "
        "This is search result {call_index}/{max_search_calls}. "
        "Answer now only if the retrieved documents explicitly identify the requested answer. "
        "If they only provide partial clues or a similar but different case, call search_browsecomp_plus again "
        "with a refined query."
    ),
    "browsecomp_plus.final_search_note": (
        "BrowseComp-Plus search round {loop_turn}/{max_loop_turns}. "
        "This is the final allowed search round. You must now answer using the retrieved evidence. "
        "Do not call search_browsecomp_plus again. Output only the exact answer string. "
        "Do not include explanation, Markdown, citations, confidence, prefixes, or caveats."
    ),
}

CANDIDATE_MODELS = (
    "qwen/qwen3-vl-8b-instruct",
    "qwen/qwen3.5-35b-a3b",
    "mistralai/mistral-small-2603",
    "gpt-5.4-mini",
    "anthropic/claude-haiku-4.5",
    "google/gemini-3-flash-preview",
    "openai/gpt-5.4",
    "anthropic/claude-sonnet-4.6",
    "google/gemini-2.5-flash-lite",
    "minimax/minimax-m3",
)

QWEN_MODELS = {
    "qwen3-vl-8b-instruct",
    "qwen3.5-35b-a3b",
}

GPT_MODELS = {
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5.5",
}

OPENROUTER_CATALOG_PREFIXES = {
    "anthropic",
    "google",
    "minimax",
    "meta-llama",
    "mistralai",
    "openai",
    "opengvlab",
    "qwen",
    "x-ai",
    "xiaomi",
    "z-ai",
}

DEEPSEEK_MODELS = {
    "deepseek-v4-pro",
    "deepseek-v4-flash",
}

DEFAULT_JUDGE_MODEL = "gpt-5.5"

BUDGET_POLICIES = {
    # Strong default for saving money: easy tasks only test cheap Qwen tiers,
    # medium tasks may escalate to open multimodal models, hard tasks can reach the full ladder.
    "adaptive": {"easy": 2, "medium": 5, "hard": len(CANDIDATE_MODELS)},
    # More aggressive savings. Useful after you already have enough high-end labels.
    "cheap": {"easy": 1, "medium": 2, "hard": 3},
    # Full cheapest-to-expensive cascade for every task.
    "full": {"easy": len(CANDIDATE_MODELS), "medium": len(CANDIDATE_MODELS), "hard": len(CANDIDATE_MODELS)},
    # Second-pass policy. After an adaptive run, only unsolved tasks are
    # escalated to the full ladder. Completed task/model pairs are skipped.
    "escalate-unsolved": {"easy": len(CANDIDATE_MODELS), "medium": len(CANDIDATE_MODELS), "hard": len(CANDIDATE_MODELS)},
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value).strip("_") or "unknown"


def provider_for_model(model: str) -> str:
    if "/" in model:
        provider, _ = model.split("/", 1)
        if provider in OPENROUTER_CATALOG_PREFIXES:
            return "openrouter"
        return provider
    if model in QWEN_MODELS:
        return "openrouter"
    if model in GPT_MODELS:
        return "openrouter"
    if model in DEEPSEEK_MODELS or model.startswith("deepseek"):
        return "deepseek"
    raise ValueError(f"Unknown candidate model: {model}")


def openrouter_model_name(model_id: str) -> str:
    env_name = "OPENROUTER_" + re.sub(r"[^A-Za-z0-9]+", "_", model_id).upper().strip("_") + "_MODEL"
    return os.environ.get(env_name, model_id)


def api_model_name(model: str) -> str:
    if "/" in model:
        provider, model_id = model.split("/", 1)
        if provider == "openrouter":
            if "/" not in model_id:
                return openrouter_model_name(f"openai/{model_id}")
            return openrouter_model_name(model_id)
        if provider in OPENROUTER_CATALOG_PREFIXES:
            return openrouter_model_name(model)
        return model_id
    if model in GPT_MODELS:
        return openrouter_model_name(f"openai/{model}")
    if model in QWEN_MODELS:
        return openrouter_model_name(f"qwen/{model}")
    return model


def load_openclaw_model_ref_map(value: str) -> dict[str, str]:
    if not value:
        return {}
    maybe_path = Path(value).expanduser()
    if not maybe_path.is_absolute():
        maybe_path = ROOT / maybe_path
    if maybe_path.exists():
        parsed = json.loads(maybe_path.read_text(encoding="utf-8"))
    else:
        parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--openclaw-model-ref-map must be a JSON object or a path to one.")
    return {str(key): str(item) for key, item in parsed.items()}


def openclaw_model_ref_for(model: str, model_ref_map: dict[str, str]) -> str:
    if model in model_ref_map:
        return model_ref_map[model]
    if provider_for_model(model) == "openrouter" and not model.startswith("openrouter/"):
        return f"openrouter/{api_model_name(model)}"
    return model


def executor_model_ref_for(model: str, model_ref_map: dict[str, str]) -> str:
    """Return the command-executor model ref; OpenClaw refs remain the legacy format."""
    return openclaw_model_ref_for(model, model_ref_map)


def extra_body_for_model(model: str, deepseek_thinking: str) -> dict[str, Any] | None:
    if provider_for_model(model) == "deepseek" and deepseek_thinking == "disabled":
        return {"thinking": {"type": "disabled"}}
    return None


def candidate_models_for_task(task: dict[str, Any], candidate_models: list[str], policy: str) -> list[str]:
    if policy not in BUDGET_POLICIES:
        raise ValueError(f"Unknown budget policy: {policy}")
    difficulty = str(task.get("difficulty_prior", "medium"))
    limit = BUDGET_POLICIES[policy].get(difficulty, len(candidate_models))
    return candidate_models[:limit]


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
            raise RuntimeError("Missing DEEPSEEK_API_KEY for DeepSeek judge model.")
        return base_url.rstrip("/") + "/chat/completions", api_key
    raise ValueError(provider)


def resolve_image_path(path: str) -> Path:
    image_path = Path(path)
    candidates = [image_path] if image_path.is_absolute() else [ROOT / image_path]
    if image_path.is_absolute():
        candidates.extend(
            [
                ROOT / image_path.name,
                ROOT / "raw_hf" / "mm" / "images" / image_path.name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def image_to_data_url(path: str) -> str:
    image_path = resolve_image_path(path)
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def convert_content(content: Any) -> Any:
    if not isinstance(content, list):
        return content

    converted = []
    for item in content:
        if not isinstance(item, dict):
            converted.append(item)
            continue
        if item.get("type") == "image_path":
            converted.append({"type": "image_url", "image_url": {"url": image_to_data_url(str(item["image_path"]))}})
        else:
            converted.append(item)
    return converted


def build_messages(executor_input: dict[str, Any]) -> list[dict[str, Any]]:
    messages = []
    system = executor_input.get("system")
    if system:
        messages.append({"role": "system", "content": str(system)})
    for message in executor_input.get("messages", []):
        item = dict(message)
        item["content"] = convert_content(item.get("content"))
        messages.append(item)
    return messages


def is_browsecomp_plus_task(task: dict[str, Any]) -> bool:
    return str(task.get("source_dataset") or "") == BROWSECOMP_PLUS_DATASET


def browsecomp_plus_query(task: dict[str, Any]) -> str:
    for message in task.get("executor_input", {}).get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "user":
            return message_text({"content": message.get("content")}) or str(message.get("content") or "")
    return str(task.get("router_view", {}).get("instruction") or "")


def expected_output_type(task: dict[str, Any]) -> str:
    expected = task.get("expected_output_format") or {}
    return str(expected.get("type") or "")


def is_function_io_task(task: dict[str, Any]) -> bool:
    verifier = task.get("verifier", {}) or {}
    reference = verifier.get("reference", {}) or {}
    input_output = reference.get("input_output", {}) or {}
    return verifier.get("type") == "input_output_tests" and bool(input_output.get("fn_name"))


def function_io_name(task: dict[str, Any]) -> str:
    verifier = task.get("verifier", {}) or {}
    reference = verifier.get("reference", {}) or {}
    input_output = reference.get("input_output", {}) or {}
    return str(input_output.get("fn_name") or "")


def official_code_benchmark_reference(task: dict[str, Any]) -> dict[str, Any]:
    verifier = task.get("verifier", {}) or {}
    reference = verifier.get("reference") if isinstance(verifier.get("reference"), dict) else {}
    if verifier.get("type") == "official_code_benchmark":
        return reference or {}
    return {}


def choice_lines_for_prompt(choices: dict[str, str]) -> str:
    return "\n".join(f"({key}) {value}" for key, value in choices.items())


def replace_or_append_choice_block(text: str, choices: dict[str, str]) -> str:
    if not choices:
        return text
    choice_block = "Choices:\n" + choice_lines_for_prompt(choices)
    if re.search(r"(?im)^\s*Choices\s*:", text):
        prefix = re.split(r"(?im)^\s*Choices\s*:\s*$", text, maxsplit=1)[0].rstrip()
        return f"{prefix}\n{choice_block}"
    return f"{text.rstrip()}\n{choice_block}"


def repair_executor_input_choices(task: dict[str, Any], executor_input: dict[str, Any]) -> None:
    if expected_output_type(task) != "vision_natural_language_answer":
        return
    choices = normalized_choices(verifier_reference(task))
    if not choices:
        return

    repaired_messages = []
    changed = False
    for message in executor_input.get("messages") or []:
        if not isinstance(message, dict):
            repaired_messages.append(message)
            continue
        repaired_message = dict(message)
        content = repaired_message.get("content")
        if isinstance(content, str):
            repaired_content = replace_or_append_choice_block(content, choices)
            changed = changed or repaired_content != content
            repaired_message["content"] = repaired_content
        elif isinstance(content, list):
            repaired_items = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    original = str(item.get("text") or "")
                    repaired_text = replace_or_append_choice_block(original, choices)
                    changed = changed or repaired_text != original
                    repaired_item = dict(item)
                    repaired_item["text"] = repaired_text
                    repaired_items.append(repaired_item)
                else:
                    repaired_items.append(item)
            repaired_message["content"] = repaired_items
        repaired_messages.append(repaired_message)
    if changed:
        executor_input["messages"] = repaired_messages


def compact_tool_description(value: Any, max_chars: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


class SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def prompt_template_key(name: str) -> str:
    return PROMPT_TEMPLATE_ALIASES.get(name, name)


def extract_prompt_template_section(text: str, section: str) -> str:
    pattern = re.compile(
        rf"(?ms)^<!--\s*template:\s*{re.escape(section)}\s*-->\s*\n?(.*?)\n?^<!--\s*/template\s*-->\s*$"
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def render_prompt_template(name: str, **values: Any) -> str:
    key = prompt_template_key(name)
    filename, section = PROMPT_TEMPLATE_SPECS.get(key, (key, ""))
    template_path = PROMPT_TEMPLATE_DIR / filename
    template = ""
    if template_path.exists():
        document = template_path.read_text(encoding="utf-8")
        template = extract_prompt_template_section(document, section) if section else document
    if not template:
        template = PROMPT_TEMPLATE_FALLBACKS.get(key, PROMPT_TEMPLATE_FALLBACKS.get(name, ""))
    return template.format_map(SafeFormatDict({key: str(value) for key, value in values.items()})).strip()


def tool_schema_signature(tool: dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    name = str(function.get("name") or "unnamed_tool")
    parameters = function.get("parameters") if isinstance(function.get("parameters"), dict) else {}
    properties = parameters.get("properties") if isinstance(parameters.get("properties"), dict) else {}
    required = set(parameters.get("required") or [])
    parts = []
    for param_name, spec in properties.items():
        if isinstance(spec, dict):
            param_type = spec.get("type") or "any"
            if isinstance(param_type, list):
                param_type = "|".join(str(item) for item in param_type)
        else:
            param_type = "any"
        suffix = "" if param_name in required else "?"
        parts.append(f"{param_name}{suffix}: {param_type}")
    return f"{name}({', '.join(parts)})"


def available_tools_contract(task: dict[str, Any], tools: list[Any]) -> str:
    normalized_tools = [tool for tool in tools if isinstance(tool, dict)]
    if not normalized_tools:
        return ""
    tool_lines = []
    for tool in normalized_tools:
        function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        description = compact_tool_description(function.get("description"))
        line = f"- {tool_schema_signature(tool)}"
        if description:
            line += f": {description}"
        tool_lines.append(line)
    tool_contract = render_prompt_template("common.available_tools", tool_lines="\n".join(tool_lines))
    blocks = [tool_contract]
    if is_browsecomp_plus_task(task):
        max_loop_turns = browsecomp_plus_max_loop_turns()
        blocks.append(
            render_prompt_template(
                "browsecomp_plus.tool_policy",
                max_loop_turns=max_loop_turns,
                max_calls=max_loop_turns,
            )
        )
    return "\n".join(block for block in blocks if block)


def executor_input_for_task(task: dict[str, Any], *, inject_tool_contract: bool = True) -> dict[str, Any]:
    executor_input = dict(task.get("executor_input", {}) or {})
    system = str(executor_input.get("system") or "").strip()
    if is_browsecomp_plus_task(task):
        browsecomp_instruction = render_prompt_template("browsecomp_plus.system")
        system = f"{system}\n{browsecomp_instruction}".strip()
    if expected_output_type(task) == "vision_natural_language_answer":
        concise_instruction = (
            "For visual QA tasks, output only the final requested answer: the number, "
            "choice label, short phrase, or entity. Do not include reasoning, derivations, "
            "Markdown, or explanatory text."
        )
        system = f"{system}\n{concise_instruction}".strip()
    if is_function_io_task(task):
        fn_name = function_io_name(task)
        function_instruction = (
            "For function implementation tasks, output only the required function or class definitions. "
            f"You must define the required callable exactly as `{fn_name}`. "
            "Do not rename it based on the problem title or your own interpretation. "
            "Do not include example calls, test code, print statements, stdin/stdout handling, "
            "or an if __name__ == '__main__' block."
        )
        system = f"{system}\n{function_instruction}".strip()
    official_reference = official_code_benchmark_reference(task)
    if official_reference:
        benchmark = str(official_reference.get("benchmark") or "code benchmark")
        test_kind = str(official_reference.get("test_kind") or "")
        code_instruction = (
            "For official code benchmark tasks, output exactly one fenced Python code block and nothing else. "
            "Do not include analysis, explanation, Markdown outside the code fence, example runs, test code, "
            "or prose saying the solution is complete. The verifier extracts the code block and submits it "
            f"to the {benchmark} official evaluator."
        )
        if benchmark == "livecodebench" and test_kind == "stdin":
            code_instruction += (
                " This is a stdin/stdout task: the code must be a complete runnable program that reads from "
                "sys.stdin or input() and prints the answer."
            )
        elif benchmark == "livecodebench" and test_kind == "functional":
            fn_name = str(official_reference.get("fn_name") or "").strip()
            if fn_name:
                code_instruction += (
                    f" This is a function task: define the required callable exactly as `{fn_name}` and do not "
                    "add stdin/stdout handling."
                )
        elif benchmark == "bigcodebench":
            entry_point = str(official_reference.get("entry_point") or "").strip()
            if entry_point:
                code_instruction += f" Define the required function exactly as `{entry_point}`."
        system = f"{system}\n{code_instruction}".strip()
    if inject_tool_contract:
        tool_contract = available_tools_contract(task, executor_input.get("tools") or [])
        if tool_contract:
            system = f"{system}\n\n{tool_contract}".strip()
    if system:
        executor_input["system"] = system
    repair_executor_input_choices(task, executor_input)
    return executor_input


def post_chat_completion(
    *,
    model: str,
    executor_input: dict[str, Any],
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    retry_sleep: float,
    http_transport: str,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provider = provider_for_model(model)
    url, api_key = api_config(provider)
    payload = {
        "model": api_model_name(model),
        "messages": build_messages(executor_input),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    tools = executor_input.get("tools") or []
    if tools:
        payload["tools"] = tools
    if extra_body:
        payload.update(extra_body)

    if http_transport == "curl":
        return post_chat_completion_curl(
            url=url,
            api_key=api_key,
            payload=payload,
            timeout=timeout,
            retries=retries,
            retry_sleep=retry_sleep,
        )

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    last_error = None
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {detail}"
            if exc.code < 500 and exc.code not in {408, 429}:
                break
        except (URLError, TimeoutError) as exc:
            last_error = repr(exc)
        if attempt < retries:
            time.sleep(retry_sleep * (attempt + 1))

    raise RuntimeError(last_error or "chat completion failed")


def post_chat_completion_messages(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: dict[str, Any] | str | None = None,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    retry_sleep: float,
    http_transport: str,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provider = provider_for_model(model)
    url, api_key = api_config(provider)
    payload = {
        "model": api_model_name(model),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if extra_body:
        payload.update(extra_body)

    if http_transport == "curl":
        return post_chat_completion_curl(
            url=url,
            api_key=api_key,
            payload=payload,
            timeout=timeout,
            retries=retries,
            retry_sleep=retry_sleep,
        )

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    last_error = None
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {detail}"
            if exc.code < 500 and exc.code not in {408, 429}:
                break
        except (URLError, TimeoutError) as exc:
            last_error = repr(exc)
        if attempt < retries:
            time.sleep(retry_sleep * (attempt + 1))

    raise RuntimeError(last_error or "chat completion failed")


def browsecomp_plus_no_answer_text(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if not normalized:
        return False
    markers = (
        "cannot be determined",
        "cannot definitively identify",
        "not possible to definitively identify",
        "not explicitly state",
        "not explicitly identify",
        "does not explicitly identify",
        "do not explicitly identify",
        "lack of explicit information",
        "based on the current evidence",
        "provided evidence",
        "no answer can be provided",
        "insufficient evidence",
        "reasonable to infer",
        "most likely candidate",
        "most likely answer",
    )
    return any(marker in normalized for marker in markers)


def browsecomp_plus_forced_tool_choice() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": "search_browsecomp_plus"},
    }


def browsecomp_plus_force_refine_search() -> bool:
    value = (
        os.environ.get("BROWSECOMP_PLUS_FORCE_REFINE_SEARCH")
        or os.environ.get("OPENCLAW_BROWSECOMP_PLUS_FORCE_REFINE_SEARCH")
        or "0"
    )
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def execute_browsecomp_plus_tool_call(call: dict[str, Any], default_query: str, default_top_k: int) -> dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    query = str(arguments.get("query") or default_query).strip()
    top_k = int(arguments.get("top_k") or arguments.get("topK") or default_top_k)
    max_doc_chars = int(os.environ.get("BROWSECOMP_PLUS_MAX_DOC_CHARS", str(BROWSECOMP_PLUS_DEFAULT_MAX_DOC_CHARS)))
    server_url = browsecomp_plus_retriever_server_url()
    if server_url:
        payload = search_browsecomp_plus_server(
            server_url,
            query=query,
            top_k=top_k,
            max_doc_chars=max_doc_chars,
            retriever=os.environ.get("BROWSECOMP_PLUS_RETRIEVER"),
        )
        return {
            "query": query,
            "top_k": top_k,
            "retriever": payload.get("retriever") or os.environ.get("BROWSECOMP_PLUS_RETRIEVER"),
            "retriever_server_url": server_url,
            "results": payload.get("results") if isinstance(payload.get("results"), list) else [],
        }
    docs = search_browsecomp_plus(query, top_k=top_k, max_doc_chars=max_doc_chars)
    return {
        "query": query,
        "top_k": top_k,
        "retriever": os.environ.get("BROWSECOMP_PLUS_RETRIEVER") or "bm25",
        "results": docs,
    }


def browsecomp_plus_retriever_server_url() -> str:
    configured = (
        os.environ.get("BROWSECOMP_PLUS_RETRIEVER_SERVER_URL")
        or os.environ.get("OPENCLAW_BROWSECOMP_PLUS_RETRIEVER_SERVER_URL")
        or ""
    )
    return configured.strip().rstrip("/")


def browsecomp_plus_retriever_server_timeout() -> float:
    configured = (
        os.environ.get("BROWSECOMP_PLUS_RETRIEVER_SERVER_TIMEOUT_MS")
        or os.environ.get("OPENCLAW_BROWSECOMP_PLUS_RETRIEVER_SERVER_TIMEOUT_MS")
        or "120000"
    )
    return max(1.0, float(configured) / 1000.0)


def search_browsecomp_plus_server(
    server_url: str,
    *,
    query: str,
    top_k: int,
    max_doc_chars: int,
    retriever: str | None,
) -> dict[str, Any]:
    body = json.dumps(
        {
            "query": query,
            "top_k": top_k,
            "max_doc_chars": max_doc_chars,
            "retriever": retriever or "qwen3-embedding-8b",
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        f"{server_url}/search",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=browsecomp_plus_retriever_server_timeout()) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"BrowseComp-Plus retriever server failed HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"BrowseComp-Plus retriever server request failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"BrowseComp-Plus retriever server returned non-object payload: {type(payload).__name__}")
    if payload.get("ok") is False:
        raise RuntimeError(f"BrowseComp-Plus retriever server error: {payload.get('error')}")
    return payload


def browsecomp_plus_max_tool_calls() -> int:
    return browsecomp_plus_max_loop_turns()


def browsecomp_plus_max_loop_turns() -> int:
    configured = (
        os.environ.get("BROWSECOMP_PLUS_MAX_AGENT_LOOP_TURNS")
        or os.environ.get("OPENCLAW_BROWSECOMP_PLUS_MAX_AGENT_LOOP_TURNS")
        or os.environ.get("BROWSECOMP_PLUS_MAX_TOOL_CALLS")
        or os.environ.get("OPENCLAW_BROWSECOMP_PLUS_MAX_TOOL_CALLS")
        or str(BROWSECOMP_PLUS_DEFAULT_MAX_TOOL_CALLS)
    )
    return max(1, int(configured))


def browsecomp_plus_max_search_calls(max_loop_turns: int) -> int:
    configured = (
        os.environ.get("BROWSECOMP_PLUS_MAX_SEARCH_CALLS")
        or os.environ.get("OPENCLAW_BROWSECOMP_PLUS_MAX_SEARCH_CALLS")
        or ""
    ).strip()
    if configured:
        return max(1, int(configured))
    calls_per_round = max(1, int(os.environ.get("BROWSECOMP_PLUS_MAX_SEARCH_CALLS_PER_ROUND", "4")))
    return max(1, max_loop_turns * calls_per_round)


def browsecomp_plus_tool_call_note(
    *,
    call_index: int,
    max_search_calls: int,
    loop_turn: int,
    max_loop_turns: int,
) -> str:
    remaining_loop_turns = max(0, max_loop_turns - loop_turn)
    if loop_turn >= max_loop_turns or call_index >= max_search_calls:
        return render_prompt_template(
            "browsecomp_plus.final_search_note",
            call_index=call_index,
            max_calls=max_loop_turns,
            max_search_calls=max_search_calls,
            loop_turn=loop_turn,
            max_loop_turns=max_loop_turns,
            remaining_loop_turns=remaining_loop_turns,
        )
    return render_prompt_template(
        "browsecomp_plus.search_note",
        call_index=call_index,
        max_calls=max_loop_turns,
        max_search_calls=max_search_calls,
        loop_turn=loop_turn,
        max_loop_turns=max_loop_turns,
        remaining=remaining_loop_turns,
        remaining_loop_turns=remaining_loop_turns,
    )


def browsecomp_plus_retrieved_docids(tool_results: list[dict[str, Any]]) -> list[str]:
    docids = {
        str(doc.get("docid"))
        for call in tool_results
        for doc in call.get("results", [])
        if isinstance(doc, dict) and doc.get("docid") not in (None, "")
    }
    return sorted(docids)


def run_browsecomp_plus_tool_loop(
    *,
    task: dict[str, Any],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    retry_sleep: float,
    http_transport: str,
    deepseek_thinking: str,
    capture_trajectory: bool = False,
) -> dict[str, Any]:
    executor_input = executor_input_for_task(task)
    messages = build_messages(executor_input)
    tools = executor_input.get("tools") or []
    default_query = browsecomp_plus_query(task)
    default_top_k = int(os.environ.get("BROWSECOMP_PLUS_TOP_K", str(BROWSECOMP_PLUS_DEFAULT_TOP_K)))
    max_loop_turns = browsecomp_plus_max_loop_turns()
    max_search_calls = browsecomp_plus_max_search_calls(max_loop_turns)
    tool_results = []
    requested_calls = 0
    truncated = False
    assistant_turns = 0
    tool_loop_turns = 0
    response: dict[str, Any] | None = None
    force_next_search = False
    usage_items = []

    while True:
        response = post_chat_completion_messages(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=browsecomp_plus_forced_tool_choice() if force_next_search else None,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries,
            retry_sleep=retry_sleep,
            http_transport=http_transport,
            extra_body=extra_body_for_model(model, deepseek_thinking),
        )
        usage_items.append(response.get("usage"))
        force_next_search = False
        assistant_turns += 1
        message = response_message(response)
        tool_calls = message.get("tool_calls") or []
        search_calls = [
            call
            for call in tool_calls
            if isinstance(call, dict)
            and isinstance(call.get("function"), dict)
            and call["function"].get("name") == "search_browsecomp_plus"
        ]
        if not search_calls:
            final_text = message_text({"content": message.get("content")}) if isinstance(message, dict) else ""
            if (
                tool_results
                and tool_loop_turns < max_loop_turns
                and len(tool_results) < max_search_calls
                and browsecomp_plus_no_answer_text(final_text)
            ):
                messages.append(message)
                messages.append(
                    {
                        "role": "user",
                        "content": render_prompt_template("browsecomp_plus.refine_search"),
                    }
                )
                force_next_search = browsecomp_plus_force_refine_search()
                continue
            break

        messages.append(message)
        tool_loop_turns += 1
        requested_calls += len(search_calls)
        for call in search_calls:
            if len(tool_results) >= max_search_calls:
                truncated = True
                break
            call_index = len(tool_results) + 1
            result = execute_browsecomp_plus_tool_call(call, default_query, default_top_k)
            result["call_index"] = call_index
            result["loop_turn"] = tool_loop_turns
            result["max_loop_turns"] = max_loop_turns
            result["max_calls"] = max_search_calls
            result["max_search_calls"] = max_search_calls
            result["instruction"] = browsecomp_plus_tool_call_note(
                call_index=call_index,
                max_search_calls=max_search_calls,
                loop_turn=tool_loop_turns,
                max_loop_turns=max_loop_turns,
            )
            tool_results.append(result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(call.get("id") or ""),
                    "name": "search_browsecomp_plus",
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

        if truncated or tool_loop_turns >= max_loop_turns or len(tool_results) >= max_search_calls:
            truncated = requested_calls > len(tool_results)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The BrowseComp-Plus agent-loop search budget is exhausted. "
                        "Do not call search_browsecomp_plus again. "
                        "Answer now using only the retrieved evidence. "
                        "Output only the exact answer string. Do not include explanation, "
                        "Markdown, citations, confidence, prefixes, or caveats."
                    ),
                }
            )
            response = post_chat_completion_messages(
                model=model,
                messages=messages,
                tools=None,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                retries=retries,
                retry_sleep=retry_sleep,
                http_transport=http_transport,
                extra_body=extra_body_for_model(model, deepseek_thinking),
            )
            usage_items.append(response.get("usage"))
            assistant_turns += 1
            break

    if response is None:
        response = post_chat_completion_messages(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries,
            retry_sleep=retry_sleep,
            http_transport=http_transport,
            extra_body=extra_body_for_model(model, deepseek_thinking),
        )
        usage_items.append(response.get("usage"))
    final_message = response_message(response)
    if final_message and not final_message.get("tool_calls") and tool_results:
        final_message["tool_calls"] = [
            {
                "id": f"browsecomp_plus_call_{call.get('call_index')}",
                "type": "function",
                "function": {
                    "name": "search_browsecomp_plus",
                    "arguments": json.dumps(
                        {"query": call.get("query"), "top_k": call.get("top_k")},
                        ensure_ascii=False,
                    ),
                },
            }
            for call in tool_results
        ]
    response["browsecomp_plus"] = {
        "tool": "search_browsecomp_plus",
        "mode": "agent_loop",
        "calls": tool_results,
        "requested_calls": requested_calls,
        "max_loop_turns": max_loop_turns,
        "max_search_calls": max_search_calls,
        "max_calls": max_search_calls,
        "truncated": truncated or requested_calls > len(tool_results),
        "retrieved_docids": browsecomp_plus_retrieved_docids(tool_results),
        "assistant_turns": assistant_turns,
        "loop_turns": tool_loop_turns,
        "retrievers": sorted(
            {
                str(doc.get("retriever"))
                for call in tool_results
                for doc in call.get("results", [])
                if isinstance(doc, dict) and doc.get("retriever")
            }
        ),
    }
    if capture_trajectory:
        response["browsecomp_plus_trajectory"] = {
            "messages": [*messages, final_message] if final_message else list(messages),
            "tools": tools,
            "tool_results": tool_results,
            "requested_calls": requested_calls,
            "max_loop_turns": max_loop_turns,
            "max_search_calls": max_search_calls,
            "max_calls": max_search_calls,
            "truncated": truncated or requested_calls > len(tool_results),
            "retrieved_docids": browsecomp_plus_retrieved_docids(tool_results),
            "assistant_turns": assistant_turns,
            "loop_turns": tool_loop_turns,
            "final_response": {
                key: value for key, value in response.items() if key != "browsecomp_plus_trajectory"
            },
        }
    agent_usage = aggregate_usage_items(usage_items)
    if agent_usage:
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        response["usage"] = {
            **usage,
            "agent_usage": agent_usage,
            "last_call_usage": usage,
        }
    return response


def run_browsecomp_plus_one_shot_tool_use(
    *,
    task: dict[str, Any],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    retry_sleep: float,
    http_transport: str,
    deepseek_thinking: str,
) -> dict[str, Any]:
    executor_input = executor_input_for_task(task)
    messages = build_messages(executor_input)
    tools = executor_input.get("tools") or []
    usage_items = []
    first = post_chat_completion_messages(
        model=model,
        messages=messages,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
        retry_sleep=retry_sleep,
        http_transport=http_transport,
        extra_body=extra_body_for_model(model, deepseek_thinking),
    )
    usage_items.append(first.get("usage"))
    first_message = response_message(first)
    tool_calls = first_message.get("tool_calls") or []
    search_calls = [
        call
        for call in tool_calls
        if isinstance(call, dict)
        and isinstance(call.get("function"), dict)
        and call["function"].get("name") == "search_browsecomp_plus"
    ]
    if not search_calls:
        agent_usage = aggregate_usage_items(usage_items)
        if agent_usage:
            usage = first.get("usage") if isinstance(first.get("usage"), dict) else {}
            first["usage"] = {
                **usage,
                "agent_usage": agent_usage,
                "last_call_usage": usage,
            }
        return first

    default_query = browsecomp_plus_query(task)
    default_top_k = int(os.environ.get("BROWSECOMP_PLUS_TOP_K", str(BROWSECOMP_PLUS_DEFAULT_TOP_K)))
    max_loop_turns = browsecomp_plus_max_loop_turns()
    max_search_calls = browsecomp_plus_max_search_calls(max_loop_turns)
    tool_results = []
    messages.append(first_message)
    for index, call in enumerate(search_calls[:max_search_calls], start=1):
        result = execute_browsecomp_plus_tool_call(call, default_query, default_top_k)
        result["call_index"] = index
        result["loop_turn"] = 1
        result["max_loop_turns"] = 1
        result["max_calls"] = max_search_calls
        result["max_search_calls"] = max_search_calls
        result["instruction"] = browsecomp_plus_tool_call_note(
            call_index=index,
            max_search_calls=max_search_calls,
            loop_turn=1,
            max_loop_turns=1,
        )
        tool_results.append(result)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": str(call.get("id") or ""),
                "name": "search_browsecomp_plus",
                "content": json.dumps(result, ensure_ascii=False),
            }
        )

    final = post_chat_completion_messages(
        model=model,
        messages=messages,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
        retry_sleep=retry_sleep,
        http_transport=http_transport,
        extra_body=extra_body_for_model(model, deepseek_thinking),
    )
    usage_items.append(final.get("usage"))
    final_message = response_message(final)
    if final_message and not final_message.get("tool_calls"):
        final_message["tool_calls"] = tool_calls
    final["browsecomp_plus"] = {
        "tool": "search_browsecomp_plus",
        "calls": tool_results,
        "requested_calls": len(search_calls),
        "max_loop_turns": 1,
        "max_search_calls": max_search_calls,
        "max_calls": max_search_calls,
        "truncated": len(search_calls) > max_search_calls,
        "retrievers": sorted(
            {
                str(doc.get("retriever"))
                for call in tool_results
                for doc in call.get("results", [])
                if isinstance(doc, dict) and doc.get("retriever")
            }
        ),
    }
    agent_usage = aggregate_usage_items(usage_items)
    if agent_usage:
        usage = final.get("usage") if isinstance(final.get("usage"), dict) else {}
        final["usage"] = {
            **usage,
            "agent_usage": agent_usage,
            "last_call_usage": usage,
        }
    return final


def run_raw_api_browsecomp_plus(**kwargs: Any) -> dict[str, Any]:
    """Backward-compatible alias for the BrowseComp-Plus search tool loop."""
    return run_browsecomp_plus_tool_loop(**kwargs)


def post_chat_completion_curl(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: int,
    retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False)
    last_error = None
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json") as body_file:
        body_file.write(body)
        body_file.flush()
        for attempt in range(retries + 1):
            result = subprocess.run(
                [
                    "curl",
                    "-sS",
                    "--max-time",
                    str(timeout),
                    "-H",
                    f"Authorization: Bearer {api_key}",
                    "-H",
                    "Content-Type: application/json",
                    "--data-binary",
                    f"@{body_file.name}",
                    url,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode == 0:
                try:
                    parsed = json.loads(result.stdout)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"curl returned non-JSON response: {result.stdout[:500]!r}") from exc
                if isinstance(parsed, dict) and parsed.get("error"):
                    last_error = f"API error: {json.dumps(parsed.get('error'), ensure_ascii=False)}"
                else:
                    return parsed
            else:
                last_error = result.stderr.strip() or result.stdout.strip() or f"curl exit {result.returncode}"
            if attempt < retries:
                time.sleep(retry_sleep * (attempt + 1))
    raise RuntimeError(last_error or "curl chat completion failed")


def openclaw_executor_payload(
    *,
    task: dict[str, Any],
    model: str,
    openclaw_model_ref: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    deepseek_thinking: str,
) -> dict[str, Any]:
    provider = provider_for_model(model)
    executor_input = executor_input_for_task(task, inject_tool_contract=False)
    return {
        "schema": "vlm_exec_routerbench_executor_v1",
        "model": model,
        "provider": provider,
        "openclaw_model_ref": openclaw_model_ref,
        "task_id": task.get("task_id"),
        "category": task.get("category"),
        "difficulty_prior": task.get("difficulty_prior"),
        "clawbench_style": task.get("clawbench_style"),
        "source_dataset": task.get("source_dataset"),
        "source_ref": task.get("source_ref"),
        "executor_input": executor_input,
        "messages": build_messages(executor_input),
        "tools": executor_input.get("tools") or [],
        "assets": executor_input.get("assets") or [],
        "router_view": task.get("router_view", {}),
        "expected_output_format": task.get("expected_output_format", {}),
        "verifier": task.get("verifier", {}),
        "generation_config": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "deepseek_thinking": deepseek_thinking,
        },
    }


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


def openclaw_response_from_output(value: Any) -> dict[str, Any]:
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

    raise RuntimeError("OpenClaw output must be a chat completion, message, assistant_text object, or text.")


def raw_model_response_for_row(response: dict[str, Any]) -> Any:
    """Return the backend-native response for per-result debugging."""
    openclaw_meta = response.get("openclaw")
    if isinstance(openclaw_meta, dict) and "raw_response" in openclaw_meta:
        return openclaw_meta.get("raw_response")
    return response


def openclaw_tool_summary(response: dict[str, Any]) -> dict[str, Any] | None:
    openclaw_meta = response.get("openclaw")
    if not isinstance(openclaw_meta, dict):
        return None
    raw_response = openclaw_meta.get("raw_response")
    if not isinstance(raw_response, dict):
        return None
    inner = raw_response.get("result") if isinstance(raw_response.get("result"), dict) else raw_response
    meta = inner.get("meta") if isinstance(inner, dict) and isinstance(inner.get("meta"), dict) else {}
    tool_summary = meta.get("toolSummary")
    return tool_summary if isinstance(tool_summary, dict) else None


def read_openclaw_output(path: Path, stdout: str) -> dict[str, Any]:
    text = ""
    if path.exists() and path.stat().st_size:
        text = path.read_text(encoding="utf-8").strip()
    if not text:
        text = stdout.strip()
    if not text:
        raise RuntimeError("OpenClaw command produced no output.")

    try:
        return openclaw_response_from_output(json.loads(text))
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

    return openclaw_response_from_output(text)


def subprocess_error_detail(stdout: str, stderr: str, fallback: str, max_chars: int = 3000) -> str:
    detail = stderr.strip() or stdout.strip() or fallback
    detail = re.sub(
        r"Error processing line 1 of .+?matplotlib-[^\n]+-nspkg\.pth:\n\n"
        r"  Traceback \(most recent call last\):\n"
        r"(?:    .+\n)+?"
        r"  AttributeError: 'NoneType' object has no attribute 'loader'\n\n"
        r"Remainder of file ignored\n?",
        "[python_startup_warning] ignored broken matplotlib namespace .pth\n",
        detail,
    )
    if len(detail) <= max_chars:
        return detail
    half = max_chars // 2
    return detail[:half].rstrip() + "\n...[truncated middle]...\n" + detail[-half:].lstrip()


def run_openclaw_executor(
    *,
    task: dict[str, Any],
    model: str,
    openclaw_model_ref: str,
    command_template: str,
    timeout: int,
    temperature: float,
    max_tokens: int,
    deepseek_thinking: str,
    keep_io: bool,
) -> dict[str, Any]:
    if not command_template:
        raise RuntimeError("--openclaw-command is required when --executor-backend=openclaw.")

    temp_dir_context = tempfile.TemporaryDirectory(prefix="openclaw_executor_")
    tmpdir = Path(temp_dir_context.name)
    input_path = tmpdir / "input.json"
    output_path = tmpdir / "output.json"
    payload = openclaw_executor_payload(
        task=task,
        model=model,
        openclaw_model_ref=openclaw_model_ref,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        deepseek_thinking=deepseek_thinking,
    )
    input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    format_values = {
        "input": str(input_path),
        "output": str(output_path),
        "model": model,
        "executor_model_ref": openclaw_model_ref,
        "openclaw_model_ref": openclaw_model_ref,
        "provider": provider_for_model(model),
        "task_id": str(task.get("task_id")),
        "category": str(task.get("category")),
        "timeout": str(timeout),
        "temperature": str(temperature),
        "max_tokens": str(max_tokens),
    }
    command = command_template.format(**format_values)
    result: subprocess.CompletedProcess[str] | None = None
    # Keep the model-facing timeout strict, but give the wrapper enough wall
    # time to start OpenClaw, run, and flush IO.
    command_timeout = timeout * 2 + 120
    try:
        for attempt in range(3):
            result = subprocess.run(
                shlex.split(command),
                text=True,
                capture_output=True,
                timeout=command_timeout,
                cwd=ROOT,
                check=False,
            )
            if result.returncode == 0:
                return read_openclaw_output(output_path, result.stdout)
            detail = subprocess_error_detail(result.stdout, result.stderr, f"exit {result.returncode}")
            if "session file locked" not in detail.lower() or attempt == 2:
                raise RuntimeError(f"OpenClaw command failed: {detail}")
            time.sleep(5 * (attempt + 1))
        raise RuntimeError("OpenClaw command failed after lock retries.")
    finally:
        if keep_io:
            keep_dir = ROOT / "runs" / "openclaw_io"
            keep_dir.mkdir(parents=True, exist_ok=True)
            stamp = int(time.time() * 1000)
            prefix = f"{task.get('task_id')}_{model}_{stamp}".replace("/", "_")
            (keep_dir / f"{prefix}.input.json").write_text(
                input_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            if output_path.exists():
                (keep_dir / f"{prefix}.output.json").write_text(
                    output_path.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            if result and result.stdout:
                (keep_dir / f"{prefix}.stdout.txt").write_text(result.stdout, encoding="utf-8")
            if result and result.stderr:
                (keep_dir / f"{prefix}.stderr.txt").write_text(result.stderr, encoding="utf-8")
        temp_dir_context.cleanup()


def run_executor(
    *,
    task: dict[str, Any],
    model: str,
    executor_backend: str,
    openclaw_command: str,
    openclaw_model_ref: str,
    openclaw_keep_io: bool,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    retry_sleep: float,
    http_transport: str,
    deepseek_thinking: str,
) -> dict[str, Any]:
    if executor_backend == "raw_api":
        if is_browsecomp_plus_task(task):
            return run_browsecomp_plus_tool_loop(
                task=task,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                retries=retries,
                retry_sleep=retry_sleep,
                http_transport=http_transport,
                deepseek_thinking=deepseek_thinking,
            )
        return post_chat_completion(
            model=model,
            executor_input=executor_input_for_task(task),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries,
            retry_sleep=retry_sleep,
            http_transport=http_transport,
            extra_body=extra_body_for_model(model, deepseek_thinking),
        )
    if executor_backend in {"openclaw", "mini_agent"}:
        return run_openclaw_executor(
            task=task,
            model=model,
            openclaw_model_ref=openclaw_model_ref,
            command_template=openclaw_command,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            deepseek_thinking=deepseek_thinking,
            keep_io=openclaw_keep_io,
        )
    raise ValueError(f"Unknown executor backend: {executor_backend}")


def smoke_executor_input(prompt: str) -> dict[str, Any]:
    return {
        "system": "You are a concise assistant.",
        "messages": [{"role": "user", "content": prompt}],
        "tools": [],
        "assets": [],
    }


def run_api_smoke(
    models: list[str],
    max_tokens: int,
    timeout: int,
    retries: int,
    retry_sleep: float,
    http_transport: str,
    deepseek_thinking: str,
) -> None:
    for model in models:
        provider = provider_for_model(model)
        url, _api_key = api_config(provider)
        print(f"[smoke] model={model} provider={provider} url={url}", flush=True)
        started = time.time()
        try:
            response = post_chat_completion(
                model=model,
                executor_input=smoke_executor_input("Reply with exactly: ok"),
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=timeout,
                retries=retries,
                retry_sleep=retry_sleep,
                http_transport=http_transport,
                extra_body=extra_body_for_model(model, deepseek_thinking),
            )
            choices = response.get("choices") or []
            finish_reason = choices[0].get("finish_reason") if choices else None
            message = response_message(response)
            text = message_text(message)
            reasoning = str(message.get("reasoning_content") or "")
            print(
                f"[smoke_done] model={model} latency_s={time.time() - started:.2f} "
                f"finish_reason={finish_reason!r} text={text[:80]!r} reasoning={reasoning[:80]!r}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[smoke_error] model={model} latency_s={time.time() - started:.2f} "
                f"error={exc!r}",
                flush=True,
            )


def response_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    if not choices:
        return {}
    return choices[0].get("message") or {}


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return ""


def parse_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments.strip())
            except json.JSONDecodeError:
                arguments = {"_raw": arguments}
        calls.append({"name": function.get("name"), "arguments": arguments})
    return calls


def result_tool_calls_for_row(message: dict[str, Any], tool_summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    calls = parse_tool_calls(message)
    if calls or not tool_summary:
        return calls

    tools = [str(name) for name in tool_summary.get("tools") or [] if str(name)]
    calls_count = int(tool_summary.get("calls") or 0)
    if not tools or calls_count <= 0:
        return []

    return [
        {
            "name": name,
            "arguments": {},
            "source": "openclaw_tool_summary",
            "call_count": calls_count if len(tools) == 1 else None,
        }
        for name in tools
    ]


def tool_schema_has_argument_fields(task: dict[str, Any], tool_name: Any) -> bool:
    expected_name = canonical_name(tool_name)
    for tool in task.get("executor_input", {}).get("tools") or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        if canonical_name(function.get("name")) != expected_name:
            continue
        parameters = function.get("parameters") if isinstance(function.get("parameters"), dict) else {}
        properties = parameters.get("properties") if isinstance(parameters.get("properties"), dict) else {}
        return bool(properties)
    return True


def canonical_name(value: Any) -> str:
    return str(value or "").strip().rsplit(".", 1)[-1].lower()


def normalize_arg_value(value: Any) -> Any:
    if isinstance(value, str):
        text = re.sub(r"\s+", " ", value.strip().lower())
        return text
    if isinstance(value, list):
        return [normalize_arg_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key).lower(): normalize_arg_value(item) for key, item in sorted(value.items())}
    return value


def flatten_args(value: Any) -> list[Any]:
    if value in (None, {}, []):
        return []
    if isinstance(value, dict):
        flattened = []
        for key in sorted(value):
            flattened.append((str(key).lower(), normalize_arg_value(value[key])))
        return flattened
    if isinstance(value, list):
        return [normalize_arg_value(item) for item in value]
    return [normalize_arg_value(value)]


def args_match(expected: Any, observed: Any) -> bool:
    expected_flat = flatten_args(expected)
    observed_flat = flatten_args(observed)
    if not expected_flat:
        return True
    if expected_flat == observed_flat:
        return True

    observed_blob = json.dumps(observed_flat, ensure_ascii=False, sort_keys=True)
    for item in expected_flat:
        if isinstance(item, tuple):
            _key, value = item
        else:
            value = item
        if isinstance(value, str) and value:
            if value not in observed_blob:
                return False
        elif value not in observed_flat:
            return False
    return True


def norm_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`]+|[\"'`.。]+$", "", text)
    text = re.sub(r"^(answer|final answer|答案)\s*[:：]\s*", "", text)
    return text


def numbers(text: str) -> list[float]:
    values = []
    for match in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", "")):
        try:
            values.append(float(match))
        except ValueError:
            pass
    return values


def clean_final_answer_candidate(value: str, *, max_chars: int = 200, max_words: int = 40) -> str:
    candidate = unicodedata.normalize("NFKC", str(value or "")).strip()
    candidate = re.sub(r"^\s*(?:[-*]\s+|\d+[.)]\s+)", "", candidate).strip()
    candidate = re.sub(r"^```[A-Za-z0-9_-]*\s*|\s*```$", "", candidate).strip()
    for wrapper in ("**", "__", "~~", "`", '"', "'"):
        if candidate.startswith(wrapper) and candidate.endswith(wrapper) and len(candidate) > 2 * len(wrapper):
            candidate = candidate[len(wrapper) : -len(wrapper)].strip()
    if not candidate or "\n" in candidate:
        return ""
    if len(candidate) > max_chars or len(candidate.split()) > max_words:
        return ""
    return candidate


def final_answer_candidate(text: str, *, strict: bool = False) -> str:
    """Best-effort final answer span for tasks that require a direct answer."""
    raw = str(text or "").strip()
    if not raw:
        return ""

    if strict:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        marker = re.compile(r"^(?:final answer|answer|答案)\s*[:：]\s*(.*)$", flags=re.IGNORECASE)
        for index in range(len(lines) - 1, -1, -1):
            match = marker.match(lines[index])
            if not match:
                continue
            value = match.group(1).strip()
            if value and index == len(lines) - 1:
                return clean_final_answer_candidate(value)
            if not value and index == len(lines) - 2:
                value = lines[index + 1]
                return clean_final_answer_candidate(value)
            continue
        if lines:
            return clean_final_answer_candidate(lines[-1])
        return clean_final_answer_candidate(raw)

    marker_matches = list(
        re.finditer(
            r"(?:final answer|answer|答案)\s*[:：]\s*(.+)",
            raw,
            flags=re.IGNORECASE,
        )
    )
    if marker_matches:
        return marker_matches[-1].group(1).strip()

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return raw


def verifier_reference(task: dict[str, Any]) -> dict[str, Any]:
    verifier = task.get("verifier") or {}
    reference = verifier.get("reference") if isinstance(verifier.get("reference"), dict) else {}
    merged = dict(reference or {})
    if "reference_answer" in verifier and "answer" not in merged:
        merged["answer"] = verifier.get("reference_answer")
    if "choices" in verifier and "choices" not in merged:
        merged["choices"] = verifier.get("choices")
    if "gold_option" in verifier and "gold_option" not in merged:
        merged["gold_option"] = verifier.get("gold_option")
    if "gold_answer" in verifier and "gold_answer" not in merged:
        merged["gold_answer"] = verifier.get("gold_answer")
    return merged


def answer_list(reference: dict[str, Any]) -> list[Any]:
    answers = reference.get("answer")
    if answers is None:
        answers = reference.get("gold_answer")
    if answers is None:
        return []
    if not isinstance(answers, list):
        answers = [answers]
    return answers


def debug_reference_answer(task: dict[str, Any]) -> Any:
    answers = answer_list(verifier_reference(task))
    if not answers:
        return None
    if len(answers) == 1:
        return answers[0]
    return answers


def normalize_option_label(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().upper()
    text = re.sub(r"^[\s\(\[\{（【]+|[\s\)\]\}）】.、:：-]+$", "", text)
    return text


def option_key_for_index(index: int) -> str:
    if 0 <= index < 26:
        return chr(ord("A") + index)
    return str(index + 1)


def parse_choices_text(value: str) -> Any:
    text = unicodedata.normalize("NFKC", value).strip()
    if not text:
        return text
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
        except Exception:
            continue
        if isinstance(parsed, (dict, list, tuple)):
            return parsed

    matches = re.findall(
        r"(?:^|\n)\s*[\(\[]?([A-Za-z])[\)\].、:：-]\s*(.+?)(?=\n\s*[\(\[]?[A-Za-z][\)\].、:：-]\s*|\Z)",
        text,
        flags=re.DOTALL,
    )
    if matches:
        return {label: choice.strip() for label, choice in matches if choice.strip()}
    return text


def normalized_choices(reference: dict[str, Any]) -> dict[str, str]:
    choices = reference.get("choices")
    if isinstance(choices, str):
        choices = parse_choices_text(choices)
    if isinstance(choices, dict):
        return {
            normalize_option_label(key): str(value).strip()
            for key, value in choices.items()
            if normalize_option_label(key)
        }
    if isinstance(choices, (list, tuple)):
        labelled_items = []
        for item in choices:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                labelled_items.append((item[0], item[1]))
                continue
            if isinstance(item, dict):
                label = item.get("label", item.get("key", item.get("option")))
                value = item.get("text", item.get("value", item.get("answer")))
                if label is not None and value is not None:
                    labelled_items.append((label, value))
                    continue
            labelled_items = []
            break
        if labelled_items:
            return {
                normalize_option_label(key): str(value).strip()
                for key, value in labelled_items
                if normalize_option_label(key)
            }

        inline_labels = []
        for item in choices:
            text = str(item).strip()
            match = re.match(r"^\s*[\(\[]?([A-Za-z])[\)\].、:：-]\s*(.+)$", unicodedata.normalize("NFKC", text))
            if not match:
                inline_labels = []
                break
            inline_labels.append((match.group(1), match.group(2).strip()))
        if inline_labels:
            return {
                normalize_option_label(key): value
                for key, value in inline_labels
                if normalize_option_label(key)
            }

        return {
            option_key_for_index(index): str(value).strip()
            for index, value in enumerate(choices)
        }
    if not isinstance(choices, dict):
        return {}
    return {}


def option_from_index_text(text: str, choices: dict[str, str], *, index_base: int | None = None) -> tuple[str | None, str | None]:
    if not re.fullmatch(r"\d+", text):
        return None, None
    keys = list(choices)
    value_index = int(text)
    if index_base is not None:
        index = value_index - index_base
        if 0 <= index < len(keys):
            key = keys[index]
            return key, choices[key]
        return None, None
    if value_index == 0 and keys:
        key = keys[0]
        return key, choices[key]
    one_based_index = value_index - 1
    if 0 <= one_based_index < len(keys):
        key = keys[one_based_index]
        return key, choices[key]
    return None, None


def reference_answer_index_base(reference: dict[str, Any], task: dict[str, Any] | None = None) -> int | None:
    for key in ("answer_index_base", "gold_index_base", "label_index_base"):
        if isinstance(reference.get(key), int):
            return int(reference[key])
    if task and task.get("source_dataset") == "lmms-lab/ai2d":
        answers = answer_list(reference)
        if len(answers) == 1 and str(answers[0]).strip().isdigit() and reference.get("choices"):
            return 0
    return None


def gold_option_from_reference(reference: dict[str, Any], task: dict[str, Any] | None = None) -> tuple[str | None, str | None]:
    choices = normalized_choices(reference)
    if not choices:
        return None, None
    explicit_gold = normalize_option_label(reference.get("gold_option") or "")
    if explicit_gold in choices:
        return explicit_gold, choices[explicit_gold]
    answer_index_base = reference_answer_index_base(reference, task)
    for index_key in ("answer_index", "gold_index", "label_index"):
        if index_key in reference and isinstance(reference.get(index_key), int):
            keys = list(choices)
            index = int(reference[index_key])
            if 0 <= index < len(keys):
                return keys[index], choices[keys[index]]
    for answer in answer_list(reference):
        raw = str(answer).strip()
        key = normalize_option_label(raw)
        if key in choices:
            return key, choices[key]
        if raw.isdigit() and answer_index_base is not None:
            indexed_key, indexed_value = option_from_index_text(raw, choices, index_base=answer_index_base)
            if indexed_key is not None:
                return indexed_key, indexed_value
        normalized_answer = norm_text(raw)
        for choice_key, choice_value in choices.items():
            if normalized_answer == norm_text(choice_value):
                return choice_key, choice_value
        if raw.isdigit():
            indexed_key, indexed_value = option_from_index_text(raw, choices)
            if indexed_key is not None:
                return indexed_key, indexed_value
    return None, None


def extract_predicted_option(text: str, choices: dict[str, str], *, index_base: int | None = None) -> tuple[str | None, str | None]:
    if not choices:
        return None, None
    stripped = unicodedata.normalize("NFKC", str(text or "")).strip()
    normalized = norm_text(stripped)
    first_line = stripped.splitlines()[0] if stripped else ""
    final_candidate = final_answer_candidate(stripped, strict=True)
    option_keys = "|".join(re.escape(key) for key in sorted(choices, key=len, reverse=True))
    prefix_pattern = rf"^\s*[*_`~\s]*[\(\[]?({option_keys})(?=$|[\)\].、:：\s,\-*`_~])"
    match = re.search(prefix_pattern, first_line, flags=re.IGNORECASE)
    if match:
        key = normalize_option_label(match.group(1))
        if key in choices:
            return key, choices[key]

    marker_pattern = (
        rf"(?:^|[\s\n])(?:answer|final answer|option|choice|答案|答|选项|选择|我选|应该选|应选)"
        rf"\s*(?:is|为|是)?\s*[:：]?\s*[*_`~\s]*[\(\[]?({option_keys})"
        rf"(?=$|[\)\].、:：\s,\-*`_~])"
    )
    marker_matches = list(re.finditer(marker_pattern, stripped, flags=re.IGNORECASE))
    if marker_matches:
        key = normalize_option_label(marker_matches[-1].group(1))
        if key in choices:
            return key, choices[key]
    normalized_key = normalize_option_label(normalized)
    if normalized_key in choices:
        key = normalized_key
        return key, choices[key]

    value_candidates = [final_candidate, first_line]
    for candidate in list(value_candidates):
        if not candidate:
            continue
        value_candidates.append(
            re.sub(
                r"^\s*[*_`~\s]*[\(\[]?(?:[A-Za-z]|\d+)[\)\].、:：-]\s*",
                "",
                candidate,
            ).strip()
        )

    matched_values = []
    for candidate in value_candidates:
        candidate_normalized = norm_text(candidate)
        if not candidate_normalized:
            continue
        if index_base is not None:
            indexed_key, indexed_value = option_from_index_text(candidate_normalized, choices, index_base=index_base)
            if indexed_key is not None:
                return indexed_key, indexed_value
        for key, value in choices.items():
            expected = norm_text(value)
            if expected and candidate_normalized == expected:
                matched_values.append((key, value))
        unique_matches = list(dict.fromkeys(matched_values))
        if len(unique_matches) == 1:
            return unique_matches[0]
        indexed_key, indexed_value = option_from_index_text(candidate_normalized, choices)
        if indexed_key is not None:
            return indexed_key, indexed_value

    indexed_key, indexed_value = option_from_index_text(normalized, choices, index_base=index_base)
    if indexed_key is not None:
        return indexed_key, indexed_value
    return None, None


def verify_multiple_choice_answer(task: dict[str, Any], text: str) -> tuple[bool, str, dict[str, Any]] | None:
    reference = verifier_reference(task)
    choices = normalized_choices(reference)
    if not choices:
        return None
    gold_option, gold_value = gold_option_from_reference(reference, task)
    meta = {"gold_option": gold_option, "gold_answer": gold_value}
    if gold_option is None:
        return False, "verifier_error: missing_gold_option", meta
    predicted_option, predicted_value = extract_predicted_option(
        text,
        choices,
        index_base=reference_answer_index_base(reference, task),
    )
    meta.update({"predicted_option": predicted_option, "predicted_answer": predicted_value})
    if predicted_option is None:
        return False, f"verifier_error: cannot_extract_option; gold={gold_option}", meta
    if predicted_option == gold_option:
        return True, f"option_match: {predicted_option}", meta
    return False, f"model_wrong: predicted={predicted_option} gold={gold_option}", meta


def extract_code_with_language(text: str) -> tuple[str | None, str]:
    fenced = re.findall(r"```([A-Za-z0-9_+#.-]*)\s*\n?(.*?)```", text, flags=re.DOTALL)
    if fenced:
        language, code = max(fenced, key=lambda item: len(item[1]))
        return language.strip().lower() or None, code.strip()
    return None, text.strip()


def extract_code(text: str) -> str:
    return extract_code_with_language(text)[1]


def run_python_file(code: str, stdin: str = "", timeout: int = 5) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="openclaw_code_") as tmpdir:
        path = Path(tmpdir) / "solution.py"
        path.write_text(code, encoding="utf-8")
        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", str(Path(tmpdir) / "matplotlib"))
        return subprocess.run(
            [sys.executable, str(path)],
            input=stdin,
            text=True,
            capture_output=True,
            timeout=timeout,
            cwd=tmpdir,
            env=env,
            check=False,
        )


def run_cpp_file(code: str, stdin: str = "", timeout: int = 10) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="openclaw_cpp_") as tmpdir:
        source_path = Path(tmpdir) / "solution.cpp"
        binary_path = Path(tmpdir) / "solution"
        source_path.write_text(code, encoding="utf-8")
        compile_result = subprocess.run(
            ["g++", "-std=c++17", "-O2", "-pipe", str(source_path), "-o", str(binary_path)],
            text=True,
            capture_output=True,
            timeout=timeout,
            cwd=tmpdir,
            check=False,
        )
        if compile_result.returncode != 0:
            return compile_result
        return subprocess.run(
            [str(binary_path)],
            input=stdin,
            text=True,
            capture_output=True,
            timeout=timeout,
            cwd=tmpdir,
            check=False,
        )


def verify_mbpp(task: dict[str, Any], text: str, timeout: int) -> tuple[bool, str]:
    reference = task.get("verifier", {}).get("reference", {})
    tests = list(reference.get("tests") or []) + list(reference.get("challenge_tests") or [])
    if not tests:
        return False, "missing_unit_tests"
    code = extract_code(text)
    test_code = code + "\n\n" + "\n".join(tests) + "\n"
    try:
        result = run_python_file(test_code, timeout=min(timeout, 10))
    except subprocess.TimeoutExpired:
        return False, "unit_tests_timeout"
    if result.returncode == 0:
        return True, "unit_tests_passed"
    return False, "unit_tests_failed"


def code_for_python_unittest(reference: dict[str, Any], code: str) -> str:
    entry_point = str(reference.get("entry_point") or "").strip()
    if entry_point and re.search(rf"^\s*def\s+{re.escape(entry_point)}\b", code, flags=re.MULTILINE):
        return code
    code_prompt = str(reference.get("code_prompt") or "").rstrip()
    if not code_prompt:
        return code
    body = code.strip("\n")
    if body and not body.startswith((" ", "\t")):
        body = "\n".join(("    " + line if line.strip() else line) for line in body.splitlines())
    return code_prompt + "\n" + body + "\n"


def verify_python_unittest(task: dict[str, Any], text: str, timeout: int) -> tuple[bool, str]:
    reference = task.get("verifier", {}).get("reference", {})
    tests = [str(test).strip() for test in reference.get("tests") or [] if str(test).strip()]
    if not tests:
        return False, "missing_python_unittest_tests"
    code = code_for_python_unittest(reference, extract_code(text))
    test_code = (
        code
        + "\n\n"
        + "\n\n".join(tests)
        + "\n\n"
        + "if __name__ == '__main__':\n"
        + "    unittest.main(argv=['openclaw_unittest'], verbosity=0)\n"
    )
    try:
        result = run_python_file(test_code, timeout=min(timeout, 20))
    except subprocess.TimeoutExpired:
        return False, "python_unittest_timeout"
    if result.returncode == 0:
        return True, "python_unittest_passed"
    missing_module = re.search(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]", result.stderr or "")
    if missing_module:
        return False, f"verifier_error: missing_dependency:{missing_module.group(1)}"
    if os.environ.get("OPENCLAW_DEBUG_PYTHON_UNITTEST") == "1":
        return (
            False,
            "python_unittest_failed; "
            f"stdout={debug_snippet(result.stdout)!r}; "
            f"stderr={debug_snippet(result.stderr)!r}",
        )
    return False, "python_unittest_failed"


def official_code_eval_command(benchmark: str) -> str:
    if benchmark == "bigcodebench":
        return os.environ.get("BIGCODEBENCH_EVAL_COMMAND", "")
    if benchmark == "livecodebench":
        return os.environ.get("LIVECODEBENCH_EVAL_COMMAND", "")
    return ""


def official_prediction_code(task: dict[str, Any], text: str) -> str:
    reference = task.get("verifier", {}).get("reference", {})
    code = extract_code(text)
    if reference.get("benchmark") == "bigcodebench":
        code = code_for_python_unittest(reference, code)
    return code


def write_official_prediction_files(
    task: dict[str, Any],
    text: str,
    tmpdir: Path,
) -> tuple[Path, Path]:
    reference = task.get("verifier", {}).get("reference", {})
    benchmark = str(reference.get("benchmark") or "")
    code = official_prediction_code(task, text)
    predictions_path = tmpdir / "predictions.jsonl"
    result_path = tmpdir / "official_eval_result.json"
    if benchmark == "bigcodebench":
        row = {
            "task_id": reference.get("benchmark_task_id") or task.get("task_id"),
            "solution": code,
            "raw_solution": text,
        }
        predictions_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    elif benchmark == "livecodebench":
        predictions_path = tmpdir / "predictions.json"
        question_id = reference.get("question_id")
        row = {
            "question_id": question_id,
            "code_list": [code],
            "output_list": [text],
            "reference": reference,
            "source_ref": task.get("source_ref") or {},
        }
        predictions_path.write_text(json.dumps([row], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        raise ValueError(f"Unsupported official code benchmark: {benchmark}")
    return predictions_path, result_path


def bool_from_official_status(value: Any) -> bool | None:
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


def official_pass_from_node(node: Any, task_key: str | None = None) -> bool | None:
    if isinstance(node, dict):
        node_id = node.get("task_id") or node.get("question_id") or node.get("id")
        if task_key is None or str(node_id) == str(task_key) or node_id is None:
            for key in ("passed", "pass", "is_passed", "success", "accepted", "correct"):
                if key in node:
                    value = bool_from_official_status(node.get(key))
                    if value is not None:
                        return value
            for key in ("status", "result", "verdict"):
                if key in node:
                    value = bool_from_official_status(node.get(key))
                    if value is not None:
                        return value
            graded_list = node.get("graded_list")
            if isinstance(graded_list, list) and graded_list:
                graded = [bool_from_official_status(item) for item in graded_list]
                if all(item is not None for item in graded):
                    return bool(graded[0])
            pass_at_1 = node.get("pass@1")
            if isinstance(pass_at_1, (int, float)):
                return pass_at_1 > 0
        for value in node.values():
            found = official_pass_from_node(value, task_key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = official_pass_from_node(item, task_key)
            if found is not None:
                return found
    return None


def load_official_eval_result(result_path: Path, stdout: str) -> Any:
    candidates = []
    if result_path.exists() and result_path.stat().st_size:
        candidates.append(result_path.read_text(encoding="utf-8"))
    candidates.append(stdout)
    for text in candidates:
        text = str(text or "").strip()
        if not text:
            continue
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


def official_wrapper_error_detail(node: Any) -> str | None:
    if isinstance(node, dict):
        error = str(node.get("error") or "").strip()
        if error.startswith(("official_eval_", "livecodebench_", "bigcodebench_")):
            parts = [error]
            returncode = node.get("returncode")
            if returncode is not None:
                parts.append(f"returncode={returncode}")
            stderr = debug_snippet(str(node.get("stderr") or ""), max_chars=800)
            stdout = debug_snippet(str(node.get("stdout") or ""), max_chars=800)
            if stderr:
                parts.append(f"stderr={stderr!r}")
            if stdout:
                parts.append(f"stdout={stdout!r}")
            return "; ".join(parts)
        for value in node.values():
            found = official_wrapper_error_detail(value)
            if found:
                return found
    elif isinstance(node, list):
        for item in node:
            found = official_wrapper_error_detail(item)
            if found:
                return found
    return None


def verify_official_code_benchmark(task: dict[str, Any], text: str, timeout: int) -> tuple[bool, str]:
    reference = task.get("verifier", {}).get("reference", {})
    benchmark = str(reference.get("benchmark") or "")
    command_template = official_code_eval_command(benchmark)
    if not command_template:
        return False, f"verifier_error: official_evaluator_not_configured:{benchmark}"
    with tempfile.TemporaryDirectory(prefix=f"openclaw_{benchmark}_eval_") as tmp:
        tmpdir = Path(tmp)
        predictions_path, result_path = write_official_prediction_files(task, text, tmpdir)
        source_ref = task.get("source_ref") or {}
        task_key = str(reference.get("benchmark_task_id") or reference.get("question_id") or task.get("task_id"))
        format_values = {
            "predictions": str(predictions_path),
            "prediction": str(predictions_path),
            "results": str(result_path),
            "result": str(result_path),
            "workdir": str(tmpdir),
            "benchmark": benchmark,
            "task_id": task_key,
            "question_id": str(reference.get("question_id") or ""),
            "raw_path": str(source_ref.get("raw_path") or ""),
            "raw_line": str(source_ref.get("raw_line") or ""),
        }
        command = command_template
        for key, value in format_values.items():
            command = command.replace("{" + key + "}", value)
        eval_timeout = int(os.environ.get("ROUTER_SFT_OFFICIAL_EVAL_TIMEOUT", str(max(timeout, 120))))
        try:
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                capture_output=True,
                timeout=eval_timeout,
                cwd=ROOT,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, f"verifier_error: official_eval_timeout:{benchmark}"
        if result.returncode != 0:
            parsed_error = load_official_eval_result(result_path, result.stdout)
            detail = official_wrapper_error_detail(parsed_error)
            if not detail:
                detail = (result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}")[:500]
            return False, f"verifier_error: official_eval_failed:{benchmark}: {detail}"
        parsed = load_official_eval_result(result_path, result.stdout)
        wrapper_error = official_wrapper_error_detail(parsed)
        if wrapper_error:
            return False, f"verifier_error: official_eval_failed:{benchmark}: {wrapper_error}"
        passed = official_pass_from_node(parsed, task_key)
        if passed is None:
            return False, f"verifier_error: official_eval_unparseable:{benchmark}"
        if passed:
            return True, "official_eval_passed"
        return False, "model_wrong: official_eval_failed"


def runtime_error_answer_reason(text: str) -> str | None:
    normalized = " ".join(str(text or "").strip().lower().split())
    if not normalized:
        return "api_error: empty_model_response"
    timeout_markers = (
        "llm request timed out",
        "request timed out before a response was generated",
        "model idle timeout",
        "increase `models.providers",
        "increase `agents.defaults.timeoutseconds",
    )
    if any(marker in normalized for marker in timeout_markers):
        return "api_error: model_timeout"
    error_markers = (
        "llm request failed",
        "provider returned an error",
        "openclaw returned error output",
    )
    if any(marker in normalized for marker in error_markers):
        return "api_error: model_runtime_error"
    return None


def normalize_stdout(text: str) -> str:
    lines = [line.rstrip() for line in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines).strip()


def debug_snippet(value: str, max_chars: int = 1000) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"


def io_mismatch_reason(index: int, stdin_text: str, expected_text: str, actual_text: str) -> str:
    reason = f"input_output_mismatch_{index}"
    if os.environ.get("OPENCLAW_DEBUG_IO_MISMATCH") != "1":
        return reason
    return (
        f"{reason}; "
        f"stdin={debug_snippet(stdin_text)!r}; "
        f"expected={debug_snippet(expected_text)!r}; "
        f"actual={debug_snippet(actual_text)!r}"
    )


def io_special_mismatch_reason(index: int, checker: str, detail: str) -> str:
    reason = f"input_output_semantic_mismatch_{checker}_{index}"
    if os.environ.get("OPENCLAW_DEBUG_IO_MISMATCH") != "1":
        return reason
    return f"{reason}; detail={debug_snippet(detail)!r}"


def function_mismatch_reason(index: int, fn_name: str, args: list[Any], expected: Any, actual_text: str) -> str:
    reason = f"function_mismatch_{index}"
    if os.environ.get("OPENCLAW_DEBUG_FUNCTION_MISMATCH") != "1":
        return reason
    return (
        f"{reason}; "
        f"fn={fn_name!r}; "
        f"args={debug_snippet(json.dumps(args, ensure_ascii=False))!r}; "
        f"expected={debug_snippet(canonical_json(expected))!r}; "
        f"actual={debug_snippet(normalize_stdout(actual_text))!r}"
    )


def function_runtime_error_reason(index: int, fn_name: str, args: list[Any], result: subprocess.CompletedProcess[str]) -> str:
    reason = f"function_runtime_error_{index}"
    if os.environ.get("OPENCLAW_DEBUG_FUNCTION_RUNTIME") != "1":
        return reason
    return (
        f"{reason}; "
        f"fn={fn_name!r}; "
        f"args={debug_snippet(json.dumps(args, ensure_ascii=False))!r}; "
        f"returncode={result.returncode}; "
        f"stdout={debug_snippet(result.stdout)!r}; "
        f"stderr={debug_snippet(result.stderr)!r}"
    )


def apps_io_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(apps_io_text(item) for item in value)
    return "" if value is None else str(value)


def task_instruction_text(task: dict[str, Any]) -> str:
    parts = []
    executor_input = task.get("executor_input") if isinstance(task.get("executor_input"), dict) else {}
    for message in executor_input.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
    router_instruction = task.get("router_view", {}).get("instruction") if isinstance(task.get("router_view"), dict) else None
    if router_instruction:
        parts.append(str(router_instruction))
    return "\n".join(parts)


def parse_apps_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def parse_function_arg(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def parse_function_args(value: Any, arg_count: int | None = None) -> list[Any]:
    if isinstance(value, list):
        return [parse_function_arg(item) for item in value]
    if not isinstance(value, str):
        return [parse_function_arg(value)]
    text = value.strip()
    if arg_count and arg_count > 1:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) == arg_count:
            return [parse_function_arg(line) for line in lines]
        parsed = parse_function_arg(text)
        if isinstance(parsed, list) and len(parsed) == arg_count:
            return parsed
    return [parse_function_arg(value)]


def canonical_json(value: Any) -> str:
    return json.dumps(parse_apps_value(value), ensure_ascii=False, sort_keys=True)


def function_output_matches(actual_text: str, expected: Any) -> bool:
    actual = parse_apps_value(normalize_stdout(actual_text))
    if canonical_json(actual) == canonical_json(expected):
        return True
    if isinstance(expected, list) and len(expected) == 1:
        return canonical_json(actual) == canonical_json(expected[0])
    return False


def is_balloon_division_task(task: dict[str, Any]) -> bool:
    text = task_instruction_text(task).lower()
    return (
        "grigory" in text
        and "andrew" in text
        and "balloon" in text
        and "do not rip the packets" in text
        and "if there are multiple ways to divide balloons" in text
    )


def balloon_division_has_solution(values: list[int]) -> bool:
    if len(values) < 2:
        return False
    total = sum(values)
    return any(value * 2 != total for value in values)


def verify_balloon_division_output(stdin_text: str, actual_text: str) -> tuple[bool, str]:
    try:
        input_numbers = [int(token) for token in str(stdin_text).split()]
    except ValueError:
        return False, "invalid_test_input"
    if not input_numbers:
        return False, "invalid_test_input"
    n = input_numbers[0]
    values = input_numbers[1 : 1 + n]
    if n != len(values):
        return False, "invalid_test_input"

    output_tokens = str(actual_text).split()
    has_solution = balloon_division_has_solution(values)
    if not output_tokens:
        return False, "empty_output"
    try:
        first = int(output_tokens[0])
    except ValueError:
        return False, "non_integer_output"

    if first == -1:
        if len(output_tokens) != 1:
            return False, "extra_tokens_after_minus_one"
        return (not has_solution), "minus_one_valid" if not has_solution else "reported_impossible_but_solution_exists"

    k = first
    if k <= 0:
        return False, "non_positive_k"
    if len(output_tokens) != k + 1:
        return False, f"expected_{k}_indices_got_{max(0, len(output_tokens) - 1)}"
    try:
        indices = [int(token) for token in output_tokens[1:]]
    except ValueError:
        return False, "non_integer_index"
    if len(set(indices)) != len(indices):
        return False, "duplicate_index"
    if any(index < 1 or index > n for index in indices):
        return False, "index_out_of_range"
    if k >= n:
        return False, "andrew_gets_no_packet"

    selected = sum(values[index - 1] for index in indices)
    other = sum(values) - selected
    if selected == other:
        return False, "equal_partition_sum"
    return True, "semantic_match: balloon_division"


def verify_apps_special_output(task: dict[str, Any], stdin_text: str, actual_text: str) -> tuple[bool, str] | None:
    if is_balloon_division_task(task):
        passed, reason = verify_balloon_division_output(stdin_text, actual_text)
        if passed:
            return True, reason
        return False, f"balloon_division:{reason}"
    return None


def apps_semantic_checker_name(task: dict[str, Any]) -> str | None:
    if is_balloon_division_task(task):
        return "balloon_division"
    return None


def apps_output_section_text(task: dict[str, Any]) -> str:
    text = task_instruction_text(task).lower()
    output_section = text.split("-----output-----", 1)[1] if "-----output-----" in text else text
    for delimiter in (
        "-----examples-----",
        "-----example-----",
        "-----note-----",
        "-----input-----",
    ):
        if delimiter in output_section:
            output_section = output_section.split(delimiter, 1)[0]
    return output_section


def apps_output_is_deterministic_scalar(output_section: str) -> bool:
    scalar_patterns = (
        r"\bprint\s+(?:the\s+)?(?:length|number|count|maximum|minimum|minimal|minimum possible|maximum possible|"
        r"largest|smallest|sum|value|answer|total|index|position|score)\b",
        r"\boutput\s+(?:the\s+)?(?:length|number|count|maximum|minimum|minimal|minimum possible|maximum possible|"
        r"largest|smallest|sum|value|answer|total|index|position|score)\b",
        r"\bprint\s+an?\s+integer\b",
        r"\boutput\s+an?\s+integer\b",
        r"\bprint\s+a\s+single\s+integer\b",
        r"\boutput\s+a\s+single\s+integer\b",
    )
    return any(re.search(pattern, output_section) for pattern in scalar_patterns)


def apps_requires_semantic_output(task: dict[str, Any]) -> bool:
    output_section = apps_output_section_text(task)
    deterministic_tie_break_markers = (
        "choose the one",
        "display the one",
        "print the one",
        "output the one",
        "smallest",
        "largest",
        "lexicographically",
        "earliest",
        "first such",
        "if there is still a tie",
    )
    if any(marker in output_section for marker in deterministic_tie_break_markers):
        return False
    if apps_output_is_deterministic_scalar(output_section):
        return False
    markers = (
        "output any",
        "print any",
        "any valid",
        "any of them",
        "any one",
        "any order",
        "any solution",
        "arbitrary",
        "any correct",
        "any suitable",
        "output one possible",
        "print one possible",
        "find any",
        "print a valid",
        "output a valid",
    )
    return any(marker in output_section for marker in markers)


def verify_apps_function(task: dict[str, Any], code: str, timeout: int) -> tuple[bool, str] | None:
    reference = task.get("verifier", {}).get("reference", {})
    input_output = reference.get("input_output") or {}
    fn_name = input_output.get("fn_name")
    arg_count = input_output.get("arg_count")
    if not isinstance(arg_count, int):
        arg_count = None
    inputs = input_output.get("inputs") or []
    outputs = input_output.get("outputs") or []
    if not fn_name:
        return None
    if not inputs or not outputs or len(inputs) != len(outputs):
        return False, "missing_function_tests"

    for index, (raw_args, expected) in enumerate(zip(inputs, outputs)):
        args = parse_function_args(raw_args, arg_count)
        harness = (
            "from typing import *\n"
            + code
            + "\n\n"
            + "import json as _openclaw_json\n"
            + f"_openclaw_args = _openclaw_json.loads({json.dumps(json.dumps(args, ensure_ascii=False))})\n"
            + f"_openclaw_fn = globals().get({json.dumps(str(fn_name))})\n"
            + "if _openclaw_fn is None:\n"
            + "    _openclaw_cls = globals().get('Solution')\n"
            + "    if _openclaw_cls is not None:\n"
            + f"        _openclaw_fn = getattr(_openclaw_cls(), {json.dumps(str(fn_name))}, None)\n"
            + "if _openclaw_fn is None:\n"
            + f"    raise NameError({json.dumps(str(fn_name))})\n"
            + "_openclaw_result = _openclaw_fn(*_openclaw_args)\n"
            + "print(_openclaw_json.dumps(_openclaw_result, ensure_ascii=False, sort_keys=True))\n"
        )
        try:
            result = run_python_file(harness, timeout=min(timeout, 10))
        except subprocess.TimeoutExpired:
            return False, f"function_timeout_{index}"
        if result.returncode != 0:
            return False, function_runtime_error_reason(index, str(fn_name), args, result)
        if not function_output_matches(result.stdout, expected):
            return False, function_mismatch_reason(index, str(fn_name), args, expected, result.stdout)
    return True, "function_tests_passed"


def verify_apps(task: dict[str, Any], text: str, timeout: int) -> tuple[bool, str]:
    reference = task.get("verifier", {}).get("reference", {})
    input_output = reference.get("input_output") or {}
    inputs = input_output.get("inputs") or []
    outputs = input_output.get("outputs") or []
    if not inputs or not outputs or len(inputs) != len(outputs):
        return False, "missing_input_output_tests"
    language, code = extract_code_with_language(text)
    function_result = verify_apps_function(task, code, timeout)
    if function_result is not None:
        return function_result
    semantic_checker = apps_semantic_checker_name(task)
    if apps_requires_semantic_output(task) and semantic_checker is None:
        return False, "verifier_unsupported: apps_semantic_output"
    for index, (stdin, expected) in enumerate(zip(inputs, outputs)):
        stdin_text = apps_io_text(stdin)
        expected_text = apps_io_text(expected)
        try:
            if language in {"c", "cc", "cpp", "c++", "cplusplus"}:
                result = run_cpp_file(code, stdin=stdin_text, timeout=min(timeout, 15))
            else:
                result = run_python_file(code, stdin=stdin_text, timeout=min(timeout, 10))
        except subprocess.TimeoutExpired:
            return False, f"input_output_timeout_{index}"
        if result.returncode != 0:
            return False, f"input_output_runtime_error_{index}"
        actual_text = result.stdout
        special_result = verify_apps_special_output(task, stdin_text, actual_text)
        if special_result is not None:
            special_passed, special_reason = special_result
            if not special_passed:
                return False, io_special_mismatch_reason(index, semantic_checker or "apps", special_reason)
            continue
        if normalize_stdout(actual_text) != normalize_stdout(expected_text):
            return False, io_mismatch_reason(index, stdin_text, expected_text, actual_text)
    return True, "input_output_tests_passed"


def ordered_expected_tool_calls(task: dict[str, Any], expected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reference = task.get("verifier", {}).get("reference", {})
    tool_steps = reference.get("tool_steps") or []
    if not isinstance(tool_steps, list):
        return expected

    calls_by_name = {canonical_name(call.get("name")): call for call in expected}
    step_order = []
    seen = set()
    for step in tool_steps:
        step_text = str(step).lower()
        matches = []
        for name in calls_by_name:
            index = step_text.find(name)
            if index >= 0:
                matches.append((index, name))
        for _index, name in sorted(matches):
            if name not in seen:
                step_order.append(name)
                seen.add(name)

    if set(step_order) == set(calls_by_name):
        return [calls_by_name[name] for name in step_order]
    return expected


def reference_has_complete_tool_step_order(task: dict[str, Any], expected: list[dict[str, Any]]) -> bool:
    reference = task.get("verifier", {}).get("reference", {})
    tool_steps = reference.get("tool_steps") or []
    if not isinstance(tool_steps, list) or not tool_steps:
        return False

    calls_by_name = {canonical_name(call.get("name")): call for call in expected}
    # Duplicate expected names cannot be reliably disambiguated from text steps.
    if len(calls_by_name) != len(expected):
        return False

    seen = set()
    for step in tool_steps:
        step_text = str(step).lower()
        for name in calls_by_name:
            if name in step_text:
                seen.add(name)
    return seen == set(calls_by_name)


def verify_tool(task: dict[str, Any], message: dict[str, Any]) -> tuple[bool, str]:
    expected = task.get("verifier", {}).get("reference", {}).get("expected_calls") or []
    observed = parse_tool_calls(message)
    if not expected:
        return bool(observed), "no_reference_expected_calls"
    if not observed:
        return False, "missing_tool_calls"

    cursor = 0
    for expected_call in ordered_expected_tool_calls(task, expected):
        expected_name = canonical_name(expected_call.get("name"))
        schema_has_argument_fields = tool_schema_has_argument_fields(task, expected_name)
        found_name = False
        match_index = None
        for index in range(cursor, len(observed)):
            observed_call = observed[index]
            if canonical_name(observed_call.get("name")) != expected_name:
                continue
            found_name = True
            if schema_has_argument_fields and not args_match(
                expected_call.get("arguments", {}),
                observed_call.get("arguments", {}),
            ):
                continue
            match_index = index
            break
        if match_index is None:
            if found_name:
                return False, "tool_call_arguments_mismatch"
            if any(canonical_name(call.get("name")) == expected_name for call in observed[:cursor]):
                return False, "tool_call_order_mismatch"
            return False, "tool_call_or_arguments_mismatch"
        cursor = match_index + 1
    return True, "tool_call_order_match"


def numeric_lists_match(expected_numbers: list[float], predicted_numbers: list[float], *, strict: bool) -> bool:
    if strict and len(expected_numbers) != len(predicted_numbers):
        return False
    unmatched = list(predicted_numbers)
    for expected in expected_numbers:
        match_index = None
        for index, predicted in enumerate(unmatched):
            if abs(expected - predicted) <= max(1e-4, abs(expected) * 0.01):
                match_index = index
                break
        if match_index is None:
            return False
        unmatched.pop(match_index)
    return True


def verify_answer_against_reference(
    reference: dict[str, Any],
    text: str,
    *,
    strict_final_answer: bool = False,
    allow_bounded_text: bool = True,
) -> tuple[bool, str]:
    answers = answer_list(reference)
    if not answers:
        return False, "missing_reference_answer"

    predicted = norm_text(text)
    predicted_numbers = numbers(predicted)
    for answer in answers:
        expected = norm_text(answer)
        if not expected:
            continue
        if predicted == expected:
            return True, "text_match"
        if (
            allow_bounded_text
            and not strict_final_answer
            and len(expected) >= 4
            and re.search(rf"(^|[\s,;:：]){re.escape(expected)}($|[\s,;。,.])", predicted)
        ):
            return True, "bounded_text_match"
        expected_numbers = numbers(expected)
        if expected_numbers and predicted_numbers:
            if numeric_lists_match(expected_numbers, predicted_numbers, strict=strict_final_answer):
                return True, "numeric_match"
    return False, "answer_mismatch"


def verify_normalized_answer(task: dict[str, Any], text: str) -> tuple[bool, str]:
    return verify_final_answer_candidate(task, text)


def verify_final_answer_candidate(task: dict[str, Any], text: str) -> tuple[bool, str]:
    candidate = final_answer_candidate(text, strict=True)
    if not candidate:
        return False, "answer_mismatch_no_final_answer_candidate"
    passed, reason = verify_answer_against_reference(
        verifier_reference(task),
        candidate,
        strict_final_answer=True,
    )
    if passed:
        return passed, reason
    if candidate.strip() != str(text or "").strip():
        return False, f"{reason}_in_final_answer"
    return False, reason


def official_final_answer_for_judge(text: str) -> str:
    candidate = final_answer_candidate(text, strict=True)
    return candidate or str(text or "").strip()


def normalized_reference_answers(reference: dict[str, Any]) -> list[str]:
    return [answer for answer in (norm_text(item) for item in answer_list(reference)) if answer]


def bare_label_reference_requires_exact_match(task: dict[str, Any]) -> bool:
    if expected_output_type(task) != "vision_natural_language_answer":
        return False

    reference = verifier_reference(task)
    if reference.get("choices"):
        return False

    answers = normalized_reference_answers(reference)
    if not answers:
        return False

    return all(re.fullmatch(r"[a-d]|\d+", answer) for answer in answers)


def exact_reference_mismatch_reason(task: dict[str, Any]) -> str:
    if os.environ.get("OPENCLAW_DEBUG_EXACT_REFERENCE") != "1":
        return "exact_reference_mismatch"
    reference = verifier_reference(task)
    debug = {
        "answer": reference.get("answer"),
        "choices": reference.get("choices"),
    }
    return f"exact_reference_mismatch; reference={debug_snippet(json.dumps(debug, ensure_ascii=False, sort_keys=True))}"


def official_final_answer_requires_rule_verification(task: dict[str, Any]) -> bool:
    if expected_output_type(task) == "gaia_agent_final_answer":
        return True
    return False


def task_image_content_items(task: dict[str, Any]) -> list[dict[str, str]]:
    image_items = []
    for message in task.get("executor_input", {}).get("messages", []) or []:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_path" and item.get("image_path"):
                image_items.append({"type": "image_path", "image_path": str(item["image_path"])})
    return image_items


def task_image_count(task: dict[str, Any]) -> int:
    return len(task_image_content_items(task))


def choice_image_reference_numbers(reference: dict[str, Any]) -> list[int]:
    choices = reference.get("choices")
    if not isinstance(choices, (list, tuple)):
        return []
    refs = []
    for choice in choices:
        for match in re.findall(r"<image\s+(\d+)>", str(choice), flags=re.IGNORECASE):
            refs.append(int(match))
    return refs


def is_malformed_mmmu_image_choice_task(task: dict[str, Any]) -> bool:
    if str(task.get("source_dataset") or "") != "MMMU/MMMU":
        return False
    refs = choice_image_reference_numbers(verifier_reference(task))
    return bool(refs) and task_image_count(task) < max(refs)


def filter_malformed_tasks(tasks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept = []
    dropped = []
    for task in tasks:
        if is_malformed_mmmu_image_choice_task(task):
            dropped.append(task)
        else:
            kept.append(task)
    return kept, dropped


def filter_rows_to_tasks(rows: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    task_ids = {str(task.get("task_id")) for task in tasks}
    return [row for row in rows if str(row.get("task_id")) in task_ids]


def reference_for_judge(task: dict[str, Any]) -> dict[str, Any]:
    reference = task.get("verifier", {}).get("reference", {})
    if not isinstance(reference, dict):
        return {}
    if is_browsecomp_plus_task(task):
        return {
            key: reference.get(key)
            for key in ("answer", "query_id")
            if reference.get(key) not in (None, "", [], {})
        }
    return reference


def build_judge_messages(task: dict[str, Any], answer: str) -> list[dict[str, Any]]:
    reference = reference_for_judge(task)
    if is_browsecomp_plus_task(task):
        judge_payload = {
            "question": task.get("router_view", {}).get("instruction"),
            "response": answer,
            "correct_answer": reference.get("answer"),
            "query_id": reference.get("query_id"),
            "grading_policy": (
                "Extract the candidate's final exact answer from the response, then judge whether it is "
                "semantically equivalent to correct_answer in the context of the question. Allow harmless "
                "string variations, aliases, formatting differences, and answers that are more precise or "
                "more verbose than correct_answer when all added details are correct. Return passed=false "
                "for inconsistent, ambiguous, non-equivalent, speculative, or incorrect answers. Do not solve "
                "the original question yourself or argue for a different answer; only compare the extracted "
                "final answer to correct_answer."
            ),
            "output_schema": {
                "passed": "boolean",
                "reason": "short reason",
                "extracted_final_answer": "string or null",
                "confidence": "0-100 if available, otherwise 100",
            },
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are a BrowseComp-Plus semantic answer judge. Return only JSON: "
                    "{\"passed\": true|false, \"reason\": \"short reason\", "
                    "\"extracted_final_answer\": string|null, \"confidence\": number}."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(judge_payload, ensure_ascii=False),
            },
        ]

    image_items = task_image_content_items(task)
    reference_grading_policy = (
        "Extract the candidate's final answer from the response before judging. "
        "Use the reference as the primary grading anchor. Prefer exact matches to "
        "`answer`, `gold_answer`, `gold_option`, and `choices` when present. Pass only "
        "when the candidate is the same answer, a clearly equivalent formatting variant, "
        "or an unambiguous alias explicitly supported by the instruction or reference. "
        "Do not guess missing mappings, infer a different gold label from the image, or "
        "replace the reference with your own solution. If the candidate cannot be "
        "confidently matched to the reference, return passed=false."
    )
    code_grading_policy = (
        "For code tasks, grade against the user's requested behavior first. Treat the "
        "reference as evidence of required behavior, not as a mandatory checklist. Do "
        "not fail solely because the candidate omits context managers, try/except "
        "blocks, logging, comments, helper functions, exact names, or the same wrapper "
        "structure unless the instruction explicitly requires them or their absence "
        "creates a clear functional, security, or resource-cleanup bug. Pass when the "
        "core requested behavior is present and the code would plausibly work. Fail "
        "when a core requirement is missing, the code changes behavior materially, is "
        "clearly non-code/error output, or would not plausibly work."
    )
    judge_payload = {
        "instruction": task.get("router_view", {}).get("instruction"),
        "category": task.get("category"),
        "verifier_type": task.get("verifier", {}).get("type"),
        "reference": reference,
        "candidate_answer": answer,
        "grading_policy": (
            "Pass only if the candidate answer would satisfy the user request. "
            f"{reference_grading_policy} "
            f"{code_grading_policy} "
            "For visual QA, inspect the attached image(s) directly and require the exact "
            "entity/value from the reference unless the reference clearly allows alternatives. "
            "Use the image only to disambiguate whether the candidate and reference denote "
            "the same thing; do not use it to override the reference answer. If the reference "
            "answer is a bare option index or option letter, do not infer an option mapping "
            "from the candidate's semantic description; pass only when the candidate includes "
            "the exact referenced label or the reference explicitly provides an allowed alias. "
            "For title/name answers in web agent tasks, pass an unambiguous official translated "
            "title or exact literal translation of the reference unless the instruction asks for "
            "the exact original-language or exact IMDb/listed title. For tool tasks, require "
            "correct tool intent and materially correct arguments."
        ),
    }
    user_content: Any
    if image_items:
        user_content = [
            {
                "type": "text",
                "text": json.dumps(judge_payload, ensure_ascii=False),
            },
            *image_items,
        ]
    else:
        user_content = json.dumps(judge_payload, ensure_ascii=False)
    return [
        {
            "role": "system",
            "content": (
                "You are a strict reference-grounded verifier for coding and task answers. "
                "Do not speculate beyond the provided reference, instruction, and attached evidence. "
                "Return only JSON: {\"passed\": true|false, \"reason\": \"short reason\"}."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def multiple_choice_llm_recheck_enabled(task: dict[str, Any]) -> bool:
    configured = os.environ.get("ROUTER_SFT_MC_LLM_RECHECK", "")
    if configured:
        return configured.strip().lower() not in {"0", "false", "no", "off"}
    return task.get("category") == "multimodal_doc_visual"


JudgeCache = dict[tuple[str, str, str, str], tuple[bool, str]]
JudgeUsageLog = list[dict[str, Any]]
TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "total_tokens",
)

COST_ESTIMATION_ENABLED = True
COST_PRICE_CACHE_PATH = ROOT / DEFAULT_OPENROUTER_PRICE_CACHE
COST_PRICE_REFRESH = False
COST_PRICE_OFFLINE = False
COST_PRICE_TIMEOUT = 15
COST_BILL_REASONING = "included"
COST_MODEL_ALIASES: dict[str, str] = {}
COST_PRICE_CATALOG: dict[str, dict[str, Decimal]] | None = None
COST_PRICE_CATALOG_ATTEMPTED = False
COST_PRICE_CATALOG_ERROR = ""


def int_token(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def first_token_value(usage: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        item: Any = usage
        for part in key.split("."):
            if not isinstance(item, dict):
                item = None
                break
            item = item.get(part)
        value = int_token(item)
        if value is not None:
            return value
    return None


def normalize_usage_tokens(usage: Any) -> dict[str, int | None]:
    if not isinstance(usage, dict):
        return {field: None for field in TOKEN_FIELDS}

    if isinstance(usage.get("agent_usage"), dict):
        return normalize_usage_tokens(usage["agent_usage"])

    input_tokens = first_token_value(
        usage,
        "input_tokens",
        "prompt_tokens",
        "totalInput",
        "input",
    )
    output_tokens = first_token_value(
        usage,
        "output_tokens",
        "completion_tokens",
        "output",
    )
    reasoning_tokens = first_token_value(
        usage,
        "reasoning_tokens",
        "reasoningTokens",
        "reasoning",
        "completion_tokens_details.reasoning_tokens",
    )
    cache_read_tokens = first_token_value(
        usage,
        "cache_read_tokens",
        "totalCacheRead",
        "cacheRead",
        "prompt_tokens_details.cached_tokens",
    )
    total_tokens = first_token_value(usage, "total_tokens", "totalTokens", "total")
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    elif total_tokens is None and cache_read_tokens is not None:
        total_tokens = cache_read_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cache_read_tokens": cache_read_tokens,
        "total_tokens": total_tokens,
    }


def agent_meta_from_raw_model_response(raw_model_response: Any) -> dict[str, Any]:
    if not isinstance(raw_model_response, dict):
        return {}
    candidates: list[Any] = [raw_model_response]
    result = raw_model_response.get("result")
    if isinstance(result, dict):
        candidates.append(result)
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        meta = candidate.get("meta") if isinstance(candidate.get("meta"), dict) else {}
        agent_meta = meta.get("agentMeta") if isinstance(meta.get("agentMeta"), dict) else {}
        if agent_meta:
            return agent_meta
    return {}


def row_raw_agent_usage(row: dict[str, Any]) -> dict[str, Any]:
    agent_meta = agent_meta_from_raw_model_response(row.get("raw_model_response"))
    usage = agent_meta.get("usage")
    return usage if isinstance(usage, dict) else {}


def row_token_fields(row: dict[str, Any]) -> dict[str, int | None]:
    raw_usage = row_raw_agent_usage(row)
    if raw_usage:
        return normalize_usage_tokens(raw_usage)
    usage = row.get("usage")
    if isinstance(usage, dict):
        return normalize_usage_tokens(usage)
    trajectory = {field: int_token(row.get("trajectory_" + field)) for field in TOKEN_FIELDS}
    if any(value is not None for value in trajectory.values()):
        return trajectory
    existing = {field: int_token(row.get(field)) for field in TOKEN_FIELDS}
    if any(value is not None for value in existing.values()):
        return existing
    return normalize_usage_tokens(usage)


def usage_token_variants(usage: Any) -> tuple[dict[str, int | None], dict[str, int | None]]:
    if not isinstance(usage, dict):
        empty = normalize_usage_tokens(None)
        return empty, empty
    trajectory = (
        normalize_usage_tokens(usage.get("agent_usage"))
        if isinstance(usage.get("agent_usage"), dict)
        else normalize_usage_tokens(usage)
    )
    last_call = (
        normalize_usage_tokens(usage.get("last_call_usage"))
        if isinstance(usage.get("last_call_usage"), dict)
        else normalize_usage_tokens(usage)
    )
    return trajectory, last_call


def sum_token_fields(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    totals = {field: 0 for field in TOKEN_FIELDS}
    for row in rows:
        fields = row_token_fields(row)
        for field in TOKEN_FIELDS:
            totals[field] += fields.get(field) or 0
    return totals


def float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def usage_cost(usage: Any) -> float:
    if not isinstance(usage, dict):
        return 0.0
    if isinstance(usage.get("agent_usage"), dict):
        return usage_cost(usage["agent_usage"])
    cost = float_value(usage.get("cost"))
    if cost:
        return cost
    cost_details = usage.get("cost_details") if isinstance(usage.get("cost_details"), dict) else {}
    return float_value(cost_details.get("upstream_inference_cost"))


def decimal_price(value: Any) -> Decimal:
    if value is None or value == "":
        return Decimal("0")
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal("0")


def fetch_openrouter_price_catalog(timeout: int) -> dict[str, Any]:
    headers = {"User-Agent": "vlm-exec-routerbench-sft-cost/1.0"}
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(OPENROUTER_MODELS_URL, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def load_openrouter_price_catalog() -> dict[str, dict[str, Decimal]]:
    global COST_PRICE_CATALOG
    global COST_PRICE_CATALOG_ATTEMPTED
    global COST_PRICE_CATALOG_ERROR

    if COST_PRICE_CATALOG is not None:
        return COST_PRICE_CATALOG
    if COST_PRICE_CATALOG_ATTEMPTED:
        return {}
    COST_PRICE_CATALOG_ATTEMPTED = True

    if not COST_ESTIMATION_ENABLED:
        COST_PRICE_CATALOG_ERROR = "disabled"
        return {}

    raw: dict[str, Any] | None = None
    if COST_PRICE_CACHE_PATH.exists() and not COST_PRICE_REFRESH:
        try:
            raw = json.loads(COST_PRICE_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            COST_PRICE_CATALOG_ERROR = f"price_cache_read_failed:{exc.__class__.__name__}"
            raw = None

    if raw is None and not COST_PRICE_OFFLINE:
        try:
            raw = fetch_openrouter_price_catalog(COST_PRICE_TIMEOUT)
            COST_PRICE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            COST_PRICE_CACHE_PATH.write_text(
                json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            COST_PRICE_CATALOG_ERROR = f"price_catalog_fetch_failed:{exc.__class__.__name__}"
            raw = None

    if raw is None:
        if not COST_PRICE_CATALOG_ERROR:
            COST_PRICE_CATALOG_ERROR = "price_catalog_unavailable"
        return {}

    prices: dict[str, dict[str, Decimal]] = {}
    for item in raw.get("data") or []:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        pricing = item.get("pricing") if isinstance(item.get("pricing"), dict) else {}
        prices[str(item["id"])] = {key: decimal_price(value) for key, value in pricing.items()}
    COST_PRICE_CATALOG = prices
    COST_PRICE_CATALOG_ERROR = ""
    return prices


def strip_openrouter_prefix(model: str) -> str:
    return model[len("openrouter/") :] if model.startswith("openrouter/") else model


def cost_model_aliases(model: str, model_ref: Any = None) -> list[str]:
    aliases = []
    try:
        api_alias = api_model_name(model) if model else ""
    except Exception:
        api_alias = ""
    for value in (
        model,
        strip_openrouter_prefix(model),
        model_ref,
        strip_openrouter_prefix(str(model_ref or "")),
        api_alias,
    ):
        value = str(value or "").strip()
        if value and value not in aliases:
            aliases.append(value)
    if "/" not in model:
        for value in (f"openai/{model}", f"qwen/{model}"):
            if value not in aliases:
                aliases.append(value)
    return aliases


def resolve_cost_price_model(model: str, model_ref: Any = None) -> str | None:
    prices = load_openrouter_price_catalog()
    if not prices:
        return None
    for alias in cost_model_aliases(model, model_ref):
        mapped = COST_MODEL_ALIASES.get(alias) or COST_MODEL_ALIASES.get(strip_openrouter_prefix(alias))
        if mapped:
            return mapped if mapped in prices else None
        if alias in prices:
            return alias
    return None


def calculate_openrouter_cost(tokens: dict[str, Any], pricing: dict[str, Decimal]) -> dict[str, Decimal]:
    input_tokens = Decimal(int_token(tokens.get("input_tokens")) or 0)
    output_tokens = Decimal(int_token(tokens.get("output_tokens")) or 0)
    reasoning_tokens = Decimal(int_token(tokens.get("reasoning_tokens")) or 0)
    cache_read_tokens = Decimal(int_token(tokens.get("cache_read_tokens")) or 0)

    prompt_price = pricing.get("prompt", Decimal("0"))
    completion_price = pricing.get("completion", Decimal("0"))
    cache_read_price = pricing.get("input_cache_read")
    reasoning_price = pricing.get("internal_reasoning", completion_price)

    if cache_read_price is not None and cache_read_tokens:
        prompt_cost = max(input_tokens - cache_read_tokens, Decimal("0")) * prompt_price
        cache_read_cost = cache_read_tokens * cache_read_price
    else:
        prompt_cost = input_tokens * prompt_price
        cache_read_cost = Decimal("0")
    completion_cost = output_tokens * completion_price
    if COST_BILL_REASONING == "separate":
        reasoning_cost = reasoning_tokens * reasoning_price
    elif COST_BILL_REASONING == "auto" and "internal_reasoning" in pricing:
        reasoning_cost = reasoning_tokens * reasoning_price
    else:
        reasoning_cost = Decimal("0")
    total = prompt_cost + completion_cost + cache_read_cost + reasoning_cost
    return {
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "cache_read_cost": cache_read_cost,
        "reasoning_cost": reasoning_cost,
        "total_cost": total,
    }


def serializable_cost_details(value: dict[str, Decimal]) -> dict[str, float]:
    return {key: float(item) for key, item in value.items()}


def estimated_usage_cost_info(
    *,
    model: str,
    tokens: dict[str, Any],
    model_ref: Any = None,
) -> dict[str, Any]:
    if not COST_ESTIMATION_ENABLED:
        return {"cost": 0.0, "source": "cost_estimation_disabled", "model": None, "details": None}
    if not any(int_token(tokens.get(field)) for field in TOKEN_FIELDS):
        return {"cost": 0.0, "source": "missing_tokens", "model": None, "details": None}
    price_model = resolve_cost_price_model(model, model_ref)
    if price_model is None:
        source = COST_PRICE_CATALOG_ERROR or "missing_price"
        return {"cost": 0.0, "source": source, "model": None, "details": None}
    details = calculate_openrouter_cost(tokens, load_openrouter_price_catalog()[price_model])
    return {
        "cost": float(details["total_cost"]),
        "source": "openrouter_price_catalog_estimate",
        "model": price_model,
        "details": serializable_cost_details(details),
    }


def usage_cost_info(model: str, usage: Any, model_ref: Any = None) -> dict[str, Any]:
    direct = usage_cost(usage)
    if direct:
        return {"cost": direct, "source": "usage", "model": None, "details": None}
    return estimated_usage_cost_info(
        model=model,
        model_ref=model_ref,
        tokens=normalize_usage_tokens(usage),
    )


def executor_metadata_candidates(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []

    containers = [value]
    raw_response = value.get("raw_model_response")
    if isinstance(raw_response, dict):
        containers.append(raw_response)
        raw_result = raw_response.get("result")
        if isinstance(raw_result, dict):
            containers.append(raw_result)

    candidates: list[dict[str, Any]] = []
    seen: set[int] = set()
    for container in containers:
        if not isinstance(container, dict):
            continue
        backend = str(container.get("executor_backend") or container.get("backend") or "")
        keys = ["executor_metadata"]
        if backend == "mini_agent":
            keys.extend(["mini_agent", "openclaw"])
        elif backend == "openclaw":
            keys.extend(["openclaw", "mini_agent"])
        else:
            keys.extend(["mini_agent", "openclaw"])
        for key in keys:
            meta = container.get(key)
            if isinstance(meta, dict) and id(meta) not in seen:
                candidates.append(meta)
                seen.add(id(meta))
        if (
            any(key in container for key in ("trajectory_model_stats", "trajectory_path", "git_diff_present"))
            and id(container) not in seen
        ):
            candidates.append(container)
            seen.add(id(container))
    return candidates


def executor_metadata_from_value(value: Any) -> dict[str, Any]:
    candidates = executor_metadata_candidates(value)
    return candidates[0] if candidates else {}


def executor_trajectory_model_stats(value: Any) -> dict[str, Any]:
    for meta in executor_metadata_candidates(value):
        stats = meta.get("trajectory_model_stats")
        if isinstance(stats, dict):
            return stats
        trajectory_path = meta.get("trajectory_path")
        if not trajectory_path:
            continue
        try:
            payload = json.loads(Path(str(trajectory_path)).read_text(encoding="utf-8"))
        except Exception:
            continue
        info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
        stats = info.get("model_stats")
        if isinstance(stats, dict):
            return stats
    return {}


def openclaw_trajectory_model_stats(value: Any) -> dict[str, Any]:
    """Backward-compatible alias for old result readers."""
    return executor_trajectory_model_stats(value)


def response_executor_cost_info(response: dict[str, Any], model: str, model_ref: Any = None) -> dict[str, Any]:
    cost = usage_cost(response.get("usage"))
    if cost:
        return {"cost": cost, "source": "usage", "model": None, "details": None}
    trajectory_cost = float_value(executor_trajectory_model_stats(response).get("instance_cost"))
    if trajectory_cost:
        return {"cost": trajectory_cost, "source": "trajectory_model_stats", "model": None, "details": None}
    return estimated_usage_cost_info(
        model=model,
        model_ref=model_ref,
        tokens=normalize_usage_tokens(response.get("usage")),
    )


def response_executor_cost(response: dict[str, Any], model: str = "", model_ref: Any = None) -> float:
    return float_value(response_executor_cost_info(response, model, model_ref).get("cost"))


def response_executor_steps(response: dict[str, Any]) -> int | None:
    return int_token(executor_trajectory_model_stats(response).get("api_calls"))


def row_executor_cost(row: dict[str, Any]) -> float:
    cost = float_value(row.get("executor_cost"))
    if cost:
        return cost
    usage = usage_cost(row.get("usage"))
    if usage:
        return usage
    trajectory_cost = float_value(executor_trajectory_model_stats(row).get("instance_cost"))
    if trajectory_cost:
        return trajectory_cost
    return float_value(
        estimated_usage_cost_info(
            model=str(row.get("candidate_model") or ""),
            model_ref=row.get("executor_model_ref") or row.get("openclaw_model_ref"),
            tokens=row_token_fields(row),
        ).get("cost")
    )


def row_executor_steps(row: dict[str, Any]) -> int:
    return int_token(row.get("executor_steps")) or int_token(executor_trajectory_model_stats(row).get("api_calls")) or 0


def row_judge_cost(row: dict[str, Any]) -> float:
    total = 0.0
    for item in row.get("judge_usage") or []:
        if not isinstance(item, dict):
            continue
        cost = float_value(item.get("cost")) or usage_cost(item.get("usage"))
        if not cost:
            cost = float_value(
                usage_cost_info(
                    str(item.get("judge_model") or ""),
                    item.get("usage"),
                ).get("cost")
            )
        total += cost
    return total


def sum_costs(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    rows = list(rows)
    executor_cost = sum(row_executor_cost(row) for row in rows)
    judge_cost = sum(row_judge_cost(row) for row in rows)
    return {
        "executor_cost": executor_cost,
        "executor_steps": sum(row_executor_steps(row) for row in rows),
        "judge_cost": judge_cost,
        "total_cost": executor_cost + judge_cost,
    }


def aggregate_usage_items(usage_items: Iterable[Any]) -> dict[str, Any]:
    totals = {field: 0 for field in TOKEN_FIELDS}
    cost = 0.0
    saw_any = False
    for usage in usage_items:
        if not isinstance(usage, dict):
            continue
        normalized = normalize_usage_tokens(usage)
        for field in TOKEN_FIELDS:
            value = normalized.get(field)
            if value is not None:
                saw_any = True
                totals[field] += value
        item_cost = usage_cost(usage)
        if item_cost:
            saw_any = True
            cost += item_cost
    if not saw_any:
        return {}
    if cost:
        totals["cost"] = cost
    return totals


def prefixed_token_fields(prefix: str, fields: dict[str, int | None]) -> dict[str, int | None]:
    return {f"{prefix}_{field}": fields.get(field) for field in TOKEN_FIELDS}


def append_judge_usage(
    judge_usage_log: JudgeUsageLog | None,
    *,
    kind: str,
    judge_model: str,
    usage: Any,
) -> None:
    if judge_usage_log is None:
        return
    normalized = normalize_usage_tokens(usage)
    cost_info = usage_cost_info(judge_model, usage)
    judge_usage_log.append(
        {
            "kind": kind,
            "judge_model": judge_model,
            "usage": usage,
            "cost": cost_info.get("cost"),
            "cost_source": cost_info.get("source"),
            "cost_model": cost_info.get("model"),
            "cost_details": cost_info.get("details"),
            **normalized,
        }
    )


def sum_judge_usage(judge_usage: Iterable[dict[str, Any]]) -> dict[str, int]:
    return sum_token_fields(judge_usage)


def judge_usage_by_model(judge_usage: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in judge_usage:
        grouped.setdefault(str(row.get("judge_model")), []).append(row)
    return {model: sum_token_fields(rows) for model, rows in sorted(grouped.items())}


def configure_cost_estimation(args: argparse.Namespace) -> None:
    global COST_ESTIMATION_ENABLED
    global COST_PRICE_CACHE_PATH
    global COST_PRICE_REFRESH
    global COST_PRICE_OFFLINE
    global COST_PRICE_TIMEOUT
    global COST_BILL_REASONING
    global COST_MODEL_ALIASES
    global COST_PRICE_CATALOG
    global COST_PRICE_CATALOG_ATTEMPTED
    global COST_PRICE_CATALOG_ERROR

    COST_ESTIMATION_ENABLED = bool(args.cost_estimation)
    COST_PRICE_CACHE_PATH = Path(args.cost_price_cache)
    if not COST_PRICE_CACHE_PATH.is_absolute():
        COST_PRICE_CACHE_PATH = ROOT / COST_PRICE_CACHE_PATH
    COST_PRICE_REFRESH = bool(args.cost_price_refresh)
    COST_PRICE_OFFLINE = bool(args.cost_price_offline)
    COST_PRICE_TIMEOUT = int(args.cost_price_timeout)
    COST_BILL_REASONING = str(args.cost_bill_reasoning)
    aliases = {}
    for item in args.cost_model_alias or []:
        if "=" not in item:
            raise ValueError(f"--cost-model-alias must be FROM=TO, got: {item}")
        left, right = item.split("=", 1)
        aliases[left.strip()] = right.strip()
    COST_MODEL_ALIASES = aliases
    COST_PRICE_CATALOG = None
    COST_PRICE_CATALOG_ATTEMPTED = False
    COST_PRICE_CATALOG_ERROR = ""


def normalized_judge_answer_hash(answer: str) -> str:
    normalized = answer.replace("\r\n", "\n").replace("\r", "\n").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def judge_cache_key(
    *,
    kind: str,
    task: dict[str, Any],
    judge_model: str,
    answer: str,
) -> tuple[str, str, str, str]:
    return (
        kind,
        str(task.get("task_id") or ""),
        judge_model,
        normalized_judge_answer_hash(answer),
    )


def judge_retry_max_tokens(judge_max_tokens: int) -> int:
    configured = int(os.environ.get("ROUTER_SFT_JUDGE_RETRY_MAX_TOKENS", "4096"))
    return max(judge_max_tokens, configured)


def parse_judge_json(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def judge_result_from_text(text: str) -> tuple[bool, str] | None:
    parsed = parse_judge_json(text)
    if parsed is None:
        return None
    return bool(parsed.get("passed")), str(parsed.get("reason", "judge_result"))


def build_choice_extraction_judge_messages(
    task: dict[str, Any],
    answer: str,
    choices: dict[str, str],
    reference: dict[str, Any],
    gold_option: str,
    rule_reason: str,
    rule_meta: dict[str, Any],
) -> list[dict[str, str]]:
    index_base = reference_answer_index_base(reference, task)
    return [
        {
            "role": "system",
            "content": (
                "You extract a multiple-choice selection from a candidate answer. "
                "Return only JSON: {\"predicted_option\": string|null, \"reason\": \"short reason\"}."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "instruction_text": task_instruction_text(task),
                    "choices": choices,
                    "reference": reference,
                    "gold_option": gold_option,
                    "gold_answer": rule_meta.get("gold_answer"),
                    "candidate_answer": answer,
                    "rule_based_result": {
                        "reason": rule_reason,
                        "predicted_option": rule_meta.get("predicted_option"),
                        "predicted_answer": rule_meta.get("predicted_answer"),
                    },
                    "numeric_option_index_base": index_base,
                    "extraction_policy": (
                        "Use only candidate_answer and choices to identify which provided option the "
                        "candidate selected. Do not solve the original question, infer from images, or "
                        "replace the candidate's answer with your own. If the candidate gives multiple "
                        "conflicting selections, use the final explicit selection. If no option is selected "
                        "or the selection is ambiguous, return predicted_option=null. Output the option label "
                        "exactly as one of the choices keys. Use reference, gold_option, and rule_based_result "
                        "only as grading context; do not change the candidate's selected option to match gold."
                    ),
                },
                ensure_ascii=False,
            ),
        },
    ]


def parse_choice_extraction_option(text: str, choices: dict[str, str], *, index_base: int | None = None) -> str | None:
    parsed = parse_judge_json(text)
    if parsed is None:
        return None

    raw_option = None
    for key in ("predicted_option", "predicted_choice", "option", "choice", "answer"):
        if key in parsed:
            raw_option = parsed.get(key)
            break
    if raw_option is None:
        return None

    option = normalize_option_label(raw_option)
    if not option or option in {"NULL", "NONE", "N/A", "NA", "UNKNOWN", "AMBIGUOUS"}:
        return None
    if option in choices:
        return option

    normalized_raw = norm_text(str(raw_option))
    if index_base is not None:
        indexed_key, _indexed_value = option_from_index_text(normalized_raw, choices, index_base=index_base)
        if indexed_key is not None:
            return indexed_key

    normalized_value = normalized_raw
    matched = [
        key
        for key, value in choices.items()
        if normalized_value and normalized_value == norm_text(value)
    ]
    if len(matched) == 1:
        return matched[0]
    indexed_key, _indexed_value = option_from_index_text(normalized_raw, choices)
    if indexed_key is not None:
        return indexed_key
    return None


def judge_multiple_choice_extraction_with_model(
    task: dict[str, Any],
    answer: str,
    choices: dict[str, str],
    reference: dict[str, Any],
    gold_option: str,
    rule_reason: str,
    rule_meta: dict[str, Any],
    judge_model: str,
    judge_max_tokens: int,
    timeout: int,
    http_transport: str,
    deepseek_thinking: str,
    judge_cache: JudgeCache | None = None,
    judge_usage_log: JudgeUsageLog | None = None,
) -> tuple[bool, str]:
    cache_answer = json.dumps(
        {
            "candidate_answer": answer,
            "choices": choices,
            "reference": reference,
            "gold_option": gold_option,
            "rule_reason": rule_reason,
            "rule_meta": rule_meta,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    cache_key = judge_cache_key(kind="choice_extract", task=task, judge_model=judge_model, answer=cache_answer)
    if judge_cache is not None and cache_key in judge_cache:
        print(f"[judge_cache] task={task.get('task_id')} judge_model={judge_model} mode=choice_extract", flush=True)
        return judge_cache[cache_key]

    print(f"[judge] task={task.get('task_id')} judge_model={judge_model} mode=choice_extract", flush=True)
    fake_input = {
        "messages": build_choice_extraction_judge_messages(
            task,
            answer,
            choices,
            reference,
            gold_option,
            rule_reason,
            rule_meta,
        ),
        "tools": [],
        "assets": [],
    }
    response = post_chat_completion(
        model=judge_model,
        executor_input=fake_input,
        temperature=0.0,
        max_tokens=judge_max_tokens,
        timeout=timeout,
        retries=1,
        retry_sleep=2.0,
        http_transport=http_transport,
        extra_body=extra_body_for_model(judge_model, deepseek_thinking),
    )
    append_judge_usage(judge_usage_log, kind="choice_extract", judge_model=judge_model, usage=response.get("usage"))
    predicted_option = parse_choice_extraction_option(
        message_text(response_message(response)),
        choices,
        index_base=reference_answer_index_base(reference, task),
    )
    if predicted_option is None:
        result = False, f"verifier_error: cannot_extract_option; gold={gold_option}; source=judge_extract"
    elif predicted_option == gold_option:
        result = True, f"option_match_judge_extract: {predicted_option}"
    else:
        result = False, f"model_wrong: predicted={predicted_option} gold={gold_option}; source=judge_extract"
    if judge_cache is not None:
        judge_cache[cache_key] = result
    return result


def judge_with_model(
    task: dict[str, Any],
    answer: str,
    judge_model: str,
    judge_max_tokens: int,
    timeout: int,
    http_transport: str,
    deepseek_thinking: str,
    judge_cache: JudgeCache | None = None,
    judge_usage_log: JudgeUsageLog | None = None,
) -> tuple[bool, str]:
    cache_key = judge_cache_key(kind="answer", task=task, judge_model=judge_model, answer=answer)
    if judge_cache is not None and cache_key in judge_cache:
        print(f"[judge_cache] task={task.get('task_id')} judge_model={judge_model}", flush=True)
        return judge_cache[cache_key]

    print(f"[judge] task={task.get('task_id')} judge_model={judge_model}", flush=True)
    fake_input = {"messages": build_judge_messages(task, answer), "tools": [], "assets": []}
    result = None
    for attempt, max_tokens in enumerate([judge_max_tokens, judge_retry_max_tokens(judge_max_tokens)]):
        if attempt > 0 and max_tokens == judge_max_tokens:
            break
        if attempt > 0:
            print(
                f"[judge_retry] task={task.get('task_id')} judge_model={judge_model} "
                f"reason=unparseable max_tokens={max_tokens}",
                flush=True,
            )
        response = post_chat_completion(
            model=judge_model,
            executor_input=fake_input,
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=1,
            retry_sleep=2.0,
            http_transport=http_transport,
            extra_body=extra_body_for_model(judge_model, deepseek_thinking),
        )
        usage_kind = "answer" if attempt == 0 else "answer_retry"
        append_judge_usage(judge_usage_log, kind=usage_kind, judge_model=judge_model, usage=response.get("usage"))
        result = judge_result_from_text(message_text(response_message(response)))
        if result is not None:
            break
    if result is None:
        return False, "judge_unparseable"
    if judge_cache is not None:
        judge_cache[cache_key] = result
    return result


def parse_judge_response(text: str) -> tuple[bool, str]:
    result = judge_result_from_text(text)
    if result is None:
        return False, "judge_unparseable"
    return result


def build_tool_judge_messages(
    task: dict[str, Any],
    message: dict[str, Any],
    strict_reason: str,
) -> list[dict[str, str]]:
    reference = task.get("verifier", {}).get("reference", {})
    tools = task.get("executor_input", {}).get("tools") or []
    return [
        {
            "role": "system",
            "content": (
                "You are a semantic verifier for tool-use tasks. "
                "Return only JSON: {\"passed\": true|false, \"reason\": \"short reason\"}."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "instruction": task.get("router_view", {}).get("instruction"),
                    "available_tools": tools,
                    "reference_expected_calls": reference.get("expected_calls") or [],
                    "observed_tool_calls": parse_tool_calls(message),
                    "assistant_text": message_text(message),
                    "strict_tool_match_reason": strict_reason,
                    "grading_policy": (
                        "Evaluate whether the observed tool calls are materially sufficient to satisfy "
                        "the user's request using the available tool schemas. Tool call order matters for "
                        "workflow tasks when later calls depend on earlier outputs. Treat "
                        "reference_expected_calls as a tool-name and argument hint; use the instruction "
                        "and any reference tool_steps to infer the required workflow order. "
                        "If reference_expected_calls appear duplicated, empty-argument, incomplete, or "
                        "in tension with the user's instruction and available tool descriptions, prioritize "
                        "the user's instruction and the tool schemas over the noisy reference. "
                        "First inspect each available tool's JSON schema. Only fields listed in the schema "
                        "can be required or judged; required arguments are exactly the fields named in the "
                        "schema's required list. If a tool schema has no properties, an empty argument object "
                        "is valid even when the reference includes positional placeholders or values that "
                        "cannot be represented by the schema. Do not require arguments that are not present "
                        "in the tool schema. Pass if the "
                        "chosen tools and arguments would reasonably obtain the requested information or "
                        "perform the requested action in a valid workflow order. Fail if a necessary tool "
                        "call is missing, the wrong tool is used for the requested action, the workflow "
                        "order is materially wrong, or schema-expressible arguments are absent "
                        "or materially wrong. Do not require extra tool calls for information that is "
                        "implicit, derivable from a prior tool call, or unnecessary to answer the user's "
                        "request."
                    ),
                },
                ensure_ascii=False,
            ),
        },
    ]


def judge_tool_with_model(
    task: dict[str, Any],
    message: dict[str, Any],
    strict_reason: str,
    judge_model: str,
    judge_max_tokens: int,
    timeout: int,
    http_transport: str,
    deepseek_thinking: str,
    judge_cache: JudgeCache | None = None,
    judge_usage_log: JudgeUsageLog | None = None,
) -> tuple[bool, str]:
    cache_answer = json.dumps(
        {
            "assistant_text": message_text(message),
            "tool_calls": parse_tool_calls(message),
            "strict_reason": strict_reason,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    cache_key = judge_cache_key(kind="tool", task=task, judge_model=judge_model, answer=cache_answer)
    if judge_cache is not None and cache_key in judge_cache:
        print(f"[judge_cache] task={task.get('task_id')} judge_model={judge_model} mode=tool", flush=True)
        return judge_cache[cache_key]

    print(f"[judge] task={task.get('task_id')} judge_model={judge_model} mode=tool", flush=True)
    fake_input = {"messages": build_tool_judge_messages(task, message, strict_reason), "tools": [], "assets": []}
    response = post_chat_completion(
        model=judge_model,
        executor_input=fake_input,
        temperature=0.0,
        max_tokens=judge_max_tokens,
        timeout=timeout,
        retries=1,
        retry_sleep=2.0,
        http_transport=http_transport,
        extra_body=extra_body_for_model(judge_model, deepseek_thinking),
    )
    append_judge_usage(judge_usage_log, kind="tool", judge_model=judge_model, usage=response.get("usage"))
    result = parse_judge_response(message_text(response_message(response)))
    if judge_cache is not None:
        judge_cache[cache_key] = result
    return result


def browsecomp_plus_llm_judge_enabled() -> bool:
    value = os.environ.get("ROUTER_SFT_BROWSECOMP_PLUS_LLM_JUDGE", "1")
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def verify(
    task: dict[str, Any],
    message: dict[str, Any],
    judge_model: str | None,
    judge_max_tokens: int,
    timeout: int,
    http_transport: str,
    deepseek_thinking: str,
    judge_cache: JudgeCache | None = None,
    judge_usage_log: JudgeUsageLog | None = None,
) -> tuple[bool, str]:
    verifier_type = task.get("verifier", {}).get("type")
    text = message_text(message)
    runtime_error_reason = runtime_error_answer_reason(text)
    if runtime_error_reason:
        return False, runtime_error_reason
    if verifier_type == "unit_tests":
        return verify_mbpp(task, text, timeout)
    if verifier_type == "python_unittest_tests":
        return verify_python_unittest(task, text, timeout)
    if verifier_type == "official_code_benchmark":
        return verify_official_code_benchmark(task, text, timeout)
    if verifier_type == "input_output_tests":
        return verify_apps(task, text, timeout)
    if verifier_type == "tool_call_match_or_llm_judge":
        passed, reason = verify_tool(task, message)
        if passed or not judge_model:
            return passed, reason
        expected = task.get("verifier", {}).get("reference", {}).get("expected_calls") or []
        if reason == "tool_call_order_mismatch" and reference_has_complete_tool_step_order(task, expected):
            return passed, reason
        return judge_tool_with_model(
            task,
            message,
            reason,
            judge_model,
            judge_max_tokens,
            timeout,
            http_transport,
            deepseek_thinking,
            judge_cache,
            judge_usage_log,
        )
    if verifier_type == "normalized_answer_or_llm_judge":
        multiple_choice = verify_multiple_choice_answer(task, text)
        if multiple_choice is not None:
            passed, reason, meta = multiple_choice
            if (
                not passed
                and judge_model
                and meta.get("gold_option")
                and (
                    reason.startswith("verifier_error: cannot_extract_option")
                    or (reason.startswith("model_wrong:") and multiple_choice_llm_recheck_enabled(task))
                )
            ):
                reference = verifier_reference(task)
                return judge_multiple_choice_extraction_with_model(
                    task,
                    text,
                    normalized_choices(reference),
                    reference,
                    str(meta["gold_option"]),
                    reason,
                    meta,
                    judge_model,
                    judge_max_tokens,
                    timeout,
                    http_transport,
                    deepseek_thinking,
                    judge_cache,
                    judge_usage_log,
                )
            return passed, reason
        passed, reason = verify_normalized_answer(task, text)
        if passed or not judge_model:
            return passed, reason
        if is_browsecomp_plus_task(task) and not browsecomp_plus_llm_judge_enabled():
            return False, reason
        if official_final_answer_requires_rule_verification(task):
            if reason == "missing_reference_answer":
                return False, f"verifier_error: {reason}"
            text = official_final_answer_for_judge(text)
        if bare_label_reference_requires_exact_match(task):
            return False, exact_reference_mismatch_reason(task)
        if task.get("category") == "multimodal_doc_visual":
            if reason.startswith("missing_"):
                return False, f"verifier_error: {reason}"
        return judge_with_model(task, text, judge_model, judge_max_tokens, timeout, http_transport, deepseek_thinking, judge_cache, judge_usage_log)
    if judge_model:
        return judge_with_model(task, text, judge_model, judge_max_tokens, timeout, http_transport, deepseek_thinking, judge_cache, judge_usage_log)
    return False, "needs_llm_judge"


def failure_type_for_result(passed: bool, reason: str) -> str | None:
    if passed:
        return None
    reason_text = str(reason)
    if reason_text.startswith("api_error"):
        return "api_error"
    if reason_text.startswith("executor_error"):
        return "executor_error"
    if reason_text.startswith("verifier_error") or reason_text.startswith("verifier_unsupported") or reason_text in {
        "missing_reference_answer",
        "missing_gold_option",
        "needs_llm_judge",
        "judge_unparseable",
    }:
        return "verifier_error"
    return "model_wrong"


def status_for_verification(passed: bool, reason: str) -> str:
    if passed:
        return "ok"
    failure_type = failure_type_for_result(passed, reason)
    if failure_type in {"api_error", "executor_error", "provider_schema_error"}:
        return "error"
    if failure_type == "verifier_error":
        return "verifier_error"
    return "model_wrong"


def verifier_unsupported_before_execution_reason(task: dict[str, Any]) -> str | None:
    verifier = task.get("verifier") if isinstance(task.get("verifier"), dict) else {}
    if verifier.get("type") != "input_output_tests":
        return None
    reference = verifier.get("reference") if isinstance(verifier.get("reference"), dict) else {}
    input_output = reference.get("input_output") if isinstance(reference.get("input_output"), dict) else {}
    if input_output.get("fn_name"):
        return None
    if apps_requires_semantic_output(task) and apps_semantic_checker_name(task) is None:
        return "verifier_unsupported: apps_semantic_output"
    return None


def executor_unsupported_before_execution_reason(task: dict[str, Any], executor_backend: str) -> str | None:
    return verifier_unsupported_before_execution_reason(task)


def verifier_unsupported_result_row(
    *,
    task: dict[str, Any],
    model: str,
    executor_backend: str,
    openclaw_model_ref: str,
    openclaw_command: str,
    openclaw_command_kind: str,
    reason: str,
) -> dict[str, Any]:
    empty_tokens = normalize_usage_tokens(None)
    return {
        "status": status_for_verification(False, reason),
        "task_id": str(task.get("task_id")),
        "candidate_model": model,
        "provider": provider_for_model(model),
        "executor_backend": executor_backend,
        "executor_model_ref": openclaw_model_ref if backend_uses_command_metadata(executor_backend) else None,
        "executor_command_kind": openclaw_command_kind if backend_uses_command_metadata(executor_backend) else None,
        "executor_command": openclaw_command if backend_uses_command_metadata(executor_backend) else None,
        "openclaw_model_ref": openclaw_model_ref if backend_uses_command_metadata(executor_backend) else None,
        "openclaw_command_kind": openclaw_command_kind if backend_uses_command_metadata(executor_backend) else None,
        "openclaw_command": openclaw_command if backend_uses_command_metadata(executor_backend) else None,
        "category": task.get("category"),
        "difficulty_prior": task.get("difficulty_prior"),
        "source_dataset": task.get("source_dataset"),
        "source_id": task.get("source_id"),
        "reference_answer": debug_reference_answer(task),
        "passed": False,
        "failure_type": failure_type_for_result(False, reason),
        "verify_reason": reason,
        "preflight_skip": True,
        "assistant_text": "",
        "tool_calls": [],
        "finish_reason": None,
        "usage": None,
        "executor_cost": 0.0,
        "executor_cost_source": "not_executed",
        "executor_cost_model": None,
        "executor_cost_details": None,
        "executor_steps": None,
        "raw_model_response": None,
        "openclaw_tool_summary": None,
        "browsecomp_plus": None,
        **empty_tokens,
        **prefixed_token_fields("trajectory", empty_tokens),
        **prefixed_token_fields("last_call", empty_tokens),
        "executor_metadata": None,
        "mini_agent": None,
        "openclaw": None,
        "judge_usage": [],
        **prefixed_token_fields("judge", empty_tokens),
        "latency_s": 0.0,
    }


def should_skip_swebench_llm_judge(task: dict[str, Any], args: argparse.Namespace) -> bool:
    verifier = task.get("verifier") if isinstance(task.get("verifier"), dict) else {}
    if verifier.get("type") != "patch_or_llm_judge":
        return False
    reference = verifier.get("reference") if isinstance(verifier.get("reference"), dict) else {}
    if not reference.get("repo") or not reference.get("base_commit"):
        return False
    return bool(args.skip_swebench_llm_judge or args.swebench_official_verify)


def is_swebench_real_repo_task(task: dict[str, Any]) -> bool:
    verifier = task.get("verifier") if isinstance(task.get("verifier"), dict) else {}
    if verifier.get("type") != "patch_or_llm_judge":
        return False
    reference = verifier.get("reference") if isinstance(verifier.get("reference"), dict) else {}
    return bool(reference.get("repo") and reference.get("base_commit"))


def is_official_swebench_real_repo_task(task: dict[str, Any]) -> bool:
    return is_swebench_real_repo_task(task) and str(task.get("source_dataset") or "") in SWEBENCH_SOURCE_DATASETS


def executor_command_for_task(task: dict[str, Any], args: argparse.Namespace) -> tuple[str, str]:
    if args.executor_backend == "mini_agent":
        if is_official_swebench_real_repo_task(task):
            return args.swebench_mini_agent_command, "swebench_real_repo"
        return args.mini_agent_command, "standard"
    if args.executor_backend != "openclaw":
        return "", ""
    if is_swebench_real_repo_task(task):
        return args.swebench_openclaw_command, "swebench_real_repo"
    return args.openclaw_command, "standard"


def openclaw_command_for_task(task: dict[str, Any], args: argparse.Namespace) -> tuple[str, str]:
    return executor_command_for_task(task, args)


def sft_row(
    task: dict[str, Any],
    selected_model: str,
    selected_executor_backend: str | None = None,
    selected_openclaw_model_ref: str | None = None,
    selected_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_result = selected_result or {}
    selected_tokens = row_token_fields(selected_result)
    usage = selected_result.get("usage")
    if isinstance(usage, dict):
        _, selected_last_call_tokens = usage_token_variants(usage)
        selected_trajectory_tokens, _ = usage_token_variants(usage)
    else:
        selected_trajectory_tokens = {
            field: int_token(selected_result.get("trajectory_" + field)) for field in TOKEN_FIELDS
        }
        if not any(value is not None for value in selected_trajectory_tokens.values()):
            selected_trajectory_tokens = selected_tokens
        selected_last_call_tokens = {
            field: int_token(selected_result.get("last_call_" + field)) for field in TOKEN_FIELDS
        }
        if not any(value is not None for value in selected_last_call_tokens.values()):
            selected_last_call_tokens = selected_tokens
    selected_judge_tokens = sum_judge_usage(selected_result.get("judge_usage") or [])
    return {
        "messages": [
            {
                "role": "user",
                "content": json.dumps(task.get("router_view"), ensure_ascii=False),
            },
            {
                "role": "assistant",
                "content": json.dumps({"selected_model": selected_model}, ensure_ascii=False),
            },
        ],
        "metadata": {
            "task_id": task.get("task_id"),
            "category": task.get("category"),
            "difficulty_prior": task.get("difficulty_prior"),
            "source_dataset": task.get("source_dataset"),
            "selected_executor_backend": selected_executor_backend,
            "selected_executor_model_ref": selected_result.get("executor_model_ref"),
            "selected_executor_command_kind": selected_result.get("executor_command_kind"),
            "selected_openclaw_model_ref": selected_openclaw_model_ref,
            "selected_openclaw_command_kind": selected_result.get("openclaw_command_kind"),
            "selected_latency_s": selected_result.get("latency_s"),
            "selected_usage": selected_result.get("usage"),
            "selected_executor_cost": row_executor_cost(selected_result),
            "selected_executor_cost_source": selected_result.get("executor_cost_source"),
            "selected_executor_cost_model": selected_result.get("executor_cost_model"),
            "selected_executor_cost_details": selected_result.get("executor_cost_details"),
            "selected_executor_steps": row_executor_steps(selected_result),
            "selected_executor_metadata": selected_result.get("executor_metadata"),
            "selected_mini_agent": selected_result.get("mini_agent"),
            "selected_raw_model_response": selected_result.get("raw_model_response"),
            "selected_openclaw_tool_summary": selected_result.get("openclaw_tool_summary"),
            "selected_input_tokens": selected_tokens["input_tokens"],
            "selected_output_tokens": selected_tokens["output_tokens"],
            "selected_reasoning_tokens": selected_tokens["reasoning_tokens"],
            "selected_cache_read_tokens": selected_tokens["cache_read_tokens"],
            "selected_total_tokens": selected_tokens["total_tokens"],
            "selected_trajectory_input_tokens": selected_trajectory_tokens["input_tokens"],
            "selected_trajectory_output_tokens": selected_trajectory_tokens["output_tokens"],
            "selected_trajectory_reasoning_tokens": selected_trajectory_tokens["reasoning_tokens"],
            "selected_trajectory_cache_read_tokens": selected_trajectory_tokens["cache_read_tokens"],
            "selected_trajectory_total_tokens": selected_trajectory_tokens["total_tokens"],
            "selected_last_call_input_tokens": selected_last_call_tokens["input_tokens"],
            "selected_last_call_output_tokens": selected_last_call_tokens["output_tokens"],
            "selected_last_call_reasoning_tokens": selected_last_call_tokens["reasoning_tokens"],
            "selected_last_call_cache_read_tokens": selected_last_call_tokens["cache_read_tokens"],
            "selected_last_call_total_tokens": selected_last_call_tokens["total_tokens"],
            "selected_judge_usage": selected_result.get("judge_usage"),
            "selected_judge_input_tokens": selected_judge_tokens["input_tokens"],
            "selected_judge_output_tokens": selected_judge_tokens["output_tokens"],
            "selected_judge_reasoning_tokens": selected_judge_tokens["reasoning_tokens"],
            "selected_judge_cache_read_tokens": selected_judge_tokens["cache_read_tokens"],
            "selected_judge_total_tokens": selected_judge_tokens["total_tokens"],
        },
    }


def row_executor_backend(row: dict[str, Any]) -> str:
    return str(row.get("executor_backend") or "raw_api")


def backend_uses_command_metadata(executor_backend: str) -> bool:
    return executor_backend in {"openclaw", "mini_agent"}


def executor_exception_failure(exc: Exception) -> tuple[str, str]:
    text = repr(exc).lower()
    if "openclaw tool policy audit failed" in text:
        return "executor_error", "executor_error: tool_policy_audit_failed"
    if "mini-swe-agent execution failed" in text or "mini-swe-agent failed" in text:
        return "executor_error", "executor_error: mini_swe_agent_failed"
    if "docker daemon" in text or "docker run" in text or "timeoutexpired" in text:
        return "executor_error", "executor_error: executor_environment_failed"
    if "api rate limit reached" in text or "rate limit" in text or "rawerror=429" in text or " 429 " in text:
        return "api_error", "api_error: rate_limit"
    if "exceeded the browsecomp-plus tool-call budget" in text:
        return "model_wrong", "tool_call_budget_exceeded"
    if "without evidence that search_browsecomp_plus executed" in text:
        return "model_wrong", "missing_browsecomp_plus_tool_call"
    if "request schema or tool payload" in text or "provider schema/tool validator" in text:
        return "provider_schema_error", "provider_schema_error"
    return "api_error", "api_error"


def filter_rows_by_backend(rows: list[dict[str, Any]], executor_backend: str) -> list[dict[str, Any]]:
    return [row for row in rows if row_executor_backend(row) == executor_backend]


def done_keys(path: Path, skip_errors: bool, rerun_failed: bool, executor_backend: str) -> set[tuple[str, str, str]]:
    keys = set()
    for row in read_jsonl(path) or []:
        row_backend = row_executor_backend(row)
        if row_backend != executor_backend:
            continue
        completed_statuses = {"ok", "model_wrong", "verifier_error", "needs_official"}
        if skip_errors:
            completed_statuses.add("error")
        if row.get("status") in completed_statuses and not (rerun_failed and row.get("passed") is False):
            keys.add((str(row.get("task_id")), str(row.get("candidate_model")), row_backend))
    return keys


def unverified_swebench_official_keys(
    rows: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    executor_backend: str,
    source_datasets: list[str],
) -> set[tuple[str, str, str]]:
    swebench_task_ids = set()
    selected_sources = set(source_datasets)
    fake_args = argparse.Namespace(skip_swebench_llm_judge=False, swebench_official_verify=True)
    for task in tasks:
        if selected_sources and str(task.get("source_dataset") or "") not in selected_sources:
            continue
        if should_skip_swebench_llm_judge(task, fake_args):
            swebench_task_ids.add(str(task.get("task_id")))

    keys = set()
    for row in rows:
        task_id = str(row.get("task_id"))
        if task_id not in swebench_task_ids:
            continue
        if row_executor_backend(row) != executor_backend:
            continue
        if isinstance(row.get("swebench_official"), dict):
            continue
        keys.add((task_id, str(row.get("candidate_model")), executor_backend))
    return keys


def load_results(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path) or [])


def solved_task_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("task_id")) for row in rows if row.get("passed") is True}


def attempted_task_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(row.get("task_id"))
        for row in rows
        if row.get("status") in {"ok", "model_wrong", "verifier_error", "error"}
    }


def build_sft_from_results(tasks: list[dict[str, Any]], rows: list[dict[str, Any]], candidate_models: list[str]) -> list[dict[str, Any]]:
    task_by_id = {task["task_id"]: task for task in tasks}
    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("passed") is True:
            rows_by_task.setdefault(str(row.get("task_id")), []).append(row)

    model_rank = {model: index for index, model in enumerate(candidate_models)}
    output = []
    for task_id, task_rows in rows_by_task.items():
        task_rows.sort(
            key=lambda row: (
                0
                if isinstance(row.get("swebench_official"), dict)
                and row.get("swebench_official", {}).get("resolved") is True
                else 1,
                model_rank.get(str(row.get("candidate_model")), 999),
            )
        )
        task = task_by_id.get(task_id)
        if task:
            selected_row = task_rows[0]
            output.append(
                sft_row(
                    task,
                    str(selected_row.get("candidate_model")),
                    row_executor_backend(selected_row),
                    selected_row.get("openclaw_model_ref"),
                    selected_row,
                )
            )
    return output


def summarize_tokens_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key)), []).append(row)
    return {name: sum_token_fields(group_rows) for name, group_rows in sorted(grouped.items())}


def summarize_costs_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key)), []).append(row)
    return {name: sum_costs(group_rows) for name, group_rows in sorted(grouped.items())}


def all_judge_usage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        judge_usage = row.get("judge_usage")
        if isinstance(judge_usage, list):
            output.extend(item for item in judge_usage if isinstance(item, dict))
    return output


def summarize_results(rows: list[dict[str, Any]], sft_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_status = Counter(str(row.get("status")) for row in rows)
    by_model = Counter(str(row.get("candidate_model")) for row in rows)
    by_model_passed = Counter(str(row.get("candidate_model")) for row in rows if row.get("passed") is True)
    by_category = Counter(str(row.get("category")) for row in rows)
    by_category_passed = Counter(str(row.get("category")) for row in rows if row.get("passed") is True)
    by_failure_type = Counter(str(row.get("failure_type")) for row in rows if row.get("failure_type"))
    by_reason = Counter(str(row.get("verify_reason")) for row in rows)
    by_executor_cost_source = Counter(str(row.get("executor_cost_source") or "unknown") for row in rows)
    sft_by_category = Counter(str(row.get("metadata", {}).get("category")) for row in sft_rows)
    sft_by_selected_model = Counter(
        json.loads(row["messages"][1]["content"]).get("selected_model")
        for row in sft_rows
    )
    judge_usage = all_judge_usage(rows)
    return {
        "executor_results": len(rows),
        "sft_rows": len(sft_rows),
        "by_status": dict(sorted(by_status.items())),
        "by_model": dict(sorted(by_model.items())),
        "by_model_passed": dict(sorted(by_model_passed.items())),
        "by_category": dict(sorted(by_category.items())),
        "by_category_passed": dict(sorted(by_category_passed.items())),
        "by_failure_type": dict(sorted(by_failure_type.items())),
        "by_verify_reason": dict(sorted(by_reason.items())),
        "by_executor_cost_source": dict(sorted(by_executor_cost_source.items())),
        "token_totals": sum_token_fields(rows),
        "tokens_by_model": summarize_tokens_by(rows, "candidate_model"),
        "tokens_by_provider": summarize_tokens_by(rows, "provider"),
        "tokens_by_category": summarize_tokens_by(rows, "category"),
        "judge_token_totals": sum_judge_usage(judge_usage),
        "judge_tokens_by_model": judge_usage_by_model(judge_usage),
        "cost_totals": sum_costs(rows),
        "costs_by_model": summarize_costs_by(rows, "candidate_model"),
        "costs_by_provider": summarize_costs_by(rows, "provider"),
        "costs_by_category": summarize_costs_by(rows, "category"),
        "sft_by_category": dict(sorted(sft_by_category.items())),
        "sft_by_selected_model": dict(sorted(sft_by_selected_model.items())),
    }


def path_relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def task_source_dataset(task: dict[str, Any]) -> str:
    return str(task.get("source_dataset") or "unknown")


def result_source_dataset(row: dict[str, Any], task_sources: dict[str, str]) -> str:
    return str(row.get("source_dataset") or task_sources.get(str(row.get("task_id")), "unknown"))


def sft_source_dataset(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("source_dataset") or "unknown")


def group_items_by_source(
    items: Iterable[dict[str, Any]],
    source_fn,
) -> dict[str, list[dict[str, Any]]]:
    grouped_items: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped_items.setdefault(source_fn(item), []).append(item)
    return grouped_items


def execution_config_for_summary(
    *,
    args: argparse.Namespace,
    candidate_models: list[str],
    openclaw_model_ref_map: dict[str, str],
    final_result_path: Path,
) -> dict[str, Any]:
    return {
        "candidate_models": candidate_models,
        "budget_policy": args.budget_policy,
        "policy_model_limits_by_difficulty": BUDGET_POLICIES[args.budget_policy],
        "run_all": args.run_all,
        "max_workers": args.max_workers,
        "judge_model": args.judge_model,
        "judge_max_tokens": args.judge_max_tokens,
        "cost_estimation": args.cost_estimation,
        "cost_price_cache": args.cost_price_cache,
        "cost_price_refresh": args.cost_price_refresh,
        "cost_price_offline": args.cost_price_offline,
        "cost_bill_reasoning": args.cost_bill_reasoning,
        "cost_model_aliases": args.cost_model_alias,
        "http_transport": args.http_transport,
        "executor_backend": args.executor_backend,
        "openclaw_command": args.openclaw_command if args.executor_backend == "openclaw" else None,
        "mini_agent_command": args.mini_agent_command if args.executor_backend == "mini_agent" else None,
        "swebench_openclaw_command": args.swebench_openclaw_command if args.executor_backend == "openclaw" else None,
        "swebench_mini_agent_command": args.swebench_mini_agent_command if args.executor_backend == "mini_agent" else None,
        "executor_model_ref_map": openclaw_model_ref_map if backend_uses_command_metadata(args.executor_backend) else None,
        "openclaw_model_ref_map": openclaw_model_ref_map if backend_uses_command_metadata(args.executor_backend) else None,
        "deepseek_thinking": args.deepseek_thinking,
        "rerun_failed": args.rerun_failed,
        "task_ids": args.task_ids,
        "source_datasets": args.source_datasets,
        "skip_swebench_llm_judge": args.skip_swebench_llm_judge or args.swebench_official_verify,
        "swebench_official_verify": args.swebench_official_verify,
        "swebench_official_backend": (
            "sbcli_export_only"
            if args.swebench_official_verify and swebench_official_uses_sbcli(args)
            else "harness"
            if args.swebench_official_verify
            else None
        ),
        "swebench_official_python": args.swebench_official_python if args.swebench_official_verify else None,
        "swebench_official_dataset": args.swebench_official_dataset if args.swebench_official_verify else None,
        "swebench_official_split": args.swebench_official_split if args.swebench_official_verify else None,
        "swebench_official_modal": args.swebench_official_modal if args.swebench_official_verify else None,
        "swebench_official_dedupe_instance": args.swebench_official_dedupe_instance if args.swebench_official_verify else None,
        "swebench_official_parallel": args.swebench_official_parallel if args.swebench_official_verify else None,
        "swebench_official_parallel_batch_size": (
            args.swebench_official_parallel_batch_size if args.swebench_official_verify else None
        ),
        "swebench_official_parallel_flush_interval": (
            args.swebench_official_parallel_flush_interval if args.swebench_official_verify else None
        ),
        "swebench_official_results_out": path_relative_to_root(final_result_path) if args.swebench_official_verify else None,
        "per_dataset_output_dir": args.per_dataset_output_dir or None,
    }


def write_per_dataset_outputs(
    *,
    output_dir: Path,
    tasks: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    sft_rows: list[dict[str, Any]],
    execution_config: dict[str, Any],
) -> Path:
    task_sources = {str(task.get("task_id")): task_source_dataset(task) for task in tasks}
    tasks_by_source = group_items_by_source(tasks, task_source_dataset)
    rows_by_source = group_items_by_source(rows, lambda row: result_source_dataset(row, task_sources))
    sft_by_source = group_items_by_source(sft_rows, sft_source_dataset)
    sources = sorted(set(tasks_by_source) | set(rows_by_source) | set(sft_by_source))

    output_dir.mkdir(parents=True, exist_ok=True)
    index_items = []
    for source in sources:
        source_dir = output_dir / safe_name(source)
        source_tasks = tasks_by_source.get(source, [])
        source_rows = rows_by_source.get(source, [])
        source_sft_rows = sft_by_source.get(source, [])
        results_path = source_dir / "executor_results.jsonl"
        sft_path = source_dir / "router_sft.jsonl"
        summary_path = source_dir / "summary.json"
        write_jsonl(results_path, source_rows)
        write_jsonl(sft_path, source_sft_rows)
        source_summary = summarize_results(source_rows, source_sft_rows)
        source_summary["source_dataset"] = source
        source_summary["task_count"] = len(source_tasks)
        source_summary["task_ids"] = [task.get("task_id") for task in source_tasks]
        source_summary["execution_config"] = execution_config
        source_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(source_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        index_items.append(
            {
                "source_dataset": source,
                "directory": path_relative_to_root(source_dir),
                "tasks": len(source_tasks),
                "executor_results": len(source_rows),
                "sft_rows": len(source_sft_rows),
                "results_path": path_relative_to_root(results_path),
                "sft_path": path_relative_to_root(sft_path),
                "summary_path": path_relative_to_root(summary_path),
            }
        )

    index_path = output_dir / "summary.json"
    index_path.write_text(
        json.dumps(
            {
                "mode": "per_dataset_outputs",
                "source_count": len(sources),
                "sources": index_items,
                "execution_config": execution_config,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return index_path


def swebench_reference(task: dict[str, Any]) -> dict[str, Any]:
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
    try:
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle, 1):
                if index == raw_line:
                    return json.loads(line)
    except Exception:
        return {}
    return {}


def swebench_instance_id_for_task(task: dict[str, Any]) -> str | None:
    reference = swebench_reference(task)
    for value in (reference.get("instance_id"), task.get("instance_id"), task.get("source_id")):
        if value:
            return str(value)
    raw = raw_row_from_source_ref(task)
    for key in ("instance_id", "id"):
        if raw.get(key):
            return str(raw[key])
    return None


def model_from_prediction_file(path: Path) -> str | None:
    rows = list(read_jsonl(path) or [])
    if not rows:
        return None
    model = rows[0].get("model_name_or_path")
    return str(model) if model else None


def is_swebench_multimodal_dataset(value: Any) -> bool:
    return str(value or "") == SWEBENCH_MULTIMODAL_DATASET


def swebench_official_uses_sbcli(args: argparse.Namespace) -> bool:
    if is_swebench_multimodal_dataset(args.swebench_official_dataset):
        return True
    return any(is_swebench_multimodal_dataset(source) for source in swebench_official_source_datasets(args))


def sbcli_json_path_for_prediction_file(path: Path) -> Path:
    return path.with_name(path.stem + "_sbcli.json")


def write_sbcli_prediction_json(jsonl_path: Path) -> Path:
    rows = list(read_jsonl(jsonl_path) or [])
    output_path = sbcli_json_path_for_prediction_file(jsonl_path)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def instance_ids_from_prediction_file(path: Path) -> list[str]:
    instance_ids = []
    for row in read_jsonl(path) or []:
        instance_id = row.get("instance_id")
        if instance_id:
            instance_ids.append(str(instance_id))
    return instance_ids


def official_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"pass", "passed", "resolved", "success", "successful", "true"}:
            return True
        if normalized in {"fail", "failed", "unresolved", "error", "false"}:
            return False
    return None


def official_status_from_node(node: dict[str, Any]) -> tuple[bool | None, str | None]:
    for key in ("resolved", "passed", "pass", "success", "successful"):
        if key in node:
            value = official_bool(node.get(key))
            if value is not None:
                return value, key
    for key in ("status", "result", "verdict"):
        if key in node:
            value = official_bool(node.get(key))
            if value is not None:
                return value, key
    return None, None


def collect_official_results(node: Any, inherited_instance_id: str | None = None) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    if isinstance(node, dict):
        for key in ("resolved_ids", "resolved_instances", "resolved"):
            value = node.get(key)
            if isinstance(value, list):
                for instance_id in value:
                    found[str(instance_id)] = {
                        "resolved": True,
                        "status_key": key,
                        "raw": {key: value},
                    }
        for key in ("unresolved_ids", "unresolved_instances", "failed_instances"):
            value = node.get(key)
            if isinstance(value, list):
                for instance_id in value:
                    found[str(instance_id)] = {
                        "resolved": False,
                        "status_key": key,
                        "raw": {key: value},
                    }
        for key in ("error_ids", "errored_ids", "error_instances", "errored_instances"):
            value = node.get(key)
            if isinstance(value, list):
                for instance_id in value:
                    found[str(instance_id)] = {
                        "resolved": None,
                        "status_key": key,
                        "error": "official_harness_error",
                        "raw": {key: value},
                    }
        instance_id = str(node.get("instance_id") or node.get("task_id") or inherited_instance_id or "")
        resolved, source_key = official_status_from_node(node)
        if instance_id and resolved is not None:
            found[instance_id] = {
                "resolved": resolved,
                "status_key": source_key,
                "raw": node,
            }
        for key, value in node.items():
            child_instance_id = str(key) if isinstance(value, dict) and key else instance_id or None
            found.update(collect_official_results(value, child_instance_id))
    elif isinstance(node, list):
        for item in node:
            found.update(collect_official_results(item, inherited_instance_id))
    return found


def load_official_results_from_path(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return results
    candidates = [path] if path.is_file() else sorted(path.rglob("*.json")) + sorted(path.rglob("*.jsonl"))
    for candidate in candidates:
        try:
            if candidate.suffix == ".jsonl":
                nodes = list(read_jsonl(candidate) or [])
            else:
                nodes = [json.loads(candidate.read_text(encoding="utf-8"))]
        except Exception:
            continue
        for node in nodes:
            results.update(collect_official_results(node))
    return results


def load_official_report_paths_from_text(text: str) -> list[Path]:
    paths = []
    for match in re.finditer(r"Report written to\s+([^\s]+\.json)", str(text or "")):
        path = Path(match.group(1))
        paths.append(path if path.is_absolute() else ROOT / path)
    return paths


def load_official_results_from_text(text: str) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    text = str(text or "").strip()
    if not text:
        return results
    for match in re.finditer(r"EvaluationError:\s+([A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+):\s+([^\n]+)", text):
        instance_id, error = match.groups()
        results[instance_id] = {
            "resolved": None,
            "status_key": "evaluation_error",
            "error": error.strip(),
            "raw": {"evaluation_error": error.strip()},
        }
    try:
        results.update(collect_official_results(json.loads(text)))
    except json.JSONDecodeError:
        pass
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith(("{", "[")):
            continue
        try:
            results.update(collect_official_results(json.loads(line)))
        except json.JSONDecodeError:
            continue
    return results


def merge_swebench_official_results(
    *,
    rows: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    official_by_model: dict[str, dict[str, dict[str, Any]]],
    submitted_task_keys: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    task_by_id = {str(task.get("task_id")): task for task in tasks}
    merged = []
    for row in rows:
        task = task_by_id.get(str(row.get("task_id")))
        model = str(row.get("candidate_model") or "")
        submitted = (str(row.get("task_id")), model) in submitted_task_keys
        instance_id = swebench_instance_id_for_task(task) if task else None
        official = official_by_model.get(model, {}).get(str(instance_id)) if submitted and instance_id else None
        row = dict(row)
        if official:
            resolved_value = official.get("resolved")
            row["swebench_official"] = {
                "instance_id": instance_id,
                "resolved": resolved_value,
                "status_key": official.get("status_key"),
                "error": official.get("error"),
                "raw": official.get("raw"),
            }
            if resolved_value is None or official.get("error"):
                row["passed"] = False
                row["status"] = "verifier_error"
                row["failure_type"] = "verifier_error"
                row["verify_reason"] = "verifier_error: swebench_official_harness_error"
            else:
                resolved = bool(resolved_value)
                row["passed"] = resolved
                row["status"] = "ok" if resolved else "model_wrong"
                row["failure_type"] = None if resolved else "model_wrong"
                row["verify_reason"] = "swebench_official_resolved" if resolved else "model_wrong: swebench_official_unresolved"
        elif submitted and instance_id and model in official_by_model:
            row["swebench_official"] = {
                "instance_id": instance_id,
                "resolved": None,
                "error": "official_result_missing",
            }
            if row.get("status") in {"ok", "needs_official"}:
                row["status"] = "verifier_error"
                row["passed"] = False
                row["failure_type"] = "verifier_error"
                row["verify_reason"] = "verifier_error: swebench_official_result_missing"
        merged.append(row)
    return merged


def export_swebench_official_predictions(
    *,
    result_path: Path,
    task_path: Path,
    output_path: Path,
    source_datasets: list[str],
    candidate_models: list[str],
    statuses: list[str],
    dedupe_instance: str,
) -> tuple[list[Path], set[tuple[str, str]]]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "export_swebench_official_predictions.py"),
        "--results",
        str(result_path),
        "--tasks",
        str(task_path),
        "--output",
        str(output_path),
        "--split-by-model",
    ]
    for source_dataset in source_datasets:
        cmd.extend(["--source-dataset", source_dataset])
    for status in statuses:
        cmd.extend(["--status", status])
    for model in candidate_models:
        cmd.extend(["--candidate-model", model])
    cmd.extend(["--dedupe-instance", dedupe_instance])
    cmd.append("--allow-empty")
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"SWE-bench official prediction export failed: {detail[:2000]}")
    index_path = output_path.with_name(output_path.stem + "_index.json")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    files = []
    submitted_task_keys: set[tuple[str, str]] = set()
    for item in index.get("files") or []:
        path = Path(str(item.get("output")))
        files.append(path if path.is_absolute() else ROOT / path)
        summary_path = Path(str(item.get("summary") or ""))
        if summary_path and not summary_path.is_absolute():
            summary_path = ROOT / summary_path
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in summary.get("items") or []:
            task_id = str(row.get("task_id") or "")
            model = str(row.get("candidate_model") or "")
            if task_id and model:
                submitted_task_keys.add((task_id, model))
    return files, submitted_task_keys


def docker_daemon_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def run_swebench_harness(
    *,
    python_executable: str,
    dataset_name: str,
    split: str,
    predictions_path: Path,
    run_id: str,
    max_workers: int,
    timeout: int | None,
    modal: bool,
    namespace: str,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        *shlex.split(python_executable),
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
    ]
    if timeout is not None:
        cmd.extend(["--timeout", str(timeout)])
    if namespace:
        cmd.extend(["--namespace", namespace])
    if modal:
        cmd.extend(["--modal", "True"])
    elif not docker_daemon_available():
        detail = (
            "docker_unavailable: SWE-bench official harness requires a reachable Docker daemon "
            "unless --swebench-official-modal is used. Start Docker, set DOCKER_HOST, or rerun "
            "with --swebench-official-modal."
        )
        return subprocess.CompletedProcess(cmd, 125, "", detail)
    env = os.environ.copy()
    env.setdefault("CONDA_SOLVER", "classic")
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False, env=env)


def run_swebench_official_prediction_files(
    *,
    args: argparse.Namespace,
    prediction_files: list[Path],
    run_id_prefix: str,
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    official_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    run_records = []
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    for prediction_file in prediction_files:
        model = model_from_prediction_file(prediction_file)
        if not model:
            continue
        run_id = run_id_prefix or args.swebench_official_run_id or f"openclaw_swebench_{run_stamp}"
        if len(prediction_files) > 1:
            run_id = f"{run_id}_{safe_name(model)}"
        print(f"[swebench_official] model={model} predictions={prediction_file} run_id={run_id}", flush=True)
        result = run_swebench_harness(
            python_executable=args.swebench_official_python,
            dataset_name=args.swebench_official_dataset,
            split=args.swebench_official_split,
            predictions_path=prediction_file,
            run_id=run_id,
            max_workers=args.swebench_official_max_workers,
            timeout=args.swebench_official_timeout,
            modal=args.swebench_official_modal,
            namespace=args.swebench_official_namespace,
        )
        results_root = ROOT / "logs" / "run_evaluation" / run_id
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
            detail_tail = detail[-2000:]
            model_results = load_official_results_from_path(results_root)
            for report_path in load_official_report_paths_from_text(result.stdout):
                model_results.update(load_official_results_from_path(report_path))
            model_results.update(load_official_results_from_text(result.stdout + "\n" + result.stderr))
            if not model_results:
                raw_error = {
                    "returncode": result.returncode,
                    "error": detail[:4000],
                    "error_tail": detail_tail,
                }
                model_results = {
                    instance_id: {
                        "resolved": None,
                        "status_key": "harness_returncode",
                        "error": "official_harness_error",
                        "raw": raw_error,
                    }
                    for instance_id in instance_ids_from_prediction_file(prediction_file)
                }
            print(
                f"[swebench_official_error] model={model} official_results={len(model_results)} "
                f"reason_tail={detail_tail}",
                flush=True,
            )
        else:
            model_results = load_official_results_from_path(results_root)
            for report_path in load_official_report_paths_from_text(result.stdout):
                model_results.update(load_official_results_from_path(report_path))
            if not model_results:
                model_results = load_official_results_from_text(result.stdout)
        official_by_model.setdefault(model, {}).update(model_results)
        run_records.append(
            {
                "model": model,
                "prediction_file": str(prediction_file),
                "run_id": run_id,
                "results_root": str(results_root),
                "returncode": result.returncode,
                "harness_error": (result.stderr.strip() or result.stdout.strip())[:4000] if result.returncode else None,
                "official_results": len(model_results),
                "stdout_tail": result.stdout[-4000:],
                "stderr_tail": result.stderr[-4000:],
            }
        )
        print(f"[swebench_official_done] model={model} official_results={len(model_results)}", flush=True)
    return official_by_model, run_records


def run_swebench_official_verify_stage(
    *,
    args: argparse.Namespace,
    task_path: Path,
    tasks: list[dict[str, Any]],
    result_path: Path,
    candidate_models: list[str],
) -> Path:
    predictions_base = ROOT / args.swebench_official_predictions_out
    source_datasets = swebench_official_source_datasets(args)
    prediction_files, submitted_task_keys = export_swebench_official_predictions(
        result_path=result_path,
        task_path=task_path,
        output_path=predictions_base,
        source_datasets=source_datasets,
        candidate_models=candidate_models,
        statuses=["needs_official"],
        dedupe_instance=args.swebench_official_dedupe_instance,
    )
    run_id_prefix = args.swebench_official_run_id or f"openclaw_swebench_{time.strftime('%Y%m%d_%H%M%S')}"
    official_by_model, run_records = run_swebench_official_prediction_files(
        args=args,
        prediction_files=prediction_files,
        run_id_prefix=run_id_prefix,
    )

    merged_rows = merge_swebench_official_results(
        rows=filter_rows_by_backend(load_results(result_path), args.executor_backend),
        tasks=tasks,
        official_by_model=official_by_model,
        submitted_task_keys=submitted_task_keys,
    )
    output_path = ROOT / args.swebench_official_results_out
    write_jsonl(output_path, merged_rows)
    meta_path = output_path.with_name(output_path.stem + "_runs.json")
    meta_path.write_text(json.dumps({"runs": run_records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[swebench_official_merge] wrote {output_path.relative_to(ROOT)}")
    print(f"[swebench_official_runs] wrote {meta_path.relative_to(ROOT)}")
    return output_path


class SWEBenchOfficialParallelVerifier:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        task_path: Path,
        tasks: list[dict[str, Any]],
        result_path: Path,
        candidate_models: list[str],
    ) -> None:
        self.args = args
        self.task_path = task_path
        self.tasks = tasks
        self.result_path = result_path
        self.candidate_models = candidate_models
        self.source_datasets = set(swebench_official_source_datasets(args))
        self.batch_size = max(1, int(args.swebench_official_parallel_batch_size))
        self.flush_interval_s = max(1.0, float(args.swebench_official_parallel_flush_interval))
        self.output_path = ROOT / args.swebench_official_results_out
        self.predictions_base = ROOT / args.swebench_official_predictions_out
        self.condition = threading.Condition()
        self.pending_rows: list[dict[str, Any]] = []
        self.first_pending_at: float | None = None
        self.stopping = False
        self.thread = threading.Thread(target=self._worker, name="swebench-official-verifier", daemon=True)
        self.official_by_model: dict[str, dict[str, dict[str, Any]]] = {}
        self.submitted_task_keys: set[tuple[str, str]] = set()
        self.run_records: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.batch_index = 0

    def start(self) -> None:
        self.thread.start()
        print(
            f"[swebench_official_parallel] enabled batch_size={self.batch_size} "
            f"flush_interval_s={self.flush_interval_s:g}",
            flush=True,
        )

    def enqueue(self, row: dict[str, Any]) -> None:
        if row.get("status") != "needs_official":
            return
        if str(row.get("source_dataset") or "") not in self.source_datasets:
            return
        if is_swebench_multimodal_dataset(row.get("source_dataset")):
            return
        with self.condition:
            if not self.pending_rows:
                self.first_pending_at = time.time()
            self.pending_rows.append(dict(row))
            self.condition.notify()

    def finish(self) -> Path:
        with self.condition:
            self.stopping = True
            self.condition.notify()
        self.thread.join()
        rows = filter_rows_by_backend(load_results(self.result_path), self.args.executor_backend)
        merged_rows = merge_swebench_official_results(
            rows=rows,
            tasks=self.tasks,
            official_by_model=self.official_by_model,
            submitted_task_keys=self.submitted_task_keys,
        )
        write_jsonl(self.output_path, merged_rows)
        meta_path = self.output_path.with_name(self.output_path.stem + "_runs.json")
        meta_path.write_text(
            json.dumps(
                {
                    "mode": "parallel_harness",
                    "batch_size": self.batch_size,
                    "flush_interval_s": self.flush_interval_s,
                    "runs": self.run_records,
                    "errors": self.errors,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[swebench_official_merge] wrote {self.output_path.relative_to(ROOT)}")
        print(f"[swebench_official_runs] wrote {meta_path.relative_to(ROOT)}")
        return self.output_path

    def _next_batch(self) -> list[dict[str, Any]] | None:
        with self.condition:
            while True:
                if self.pending_rows and (
                    self.stopping
                    or len(self.pending_rows) >= self.batch_size
                    or (
                        self.first_pending_at is not None
                        and time.time() - self.first_pending_at >= self.flush_interval_s
                    )
                ):
                    batch = self.pending_rows[: self.batch_size]
                    del self.pending_rows[: self.batch_size]
                    self.first_pending_at = time.time() if self.pending_rows else None
                    return batch
                if self.stopping:
                    return None
                timeout = None
                if self.pending_rows and self.first_pending_at is not None:
                    timeout = max(0.1, self.flush_interval_s - (time.time() - self.first_pending_at))
                self.condition.wait(timeout)

    def _worker(self) -> None:
        while True:
            batch_rows = self._next_batch()
            if batch_rows is None:
                return
            self.batch_index += 1
            try:
                self._run_batch(batch_rows, self.batch_index)
            except Exception as exc:
                error = f"{exc.__class__.__name__}: {exc}"
                self.errors.append(error)
                print(f"[swebench_official_parallel_error] batch={self.batch_index} error={error}", flush=True)

    def _run_batch(self, batch_rows: list[dict[str, Any]], batch_index: int) -> None:
        batch_label = f"batch{batch_index:04d}"
        batch_result_path = self.predictions_base.with_name(
            f"{self.predictions_base.stem}_{batch_label}_results.jsonl"
        )
        batch_predictions_base = self.predictions_base.with_name(
            f"{self.predictions_base.stem}_{batch_label}.jsonl"
        )
        write_jsonl(batch_result_path, batch_rows)
        prediction_files, submitted_task_keys = export_swebench_official_predictions(
            result_path=batch_result_path,
            task_path=self.task_path,
            output_path=batch_predictions_base,
            source_datasets=sorted(self.source_datasets),
            candidate_models=self.candidate_models,
            statuses=["needs_official"],
            dedupe_instance=self.args.swebench_official_dedupe_instance,
        )
        if not prediction_files:
            return
        run_id_base = self.args.swebench_official_run_id or f"openclaw_swebench_{time.strftime('%Y%m%d_%H%M%S')}"
        run_id_prefix = f"{run_id_base}_{batch_label}"
        print(
            f"[swebench_official_parallel_batch] batch={batch_index} rows={len(batch_rows)} "
            f"prediction_files={len(prediction_files)}",
            flush=True,
        )
        official_by_model, run_records = run_swebench_official_prediction_files(
            args=self.args,
            prediction_files=prediction_files,
            run_id_prefix=run_id_prefix,
        )
        for model, model_results in official_by_model.items():
            self.official_by_model.setdefault(model, {}).update(model_results)
        self.submitted_task_keys.update(submitted_task_keys)
        self.run_records.extend({**record, "parallel_batch": batch_index} for record in run_records)


def run_swebench_sbcli_export_stage(
    *,
    args: argparse.Namespace,
    task_path: Path,
    result_path: Path,
    candidate_models: list[str],
) -> Path:
    predictions_base = ROOT / args.swebench_official_predictions_out
    source_datasets = swebench_official_source_datasets(args)
    prediction_files, submitted_task_keys = export_swebench_official_predictions(
        result_path=result_path,
        task_path=task_path,
        output_path=predictions_base,
        source_datasets=source_datasets,
        candidate_models=candidate_models,
        statuses=["needs_official"],
        dedupe_instance=args.swebench_official_dedupe_instance,
    )
    exports = []
    for prediction_file in prediction_files:
        model = model_from_prediction_file(prediction_file)
        if not model:
            continue
        sbcli_path = write_sbcli_prediction_json(prediction_file)
        row_count = len(list(read_jsonl(prediction_file) or []))
        run_id_base = args.swebench_official_run_id or f"openclaw_swebench_mm_{time.strftime('%Y%m%d_%H%M%S')}"
        run_id = run_id_base if len(prediction_files) == 1 else f"{run_id_base}_{safe_name(model)}"
        submit_command = [
            "sb-cli",
            "submit",
            SWEBENCH_MULTIMODAL_SBCLI_SUBSET,
            args.swebench_official_split,
            "--predictions_path",
            str(sbcli_path),
            "--run_id",
            run_id,
        ]
        exports.append(
            {
                "model": model,
                "prediction_jsonl": str(prediction_file),
                "prediction_json": str(sbcli_path),
                "rows": row_count,
                "subset": SWEBENCH_MULTIMODAL_SBCLI_SUBSET,
                "split": args.swebench_official_split,
                "run_id": run_id,
                "submitted_task_keys": [
                    {"task_id": task_id, "candidate_model": candidate_model}
                    for task_id, candidate_model in sorted(submitted_task_keys)
                    if candidate_model == model
                ],
                "submit_command": submit_command,
            }
        )
        print(
            f"[swebench_sbcli_export] model={model} rows={row_count} "
            f"json={sbcli_path.relative_to(ROOT)}",
            flush=True,
        )

    meta_path = ROOT / args.swebench_official_results_out
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {
                "mode": "sbcli_export_only",
                "dataset": args.swebench_official_dataset,
                "source_datasets": source_datasets,
                "exports": exports,
                "note": (
                    "SWE-bench Multimodal official test evaluation is submitted through sb-cli. "
                    "This stage only exported sb-cli-compatible JSON files and did not submit them."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[swebench_sbcli_exports] wrote {meta_path.relative_to(ROOT)}")
    for item in exports:
        print("[swebench_sbcli_submit_hint] " + " ".join(shlex.quote(part) for part in item["submit_command"]))
    return result_path


def swebench_official_source_datasets(args: argparse.Namespace) -> list[str]:
    return args.swebench_official_source_dataset or [args.swebench_official_dataset]


def fill_default_swebench_official_outputs(args: argparse.Namespace) -> None:
    result_path = Path(str(args.results_out))
    output_dir = result_path.parent
    stem = result_path.stem or "executor_results"
    if not args.swebench_official_predictions_out:
        args.swebench_official_predictions_out = str(output_dir / f"{stem}_official_predictions.jsonl")
    if not args.swebench_official_results_out:
        args.swebench_official_results_out = str(output_dir / f"{stem}_official_merged.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run candidate executors and build router SFT labels from verified results.")
    parser.add_argument("--tasks", default="manifests/executor_task_pool_v1_seed.jsonl")
    parser.add_argument("--results-out", default="runs/executor_results_v1_seed.jsonl")
    parser.add_argument("--sft-out", default="sft/router_sft_v1_seed.jsonl")
    parser.add_argument("--summary-out", default="sft/router_sft_v1_seed_summary.json")
    parser.add_argument(
        "--per-dataset-output-dir",
        default="",
        help=(
            "Optional directory for per-source_dataset result bundles. Writes one subdirectory per "
            "dataset with executor_results.jsonl, router_sft.jsonl, and summary.json, plus an index summary."
        ),
    )
    parser.add_argument("--candidate-model", action="append", dest="candidate_models", default=None)
    parser.add_argument(
        "--category",
        choices=["all", "code_debug_edit", "tool_workflow", "multimodal_doc_visual"],
        default="all",
    )
    parser.add_argument("--target-sft-rows", type=int, default=1500)
    parser.add_argument("--task-id", action="append", dest="task_ids", default=None)
    parser.add_argument(
        "--source-dataset",
        action="append",
        dest="source_datasets",
        default=None,
        help="Only run tasks whose source_dataset matches this value. Repeat for multiple sources.",
    )
    parser.add_argument("--limit-tasks", type=int, default=None)
    parser.add_argument("--limit-calls", type=int, default=None)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Run executor calls in parallel. Values >1 require --run-all.",
    )
    parser.add_argument("--budget-policy", choices=sorted(BUDGET_POLICIES), default="full")
    parser.add_argument(
        "--run-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run every eligible candidate model for each task. This is the default; "
            "use --no-run-all to stop after the first verified passing model."
        ),
    )
    parser.add_argument("--build-sft-only", action="store_true")
    parser.add_argument("--api-smoke", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--executor-backend", choices=["raw_api", "openclaw", "mini_agent"], default="raw_api")
    parser.add_argument(
        "--openclaw-command",
        default=os.environ.get("OPENCLAW_EXECUTOR_COMMAND", DEFAULT_OPENCLAW_COMMAND),
        help=(
            "Command template used when --executor-backend=openclaw. "
            "Placeholders: {input}, {output}, {model}, {provider}, {task_id}, "
            "{category}, {timeout}, {temperature}, {max_tokens}, {executor_model_ref}, "
            "and the legacy {openclaw_model_ref} alias."
        ),
    )
    parser.add_argument(
        "--mini-agent-command",
        default=os.environ.get("MINI_AGENT_EXECUTOR_COMMAND", DEFAULT_MINI_AGENT_COMMAND),
        help=(
            "Command template used when --executor-backend=mini_agent for non-SWE-bench tasks. "
            "SWE-bench Lite/Multilingual/Multimodal real-repo tasks use --swebench-mini-agent-command. "
            "Placeholders: {input}, {output}, {model}, {provider}, {task_id}, {category}, "
            "{timeout}, {temperature}, {max_tokens}, {executor_model_ref}, and the legacy "
            "{openclaw_model_ref} alias."
        ),
    )
    parser.add_argument(
        "--swebench-openclaw-command",
        default=os.environ.get("SWEBENCH_OPENCLAW_EXECUTOR_COMMAND", DEFAULT_SWEBENCH_OPENCLAW_COMMAND),
        help=(
            "Command template used for SWE-bench real-repo patch tasks when "
            "--executor-backend=openclaw. Defaults to run_swebench_openclaw_executor.py; "
            "uses the same placeholders as --openclaw-command."
        ),
    )
    parser.add_argument(
        "--swebench-mini-agent-command",
        default=(
            os.environ.get("SWEBENCH_MINI_AGENT_COMMAND")
            or os.environ.get("SWEBENCH_MINI_SWE_AGENT_COMMAND")
            or DEFAULT_SWEBENCH_MINI_AGENT_COMMAND
        ),
        help=(
            "Command template used for SWE-bench real-repo patch tasks when "
            "--executor-backend=mini_agent. Defaults to run_swebench_mini_agent_executor.py; "
            "placeholders include {input}, {output}, {model}, {provider}, {task_id}, {category}, "
            "{timeout}, {temperature}, {max_tokens}, {executor_model_ref}, and the legacy "
            "{openclaw_model_ref} alias."
        ),
    )
    parser.add_argument(
        "--openclaw-model-ref-map",
        default=os.environ.get("OPENCLAW_MODEL_REF_MAP", ""),
        help=(
            "JSON object or path mapping candidate model names to command executor model refs. "
            "OpenClaw refs are still accepted for the public OpenClaw backend, "
            'for example {"qwen/qwen3-vl-8b-instruct":"openrouter/qwen/qwen3-vl-8b-instruct"}.'
        ),
    )
    parser.add_argument(
        "--openclaw-keep-io",
        action="store_true",
        help="Persist command executor input/output/stdout/stderr under runs/openclaw_io for debugging.",
    )
    parser.add_argument("--judge-model", default=os.environ.get("ROUTER_SFT_JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--cost-estimation",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("ROUTER_SFT_COST_ESTIMATION", "1") != "0",
        help="Estimate monetary cost from token usage when the provider response does not include cost.",
    )
    parser.add_argument(
        "--cost-price-cache",
        default=os.environ.get("ROUTER_SFT_COST_PRICE_CACHE", DEFAULT_OPENROUTER_PRICE_CACHE),
        help="OpenRouter model price catalog cache used for cost estimation.",
    )
    parser.add_argument(
        "--cost-price-refresh",
        action="store_true",
        help="Fetch a fresh OpenRouter price catalog even if --cost-price-cache exists.",
    )
    parser.add_argument(
        "--cost-price-offline",
        action="store_true",
        help="Use --cost-price-cache only; do not fetch OpenRouter prices.",
    )
    parser.add_argument("--cost-price-timeout", type=int, default=15)
    parser.add_argument(
        "--cost-bill-reasoning",
        choices=["included", "separate", "auto"],
        default=os.environ.get("ROUTER_SFT_COST_BILL_REASONING", "included"),
        help=(
            "How to estimate reasoning_tokens. Default assumes reasoning is included in output_tokens; "
            "auto bills reasoning only when the price catalog exposes internal_reasoning."
        ),
    )
    parser.add_argument(
        "--cost-model-alias",
        action="append",
        default=[],
        metavar="FROM=TO",
        help="Map a local/candidate model name to an OpenRouter price catalog id. Repeat as needed.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--http-transport", choices=["curl", "urllib"], default="curl")
    parser.add_argument("--deepseek-thinking", choices=["disabled", "enabled"], default="disabled")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-swebench-llm-judge",
        action="store_true",
        help=(
            "For SWE-bench patch tasks, skip the LLM judge and mark rows needs_official. "
            "This is implied by --swebench-official-verify."
        ),
    )
    parser.add_argument(
        "--swebench-official-verify",
        action="store_true",
        help="After executor generation, run SWE-bench official harness and write merged official results.",
    )
    parser.add_argument("--swebench-official-python", default=DEFAULT_SWEBENCH_OFFICIAL_PYTHON)
    parser.add_argument("--swebench-official-dataset", default=os.environ.get("SWEBENCH_OFFICIAL_DATASET", "princeton-nlp/SWE-bench_Lite"))
    parser.add_argument("--swebench-official-split", default=os.environ.get("SWEBENCH_OFFICIAL_SPLIT", "test"))
    parser.add_argument("--swebench-official-source-dataset", action="append", default=[])
    parser.add_argument(
        "--swebench-official-predictions-out",
        default=os.environ.get("SWEBENCH_OFFICIAL_PREDICTIONS_OUT", ""),
        help=(
            "Base predictions path; official verify writes one per-model file next to it. "
            "Defaults to <results-out stem>_official_predictions.jsonl in the results-out directory."
        ),
    )
    parser.add_argument(
        "--swebench-official-results-out",
        default=os.environ.get("SWEBENCH_OFFICIAL_RESULTS_OUT", ""),
        help=(
            "Merged executor results with SWE-bench official resolved/unresolved fields. "
            "Defaults to <results-out stem>_official_merged.jsonl in the results-out directory."
        ),
    )
    parser.add_argument("--swebench-official-run-id", default="")
    parser.add_argument("--swebench-official-max-workers", type=int, default=4)
    parser.add_argument("--swebench-official-timeout", type=int, default=None)
    parser.add_argument(
        "--swebench-official-parallel",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("SWEBENCH_OFFICIAL_PARALLEL", "0") == "1",
        help=(
            "Run SWE-bench official harness in a background batch worker while executor calls continue. "
            "Only applies to non-multimodal SWE-bench; Multimodal still uses sb-cli export at the end."
        ),
    )
    parser.add_argument(
        "--swebench-official-parallel-batch-size",
        type=int,
        default=int(os.environ.get("SWEBENCH_OFFICIAL_PARALLEL_BATCH_SIZE", "8")),
        help="Number of newly generated needs_official rows per background official verify batch.",
    )
    parser.add_argument(
        "--swebench-official-parallel-flush-interval",
        type=float,
        default=float(os.environ.get("SWEBENCH_OFFICIAL_PARALLEL_FLUSH_INTERVAL", "300")),
        help="Flush a partial background official verify batch after this many seconds.",
    )
    parser.add_argument(
        "--swebench-official-dedupe-instance",
        choices=["error", "first", "best"],
        default=os.environ.get("SWEBENCH_OFFICIAL_DEDUPE_INSTANCE", "best"),
        help="How to dedupe repeated selected rows for the same SWE-bench instance before official harness export.",
    )
    parser.add_argument(
        "--swebench-official-modal",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("SWEBENCH_OFFICIAL_MODAL", "1") != "0",
        help="Run SWE-bench official harness through Modal by default; use --no-swebench-official-modal for local Docker.",
    )
    parser.add_argument("--swebench-official-namespace", default="")
    args = parser.parse_args()
    fill_default_swebench_official_outputs(args)
    configure_cost_estimation(args)
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")
    if args.max_workers > 1 and not args.run_all:
        raise ValueError("--max-workers > 1 requires --run-all; --no-run-all depends on serial stop-after-pass behavior.")
    if args.swebench_official_parallel_batch_size < 1:
        raise ValueError("--swebench-official-parallel-batch-size must be >= 1")
    if args.swebench_official_parallel_flush_interval <= 0:
        raise ValueError("--swebench-official-parallel-flush-interval must be > 0")

    task_path = ROOT / args.tasks
    result_path = ROOT / args.results_out
    sft_path = ROOT / args.sft_out
    candidate_models = args.candidate_models or list(CANDIDATE_MODELS)
    openclaw_model_ref_map = load_openclaw_model_ref_map(args.openclaw_model_ref_map)
    if args.api_smoke:
        run_api_smoke(
            candidate_models,
            args.max_tokens,
            args.timeout,
            args.retries,
            args.retry_sleep,
            args.http_transport,
            args.deepseek_thinking,
        )
        return

    tasks = list(read_jsonl(task_path) or [])
    if args.category != "all":
        tasks = [task for task in tasks if task.get("category") == args.category]
    if args.source_datasets:
        source_datasets = {str(source) for source in args.source_datasets}
        tasks = [task for task in tasks if str(task.get("source_dataset") or "") in source_datasets]
    if args.task_ids:
        task_ids = {str(task_id) for task_id in args.task_ids}
        tasks = [task for task in tasks if str(task.get("task_id")) in task_ids]
    tasks, dropped_malformed_tasks = filter_malformed_tasks(tasks)
    if dropped_malformed_tasks:
        by_source = Counter(str(task.get("source_dataset") or "unknown") for task in dropped_malformed_tasks)
        print(
            "[filter] dropped_malformed_tasks="
            f"{len(dropped_malformed_tasks)} reason=malformed_mmmu_image_choice by_source={dict(sorted(by_source.items()))}",
            flush=True,
        )
    if args.limit_tasks is not None:
        tasks = tasks[: args.limit_tasks]

    existing_rows = filter_rows_to_tasks(
        filter_rows_by_backend(load_results(result_path), args.executor_backend),
        tasks,
    )
    completed = done_keys(
        result_path,
        skip_errors=args.skip_errors,
        rerun_failed=args.rerun_failed,
        executor_backend=args.executor_backend,
    )
    solved = solved_task_ids(existing_rows)
    attempted = attempted_task_ids(existing_rows)
    if args.swebench_official_verify:
        unverified_official_keys = unverified_swebench_official_keys(
            existing_rows,
            tasks,
            args.executor_backend,
            swebench_official_source_datasets(args),
        )
        completed -= unverified_official_keys
        solved -= {task_id for task_id, _model, _backend in unverified_official_keys}
    judge_cache: JudgeCache = {}
    calls = 0
    official_parallel_verifier = None
    if (
        args.swebench_official_verify
        and args.swebench_official_parallel
        and not args.build_sft_only
        and not args.dry_run
        and not swebench_official_uses_sbcli(args)
    ):
        official_parallel_verifier = SWEBenchOfficialParallelVerifier(
            args=args,
            task_path=task_path,
            tasks=tasks,
            result_path=result_path,
            candidate_models=candidate_models,
        )
        official_parallel_verifier.start()

    if not args.build_sft_only and args.max_workers > 1:
        pending_calls: list[tuple[dict[str, Any], str]] = []
        for task in tasks:
            task_id = str(task["task_id"])
            task_models = candidate_models
            for model in task_models:
                if args.limit_calls is not None and len(pending_calls) >= args.limit_calls:
                    break
                if (task_id, model, args.executor_backend) in completed:
                    continue
                unsupported_reason = executor_unsupported_before_execution_reason(task, args.executor_backend)
                if unsupported_reason:
                    openclaw_model_ref = executor_model_ref_for(model, openclaw_model_ref_map)
                    effective_openclaw_command, openclaw_command_kind = executor_command_for_task(task, args)
                    print(
                        f"[skip] task={task_id} model={model} reason={unsupported_reason}",
                        flush=True,
                    )
                    if not args.dry_run:
                        append_jsonl(
                            result_path,
                            verifier_unsupported_result_row(
                                task=task,
                                model=model,
                                executor_backend=args.executor_backend,
                                openclaw_model_ref=openclaw_model_ref,
                                openclaw_command=effective_openclaw_command,
                                openclaw_command_kind=openclaw_command_kind,
                                reason=unsupported_reason,
                            ),
                        )
                        completed.add((task_id, model, args.executor_backend))
                    continue
                pending_calls.append((task, model))
            if args.limit_calls is not None and len(pending_calls) >= args.limit_calls:
                break

        calls = len(pending_calls)
        result_lock = threading.Lock()
        judge_cache_lock = threading.Lock()

        def run_one_parallel(task: dict[str, Any], model: str) -> tuple[str, str, bool]:
            task_id = str(task["task_id"])
            openclaw_model_ref = executor_model_ref_for(model, openclaw_model_ref_map)
            effective_openclaw_command, openclaw_command_kind = executor_command_for_task(task, args)
            print(
                f"[call] task={task_id} category={task.get('category')} "
                f"difficulty={task.get('difficulty_prior')} model={model} "
                f"backend={args.executor_backend} "
                f"executor_command_kind={openclaw_command_kind if backend_uses_command_metadata(args.executor_backend) else ''} "
                f"executor_model_ref={openclaw_model_ref if backend_uses_command_metadata(args.executor_backend) else ''} "
                f"executor_command={debug_snippet(effective_openclaw_command, 240) if backend_uses_command_metadata(args.executor_backend) else ''} "
                f"policy={args.budget_policy}",
                flush=True,
            )
            if args.dry_run:
                return task_id, model, False

            started = time.time()
            try:
                response = run_executor(
                    task=task,
                    model=model,
                    executor_backend=args.executor_backend,
                    openclaw_command=effective_openclaw_command,
                    openclaw_model_ref=openclaw_model_ref,
                    openclaw_keep_io=args.openclaw_keep_io,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    retries=args.retries,
                    retry_sleep=args.retry_sleep,
                    http_transport=args.http_transport,
                    deepseek_thinking=args.deepseek_thinking,
                )
                choices = response.get("choices") or []
                finish_reason = choices[0].get("finish_reason") if choices else None
                print(f"[model_done] task={task_id} model={model} finish_reason={finish_reason}", flush=True)
                message = response_message(response)
                judge_usage_log: JudgeUsageLog = []
                if should_skip_swebench_llm_judge(task, args):
                    executor_meta = executor_metadata_from_value(response)
                    if executor_meta.get("swebench_real_repo") and not executor_meta.get("git_diff_present"):
                        passed, reason = False, "model_wrong: no_git_diff"
                    else:
                        passed, reason = False, "swebench_official_pending"
                else:
                    with judge_cache_lock:
                        passed, reason = verify(
                            task,
                            message,
                            args.judge_model,
                            args.judge_max_tokens,
                            args.timeout,
                            args.http_transport,
                            args.deepseek_thinking,
                            judge_cache,
                            judge_usage_log,
                        )
                print(
                    f"[verify] task={task_id} model={model} passed={passed} reason={reason}",
                    flush=True,
                )
                usage = response.get("usage")
                usage_tokens = normalize_usage_tokens(usage)
                trajectory_tokens, last_call_tokens = usage_token_variants(usage)
                executor_cost_info = response_executor_cost_info(response, model, openclaw_model_ref)
                executor_cost = float_value(executor_cost_info.get("cost"))
                executor_steps = response_executor_steps(response)
                raw_model_response = raw_model_response_for_row(response)
                executor_metadata = executor_metadata_from_value(response)
                tool_summary = openclaw_tool_summary(response)
                row_tool_calls = result_tool_calls_for_row(message, tool_summary)
                judge_tokens = sum_judge_usage(judge_usage_log)
                assistant_text = message_text(message)
                mc_result = verify_multiple_choice_answer(task, assistant_text)
                mc_meta = mc_result[2] if mc_result is not None else {}
                failure_type = failure_type_for_result(passed, reason)
                status = "needs_official" if reason == "swebench_official_pending" else status_for_verification(passed, reason)
                if reason == "swebench_official_pending":
                    failure_type = None
                row = {
                    "status": status,
                    "task_id": task_id,
                    "candidate_model": model,
                    "provider": provider_for_model(model),
                    "executor_backend": args.executor_backend,
                    "executor_model_ref": openclaw_model_ref if backend_uses_command_metadata(args.executor_backend) else None,
                    "executor_command_kind": openclaw_command_kind if backend_uses_command_metadata(args.executor_backend) else None,
                    "executor_command": effective_openclaw_command if backend_uses_command_metadata(args.executor_backend) else None,
                    "openclaw_model_ref": openclaw_model_ref if backend_uses_command_metadata(args.executor_backend) else None,
                    "openclaw_command_kind": openclaw_command_kind if backend_uses_command_metadata(args.executor_backend) else None,
                    "openclaw_command": effective_openclaw_command if backend_uses_command_metadata(args.executor_backend) else None,
                    "category": task.get("category"),
                    "difficulty_prior": task.get("difficulty_prior"),
                    "source_dataset": task.get("source_dataset"),
                    "source_id": task.get("source_id"),
                    "reference_answer": debug_reference_answer(task),
                    "passed": passed,
                    "failure_type": failure_type,
                    "verify_reason": reason,
                    "assistant_text": assistant_text,
                    **mc_meta,
                    "tool_calls": row_tool_calls,
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "executor_cost": executor_cost,
                    "executor_cost_source": executor_cost_info.get("source"),
                    "executor_cost_model": executor_cost_info.get("model"),
                    "executor_cost_details": executor_cost_info.get("details"),
                    "executor_steps": executor_steps,
                    "raw_model_response": raw_model_response,
                    "openclaw_tool_summary": tool_summary,
                    "browsecomp_plus": response.get("browsecomp_plus") if isinstance(response.get("browsecomp_plus"), dict) else None,
                    **usage_tokens,
                    **prefixed_token_fields("trajectory", trajectory_tokens),
                    **prefixed_token_fields("last_call", last_call_tokens),
                    "executor_metadata": executor_metadata or None,
                    "mini_agent": response.get("mini_agent") if isinstance(response.get("mini_agent"), dict) else None,
                    "openclaw": response.get("openclaw") if isinstance(response.get("openclaw"), dict) else None,
                    "judge_usage": judge_usage_log,
                    **prefixed_token_fields("judge", judge_tokens),
                    "latency_s": round(time.time() - started, 3),
                }
                with result_lock:
                    append_jsonl(result_path, row)
                    if official_parallel_verifier is not None:
                        official_parallel_verifier.enqueue(row)
                    completed.add((task_id, model, args.executor_backend))
                    if passed:
                        solved.add(task_id)
                return task_id, model, passed
            except Exception as exc:
                print(f"[error] task={task_id} model={model} error={exc!r}", flush=True)
                failure_type, verify_reason = executor_exception_failure(exc)
                error_status = "model_wrong" if failure_type == "model_wrong" else "error"
                row = {
                    "status": error_status,
                    "task_id": task_id,
                    "candidate_model": model,
                    "provider": provider_for_model(model),
                    "executor_backend": args.executor_backend,
                    "executor_model_ref": openclaw_model_ref if backend_uses_command_metadata(args.executor_backend) else None,
                    "executor_command_kind": openclaw_command_kind if backend_uses_command_metadata(args.executor_backend) else None,
                    "executor_command": effective_openclaw_command if backend_uses_command_metadata(args.executor_backend) else None,
                    "openclaw_model_ref": openclaw_model_ref if backend_uses_command_metadata(args.executor_backend) else None,
                    "openclaw_command_kind": openclaw_command_kind if backend_uses_command_metadata(args.executor_backend) else None,
                    "openclaw_command": effective_openclaw_command if backend_uses_command_metadata(args.executor_backend) else None,
                    "category": task.get("category"),
                    "difficulty_prior": task.get("difficulty_prior"),
                    "source_dataset": task.get("source_dataset"),
                    "source_id": task.get("source_id"),
                    "reference_answer": debug_reference_answer(task),
                    "passed": False,
                    "failure_type": failure_type,
                    "verify_reason": verify_reason,
                    "error": repr(exc),
                    "raw_model_response": None,
                    "executor_cost": 0.0,
                    "executor_cost_source": "executor_error",
                    "executor_cost_model": None,
                    "executor_cost_details": None,
                    "executor_steps": None,
                    **normalize_usage_tokens(None),
                    "judge_usage": [],
                    **prefixed_token_fields("judge", normalize_usage_tokens(None)),
                    "latency_s": round(time.time() - started, 3),
                }
                with result_lock:
                    append_jsonl(result_path, row)
                    completed.add((task_id, model, args.executor_backend))
                return task_id, model, False

        if args.dry_run:
            for task, model in pending_calls:
                run_one_parallel(task, model)
        else:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [executor.submit(run_one_parallel, task, model) for task, model in pending_calls]
                for future in as_completed(futures):
                    future.result()

    elif not args.build_sft_only:
        for task in tasks:
            task_id = str(task["task_id"])
            if not args.run_all and task_id in solved:
                continue
            if args.budget_policy == "escalate-unsolved" and task_id not in attempted:
                continue
            task_models = candidate_models if args.run_all else candidate_models_for_task(task, candidate_models, args.budget_policy)
            for model in task_models:
                if args.limit_calls is not None and calls >= args.limit_calls:
                    break
                if (task_id, model, args.executor_backend) in completed:
                    continue
                if not args.run_all and task_id in solved:
                    break
                openclaw_model_ref = executor_model_ref_for(model, openclaw_model_ref_map)
                effective_openclaw_command, openclaw_command_kind = executor_command_for_task(task, args)
                unsupported_reason = executor_unsupported_before_execution_reason(task, args.executor_backend)
                if unsupported_reason:
                    print(
                        f"[skip] task={task_id} model={model} reason={unsupported_reason}",
                        flush=True,
                    )
                    if not args.dry_run:
                        append_jsonl(
                            result_path,
                            verifier_unsupported_result_row(
                                task=task,
                                model=model,
                                executor_backend=args.executor_backend,
                                openclaw_model_ref=openclaw_model_ref,
                                openclaw_command=effective_openclaw_command,
                                openclaw_command_kind=openclaw_command_kind,
                                reason=unsupported_reason,
                            ),
                        )
                        completed.add((task_id, model, args.executor_backend))
                    continue

                print(
                    f"[call] task={task_id} category={task.get('category')} "
                    f"difficulty={task.get('difficulty_prior')} model={model} "
                    f"backend={args.executor_backend} "
                    f"executor_command_kind={openclaw_command_kind if backend_uses_command_metadata(args.executor_backend) else ''} "
                    f"executor_model_ref={openclaw_model_ref if backend_uses_command_metadata(args.executor_backend) else ''} "
                    f"executor_command={debug_snippet(effective_openclaw_command, 240) if backend_uses_command_metadata(args.executor_backend) else ''} "
                    f"policy={args.budget_policy}",
                    flush=True,
                )
                calls += 1
                if args.dry_run:
                    continue

                started = time.time()
                try:
                    response = run_executor(
                        task=task,
                        model=model,
                        executor_backend=args.executor_backend,
                        openclaw_command=effective_openclaw_command,
                        openclaw_model_ref=openclaw_model_ref,
                        openclaw_keep_io=args.openclaw_keep_io,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                        retries=args.retries,
                        retry_sleep=args.retry_sleep,
                        http_transport=args.http_transport,
                        deepseek_thinking=args.deepseek_thinking,
                    )
                    choices = response.get("choices") or []
                    finish_reason = choices[0].get("finish_reason") if choices else None
                    print(f"[model_done] task={task_id} model={model} finish_reason={finish_reason}", flush=True)
                    message = response_message(response)
                    judge_usage_log: JudgeUsageLog = []
                    if should_skip_swebench_llm_judge(task, args):
                        executor_meta = executor_metadata_from_value(response)
                        if executor_meta.get("swebench_real_repo") and not executor_meta.get("git_diff_present"):
                            passed, reason = False, "model_wrong: no_git_diff"
                        else:
                            passed, reason = False, "swebench_official_pending"
                    else:
                        passed, reason = verify(
                            task,
                            message,
                            args.judge_model,
                            args.judge_max_tokens,
                            args.timeout,
                            args.http_transport,
                            args.deepseek_thinking,
                            judge_cache,
                            judge_usage_log,
                        )
                    print(
                        f"[verify] task={task_id} model={model} passed={passed} reason={reason}",
                        flush=True,
                    )
                    usage = response.get("usage")
                    usage_tokens = normalize_usage_tokens(usage)
                    trajectory_tokens, last_call_tokens = usage_token_variants(usage)
                    executor_cost_info = response_executor_cost_info(response, model, openclaw_model_ref)
                    executor_cost = float_value(executor_cost_info.get("cost"))
                    executor_steps = response_executor_steps(response)
                    raw_model_response = raw_model_response_for_row(response)
                    executor_metadata = executor_metadata_from_value(response)
                    tool_summary = openclaw_tool_summary(response)
                    row_tool_calls = result_tool_calls_for_row(message, tool_summary)
                    judge_tokens = sum_judge_usage(judge_usage_log)
                    assistant_text = message_text(message)
                    mc_result = verify_multiple_choice_answer(task, assistant_text)
                    mc_meta = mc_result[2] if mc_result is not None else {}
                    failure_type = failure_type_for_result(passed, reason)
                    status = "needs_official" if reason == "swebench_official_pending" else status_for_verification(passed, reason)
                    if reason == "swebench_official_pending":
                        failure_type = None
                    row = {
                        "status": status,
                        "task_id": task_id,
                        "candidate_model": model,
                        "provider": provider_for_model(model),
                        "executor_backend": args.executor_backend,
                        "executor_model_ref": openclaw_model_ref if backend_uses_command_metadata(args.executor_backend) else None,
                        "executor_command_kind": openclaw_command_kind if backend_uses_command_metadata(args.executor_backend) else None,
                        "executor_command": effective_openclaw_command if backend_uses_command_metadata(args.executor_backend) else None,
                        "openclaw_model_ref": openclaw_model_ref if backend_uses_command_metadata(args.executor_backend) else None,
                        "openclaw_command_kind": openclaw_command_kind if backend_uses_command_metadata(args.executor_backend) else None,
                        "openclaw_command": effective_openclaw_command if backend_uses_command_metadata(args.executor_backend) else None,
                        "category": task.get("category"),
                        "difficulty_prior": task.get("difficulty_prior"),
                        "source_dataset": task.get("source_dataset"),
                        "source_id": task.get("source_id"),
                        "reference_answer": debug_reference_answer(task),
                        "passed": passed,
                        "failure_type": failure_type,
                        "verify_reason": reason,
                        "assistant_text": assistant_text,
                        **mc_meta,
                        "tool_calls": row_tool_calls,
                        "finish_reason": finish_reason,
                        "usage": usage,
                        "executor_cost": executor_cost,
                        "executor_cost_source": executor_cost_info.get("source"),
                        "executor_cost_model": executor_cost_info.get("model"),
                        "executor_cost_details": executor_cost_info.get("details"),
                        "executor_steps": executor_steps,
                        "raw_model_response": raw_model_response,
                        "openclaw_tool_summary": tool_summary,
                        "browsecomp_plus": response.get("browsecomp_plus") if isinstance(response.get("browsecomp_plus"), dict) else None,
                        **usage_tokens,
                        **prefixed_token_fields("trajectory", trajectory_tokens),
                        **prefixed_token_fields("last_call", last_call_tokens),
                        "executor_metadata": executor_metadata or None,
                        "mini_agent": response.get("mini_agent") if isinstance(response.get("mini_agent"), dict) else None,
                        "openclaw": response.get("openclaw") if isinstance(response.get("openclaw"), dict) else None,
                        "judge_usage": judge_usage_log,
                        **prefixed_token_fields("judge", judge_tokens),
                        "latency_s": round(time.time() - started, 3),
                    }
                    append_jsonl(result_path, row)
                    if official_parallel_verifier is not None:
                        official_parallel_verifier.enqueue(row)
                    completed.add((task_id, model, args.executor_backend))
                    if passed:
                        solved.add(task_id)
                except Exception as exc:
                    print(f"[error] task={task_id} model={model} error={exc!r}", flush=True)
                    failure_type, verify_reason = executor_exception_failure(exc)
                    error_status = "model_wrong" if failure_type == "model_wrong" else "error"
                    append_jsonl(
                        result_path,
                        {
                            "status": error_status,
                            "task_id": task_id,
                            "candidate_model": model,
                            "provider": provider_for_model(model),
                            "executor_backend": args.executor_backend,
                            "executor_model_ref": openclaw_model_ref if backend_uses_command_metadata(args.executor_backend) else None,
                            "executor_command_kind": openclaw_command_kind if backend_uses_command_metadata(args.executor_backend) else None,
                            "executor_command": effective_openclaw_command if backend_uses_command_metadata(args.executor_backend) else None,
                            "openclaw_model_ref": openclaw_model_ref if backend_uses_command_metadata(args.executor_backend) else None,
                            "openclaw_command_kind": openclaw_command_kind if backend_uses_command_metadata(args.executor_backend) else None,
                            "openclaw_command": effective_openclaw_command if backend_uses_command_metadata(args.executor_backend) else None,
                            "category": task.get("category"),
                            "difficulty_prior": task.get("difficulty_prior"),
                            "source_dataset": task.get("source_dataset"),
                            "source_id": task.get("source_id"),
                            "reference_answer": debug_reference_answer(task),
                            "passed": False,
                            "failure_type": failure_type,
                            "verify_reason": verify_reason,
                            "error": repr(exc),
                            "raw_model_response": None,
                            "executor_cost": 0.0,
                            "executor_cost_source": "executor_error",
                            "executor_cost_model": None,
                            "executor_cost_details": None,
                            "executor_steps": None,
                            **normalize_usage_tokens(None),
                            "judge_usage": [],
                            **prefixed_token_fields("judge", normalize_usage_tokens(None)),
                            "latency_s": round(time.time() - started, 3),
                        },
                    )
                    completed.add((task_id, model, args.executor_backend))
            if args.limit_calls is not None and calls >= args.limit_calls:
                break

    if not args.dry_run:
        final_result_path = result_path
        if args.swebench_official_verify:
            if swebench_official_uses_sbcli(args):
                final_result_path = run_swebench_sbcli_export_stage(
                    args=args,
                    task_path=task_path,
                    result_path=result_path,
                    candidate_models=candidate_models,
                )
            elif official_parallel_verifier is not None:
                final_result_path = official_parallel_verifier.finish()
            else:
                final_result_path = run_swebench_official_verify_stage(
                    args=args,
                    task_path=task_path,
                    tasks=tasks,
                    result_path=result_path,
                    candidate_models=candidate_models,
                )
        final_rows = filter_rows_to_tasks(
            filter_rows_by_backend(load_results(final_result_path), args.executor_backend),
            tasks,
        )
        solved = solved_task_ids(final_rows)
        sft_rows = build_sft_from_results(tasks, final_rows, candidate_models)
        selected_sft_rows = sft_rows[: args.target_sft_rows]
        write_jsonl(sft_path, selected_sft_rows)
        execution_config = execution_config_for_summary(
            args=args,
            candidate_models=candidate_models,
            openclaw_model_ref_map=openclaw_model_ref_map,
            final_result_path=final_result_path,
        )
        summary = summarize_results(final_rows, selected_sft_rows)
        summary["execution_config"] = execution_config
        summary_path = ROOT / args.summary_out
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        if args.per_dataset_output_dir:
            per_dataset_index = write_per_dataset_outputs(
                output_dir=ROOT / args.per_dataset_output_dir,
                tasks=tasks,
                rows=final_rows,
                sft_rows=selected_sft_rows,
                execution_config=execution_config,
            )
            print(f"[per_dataset] wrote {path_relative_to_root(per_dataset_index)}")
        print(f"[sft] wrote {min(len(sft_rows), args.target_sft_rows)} rows to {path_relative_to_root(sft_path)}")
        print(f"[summary] wrote {path_relative_to_root(summary_path)}")
    print(f"[done] new_calls={calls} solved_tasks={len(solved)} target_sft_rows={args.target_sft_rows}")


if __name__ == "__main__":
    main()
