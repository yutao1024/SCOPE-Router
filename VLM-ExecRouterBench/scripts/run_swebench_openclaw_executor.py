#!/usr/bin/env python3
"""Run a SWE-bench-style OpenClaw task inside a real repository checkout.

This wrapper keeps the existing RouterSFT executor contract:
  * read one executor payload from --input,
  * write a chat-completion-like JSON object to --output.

The difference from run_openclaw_executor.py is that this script prepares a
real repository workspace from verifier.reference.repo/base_commit, runs
OpenClaw with that workspace as cwd, then captures `git diff` as the candidate
patch. Use it for stricter SWE-bench calibration runs, not for cheap bulk
pre-filtering.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DELEGATE = Path(__file__).with_name("run_openclaw_executor.py")
OPENCLAW_BOOTSTRAP_FILES = (
    "AGENTS.md",
    "BOOTSTRAP.md",
    "HEARTBEAT.md",
    "IDENTITY.md",
    "SOUL.md",
    "TOOLS.md",
    "USER.md",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 300,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"{cmd[0]} failed: {detail[:2000]}")
    return result


def subprocess_error_detail(stdout: str, stderr: str, fallback: str, max_chars: int = 4000) -> str:
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


def verifier_reference(payload: dict[str, Any]) -> dict[str, Any]:
    verifier = payload.get("verifier") if isinstance(payload.get("verifier"), dict) else {}
    reference = verifier.get("reference") if isinstance(verifier.get("reference"), dict) else {}
    return reference


def safe_repo_name(repo: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", repo).strip("_") or "repo"


def safe_openclaw_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "-", value).strip("-") or "agent"


def repo_url(repo: str) -> str:
    if repo.startswith("http://") or repo.startswith("https://") or repo.startswith("git@"):
        return repo
    return f"https://github.com/{repo}.git"


def prepare_repo_workspace(
    *,
    repo: str,
    base_commit: str,
    work_root: Path,
    fetch_timeout: int,
) -> Path:
    if not repo or not base_commit:
        raise RuntimeError("SWE-bench real-repo mode requires verifier.reference.repo and base_commit.")

    repos_dir = work_root / "_repo_cache"
    runs_dir = work_root / "runs"
    repos_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    mirror = repos_dir / f"{safe_repo_name(repo)}.git"
    if not mirror.exists():
        run_cmd(["git", "clone", "--mirror", repo_url(repo), str(mirror)], timeout=fetch_timeout)
    else:
        run_cmd(["git", "remote", "update", "--prune"], cwd=mirror, timeout=fetch_timeout)

    run_dir = runs_dir / f"{safe_repo_name(repo)}-{base_commit[:12]}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
    run_cmd(["git", "clone", "--shared", str(mirror), str(run_dir)], timeout=fetch_timeout)
    run_cmd(["git", "checkout", "--detach", base_commit], cwd=run_dir, timeout=120)
    run_cmd(["git", "submodule", "update", "--init", "--recursive"], cwd=run_dir, timeout=fetch_timeout, check=False)
    return run_dir


def write_runtime_openclaw_config(source_config: Path, work_root: Path, task_id: str) -> Path:
    """Copy OpenClaw config so per-run workspace patches never persist globally."""
    source_config = source_config.expanduser().resolve()
    runtime_dir = work_root / "runtime_configs"
    runtime_config = runtime_dir / (
        f"openclaw-{re.sub(r'[^A-Za-z0-9_.:-]+', '-', task_id or 'task')}-"
        f"{os.getpid()}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.json"
    )
    try:
        config = read_json(source_config)
    except (OSError, json.JSONDecodeError):
        write_text(runtime_config, source_config.read_text(encoding="utf-8"))
        return runtime_config

    if isinstance(config, dict):
        defaults = ((config.get("agents") or {}).get("defaults") if isinstance(config.get("agents"), dict) else None)
        if isinstance(defaults, dict):
            defaults.pop("workspace", None)
            defaults.pop("workspaceDir", None)
    write_json(runtime_config, config)
    return runtime_config


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            if item.get("type") == "text":
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
    return "\n".join(part for part in parts if part)


def split_identifier_words(value: str) -> set[str]:
    words = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", value):
        normalized = token.replace("-", "_")
        words.add(normalized.lower())
        for part in re.split(r"[_-]+", normalized):
            if len(part) >= 3:
                words.add(part.lower())
        camel_parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", token)
        for part in camel_parts:
            if len(part) >= 3:
                words.add(part.lower())
    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "from",
        "when",
        "where",
        "should",
        "would",
        "could",
        "issue",
        "problem",
        "repository",
        "base",
        "commit",
        "image",
        "attached",
    }
    return words - stop_words


def task_path_keywords(value: str) -> set[str]:
    words = split_identifier_words(value)
    plain_tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", value)
    ]
    for left, right in zip(plain_tokens, plain_tokens[1:]):
        if len(left) >= 3 and len(right) >= 3:
            words.add(left + right)
            words.add(f"{left}-{right}")
            words.add(f"{left}_{right}")
    return words


def git_tracked_files(workspace: Path) -> list[str]:
    result = run_cmd(["git", "ls-files"], cwd=workspace, timeout=120, check=False)
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def source_file_score(path: str, keywords: set[str]) -> int:
    lowered = path.lower()
    stem = Path(path).stem.lower()
    name = Path(path).name.lower()
    suffix = Path(path).suffix.lower()
    score = 0
    for keyword in keywords:
        if keyword in lowered:
            score += 4
        if keyword in name:
            score += 8
        if keyword in stem:
            score += 5
        if lowered.endswith(f"/{keyword}{suffix}") or name == f"{keyword}{suffix}":
            score += 12
    if suffix in {
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".mjs",
        ".cjs",
        ".py",
        ".rb",
        ".php",
        ".java",
        ".go",
        ".rs",
        ".css",
        ".scss",
        ".html",
        ".vue",
        ".svelte",
    }:
        score += 1
    if lowered.startswith(("src/", "lib/", "packages/", "components/")):
        score += 2
    if any(part in lowered for part in ("/test/", "/tests/", "__tests__", ".spec.", ".test.")):
        score -= 1
        if any(keyword in name or keyword in stem for keyword in keywords):
            score += 3
    if any(part in lowered for part in ("package-lock.json", "yarn.lock", "pnpm-lock.yaml", "dist/", "vendor/")):
        score -= 3
    return score


def source_counterpart_paths(path: str, file_set: set[str]) -> list[str]:
    candidates = []
    prefix_pairs = (
        ("src/", "lib/"),
        ("src/", "es/"),
        ("lib/", "src/"),
        ("es/", "src/"),
        ("source/", "lib/"),
        ("lib/", "source/"),
    )
    for source_prefix, target_prefix in prefix_pairs:
        if path.startswith(source_prefix):
            candidates.append(target_prefix + path[len(source_prefix) :])

    segment_pairs = (
        ("/src/", "/lib/"),
        ("/src/", "/es/"),
        ("/lib/", "/src/"),
        ("/es/", "/src/"),
        ("/source/", "/lib/"),
        ("/lib/", "/source/"),
    )
    for source_segment, target_segment in segment_pairs:
        if source_segment in path:
            candidates.append(path.replace(source_segment, target_segment, 1))

    output = []
    for candidate in candidates:
        if candidate in file_set and candidate not in output:
            output.append(candidate)
    return output


def related_test_paths(path: str, file_set: set[str], *, limit: int = 12) -> list[str]:
    candidates = []
    prism_match = re.match(r"components/prism-(.+?)(?:\.min)?\.js$", path)
    if prism_match:
        language_id = prism_match.group(1)
        prefix = f"tests/languages/{language_id}/"
        candidates.extend(sorted(candidate for candidate in file_set if candidate.startswith(prefix)))

    output = []
    for candidate in candidates:
        if candidate in file_set and candidate not in output:
            output.append(candidate)
        if len(output) >= limit:
            break
    return output


def expand_with_related_paths(selected: list[str], files: list[str], limit: int) -> list[str]:
    file_set = set(files)
    expanded = []
    for path in selected:
        if path not in expanded:
            expanded.append(path)
        for counterpart in source_counterpart_paths(path, file_set):
            if counterpart not in expanded:
                expanded.append(counterpart)
        for test_path in related_test_paths(path, file_set):
            if test_path not in expanded:
                expanded.append(test_path)
        if len(expanded) >= limit:
            return expanded[:limit]
    return expanded


def repo_file_hints(payload: dict[str, Any], workspace: Path, *, limit: int = 80) -> str:
    files = git_tracked_files(workspace)
    if not files:
        payload["_swebench_repo_path_hint_count"] = 0
        return ""

    keywords = task_path_keywords(task_text(payload))
    scored = []
    for path in files:
        score = source_file_score(path, keywords)
        if score > 0:
            scored.append((score, path))
    scored.sort(key=lambda item: (-item[0], item[1]))

    selected = []
    for _score, path in scored:
        if path not in selected:
            selected.append(path)
        if len(selected) >= limit:
            break
    if not selected:
        selected = files[:limit]
    selected = expand_with_related_paths(selected, files, limit)
    absolute_selected = [str(workspace / path) for path in selected]
    payload["_swebench_repo_path_hint_count"] = len(selected)
    payload["_swebench_repo_path_hint_sample"] = absolute_selected[:20]

    lines = [
        "Repository path hints from `git ls-files` (all entries below are verified to exist, relative to the workspace root):",
        f"Exact workspace root: {workspace}",
        "Prefer these exact relative paths for read/edit operations. If the needed path is not listed, "
        "search or list the repository before reading it. Use absolute paths only when required, and "
        "preserve the workspace root exactly. Related source/build counterparts and tests are included "
        "when found; inspect them only when relevant. After any missing-path error, stop using that path.",
    ]
    lines.extend(f"- {path}" for path in selected)
    if len(files) > len(selected):
        lines.append(f"... {len(files) - len(selected)} additional tracked files omitted.")
    return "\n".join(lines)


def payload_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages = payload.get("messages")
    if isinstance(messages, list) and messages:
        return [message for message in messages if isinstance(message, dict)]

    executor_input = payload.get("executor_input") if isinstance(payload.get("executor_input"), dict) else {}
    messages = executor_input.get("messages") if isinstance(executor_input, dict) else []
    if isinstance(messages, list):
        return [message for message in messages if isinstance(message, dict)]
    return []


def payload_has_images(payload: dict[str, Any]) -> bool:
    for message in payload_messages(payload):
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"image_path", "image_url"}:
                    return True
    return False


def user_content_with_repo_instruction(content: Any, preamble: str, final_instruction: str) -> Any:
    preamble_text = f"{preamble}\n\n"
    final_text = f"\n\n{final_instruction}"
    if isinstance(content, list):
        return [
            {"type": "text", "text": preamble_text},
            *content,
            {"type": "text", "text": final_text},
        ]
    text = text_from_content(content)
    return f"{preamble_text}{text}\n\n{final_instruction}".strip()


def inject_real_repo_instruction(payload: dict[str, Any], workspace: Path) -> dict[str, Any]:
    payload = json.loads(json.dumps(payload, ensure_ascii=False))
    executor_input = payload.setdefault("executor_input", {})
    system = str(executor_input.get("system") or "")
    guard = (
        "Real SWE-bench repository mode is active. The current working directory is the "
        f"checked-out repository at base commit, mounted at {workspace}. Inspect and edit the "
        "actual repository files as needed. Prefer exact relative paths from repository path hints, "
        "directory listings, or local search results. "
        f"The exact workspace root is {workspace}; use it only if a tool requires an absolute path, "
        "and never alter it when forming absolute paths. "
        "Do not use web or network tools; solve from the checked-out repository and prompt text. "
        "Make the smallest correct code change. Do not run destructive or history-restoring git commands such as "
        "`git checkout`, `git reset`, `git restore`, `git clean`, or `git revert`. Do not copy "
        "patches from historical commits, changelogs, release notes, or prior diffs. If a read or edit "
        "fails, inspect the current repository state before retrying. If an exact-text edit fails, reread "
        "the current file range and either make a smaller exact edit or use apply_patch against the current "
        "file contents. When using edit, copy oldText verbatim from the latest read/search output, including "
        "leading spaces, blank lines, and newlines; do not reconstruct code blocks from memory. For generated "
        "bundle files that say DO NOT EDIT, prefer editing the source file first unless the benchmark clearly "
        "requires committed generated outputs. Never end by explaining that a tool failed, and never return "
        "NO_REPLY while the "
        "repository still has no tracked diff. When finished, provide a concise summary; the harness will "
        "capture `git diff` from the workspace as the final model_patch."
    )
    path_hints = repo_file_hints(payload, workspace)
    if path_hints:
        guard = f"{guard}\n\n{path_hints}"
    final_edit_instruction = (
        "Important: do not merely describe the fix or print a patch/diff as your answer. "
        "Actually modify the repository file(s) in the current workspace. The final answer "
        "can be a brief summary only; `git diff` will be captured automatically after you finish. "
        "If a tool call fails, recover by rereading/searching current files and trying a smaller edit "
        "or apply_patch; do not finish with a tool-failure explanation or NO_REPLY."
    )
    executor_input["system"] = f"{system}\n\n{guard}".strip()
    payload["expected_output_format"] = {
        "type": "real_repo_edit_then_final_answer",
        "description": (
            "Edit the checked-out repository workspace directly. Do not return patch text as "
            "the only artifact; the wrapper captures git diff after the agent finishes."
        ),
    }

    messages = executor_input.get("messages") if isinstance(executor_input.get("messages"), list) else []
    if messages:
        first = dict(messages[0])
        first["content"] = user_content_with_repo_instruction(
            first.get("content"),
            "Task follows. Use the real repository workspace, and leave the intended fix as file edits.",
            final_edit_instruction,
        )
        messages[0] = first
    else:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Task follows. Use the real repository workspace, and leave the intended fix "
                    f"as file edits.\n\n{final_edit_instruction}"
                ),
            }
        )
    executor_input["messages"] = messages
    payload["messages"] = []
    return payload


def append_self_repair_instruction(payload: dict[str, Any], repair_summary: str, repair_round: int) -> dict[str, Any]:
    payload = json.loads(json.dumps(payload, ensure_ascii=False))
    executor_input = payload.setdefault("executor_input", {})
    messages = executor_input.get("messages") if isinstance(executor_input.get("messages"), list) else []
    messages.append(
        {
            "role": "user",
            "content": (
                f"Repair turn {repair_round}: the previous attempt did not produce a usable repository edit.\n\n"
                f"Observed failure:\n{repair_summary[:4000]}\n\n"
                "Continue in the same checked-out repository workspace and keep any valid edits already made. "
                "Do not repeat the same failed read/edit/apply_patch call. If a path was missing, search or "
                "list the local repository and use an exact existing path. If oldText did not match, reread "
                "the smallest current file range containing the target before editing again. Copy the next "
                "edit.oldText exactly from that latest read/search output, including indentation and blank "
                "lines; do not hand-format or infer whitespace. If exact edit still looks fragile, use "
                "apply_patch with narrow context against the current file contents instead. Do not claim "
                "success unless a tool call actually edits the "
                "repository and leaves a non-empty git diff. Do not answer with a tool-failure explanation "
                "or NO_REPLY. Do not use web/network tools or destructive git commands. Leave the final fix "
                "as actual file edits; the harness will capture git diff."
            ),
        }
    )
    executor_input["messages"] = messages
    payload["messages"] = []
    return payload


def git_diff(workspace: Path, base_commit: str) -> str:
    result = run_cmd(["git", "diff", "--binary", base_commit], cwd=workspace, timeout=120, check=False)
    return result.stdout.strip()


def git_head(workspace: Path) -> str:
    result = run_cmd(["git", "rev-parse", "HEAD"], cwd=workspace, timeout=30, check=False)
    return result.stdout.strip()


def git_status_short(workspace: Path) -> str:
    result = run_cmd(["git", "status", "--short"], cwd=workspace, timeout=30, check=False)
    return result.stdout.strip()


def cleanup_untracked_openclaw_bootstrap(workspace: Path) -> None:
    for name in OPENCLAW_BOOTSTRAP_FILES:
        path = workspace / name
        if not path.exists():
            continue
        tracked = run_cmd(
            ["git", "ls-files", "--error-unmatch", "--", name],
            cwd=workspace,
            timeout=30,
            check=False,
        )
        if tracked.returncode == 0:
            continue
        try:
            path.unlink()
        except OSError:
            pass


def reset_workspace(workspace: Path, base_commit: str) -> None:
    run_cmd(["git", "reset", "--hard", base_commit], cwd=workspace, timeout=120, check=False)
    run_cmd(["git", "clean", "-fd"], cwd=workspace, timeout=120, check=False)


def retryable_delegate_error(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    retry_markers = (
        "request timed out before a response was generated",
        "llm request timed out",
        "model idle timeout",
        "session file locked",
        "agent couldn't generate a response",
        "workspacevanishederror",
        "workspace appears to have disappeared",
        "api rate limit reached",
        "rate limit",
        "rawerror=429",
        "provider returned error 429",
        "429 provider returned error",
    )
    return any(marker in normalized for marker in retry_markers)


def should_isolate_openclaw_agent() -> bool:
    # Gateway rejects ad-hoc agent ids unless they are registered in the active
    # OpenClaw config, so keep isolation opt-in for real-repo media runs.
    return os.environ.get("SWEBENCH_ISOLATE_OPENCLAW_AGENT", "0") == "1"


def assistant_text(delegate_output: dict[str, Any]) -> str:
    choices = delegate_output.get("choices") if isinstance(delegate_output.get("choices"), list) else []
    message = choices[0].get("message") if choices and isinstance(choices[0], dict) else {}
    return str(message.get("content") or "") if isinstance(message, dict) else ""


def repairable_delegate_error(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    repair_markers = (
        "read failed",
        "edit failed",
        "apply_patch failed",
        "enoent",
        "no such file or directory",
        "path escapes sandbox root",
        "oldtext must match exactly",
        "could not find edits",
        "could not find the exact text",
        "missing-path",
        "web_search failed",
        "web_fetch failed",
        "bot-detection challenge",
        "duckduckgo returned",
        "request timed out before a response was generated",
        "llm request timed out",
        "embedded run timeout",
        "git checkout",
        "git reset",
        "git restore",
        "git clean",
        "git revert",
        "author:",
        "date:",
        "files changed",
    )
    return any(marker in normalized for marker in repair_markers)


def schema_echo_without_edit(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    parsed = None
    try:
        parsed = json.loads(str(text or "").strip())
    except json.JSONDecodeError:
        parsed = None
    return (
        "edit the checked-out repository workspace directly" in normalized
        and "the wrapper captures git diff" in normalized
    ) or (
        isinstance(parsed, dict)
        and parsed.get("type") == "real_repo_edit_then_final_answer"
    )


def textual_tool_call_without_edit(text: str) -> bool:
    lowered = str(text or "").lower()
    return (
        re.search(r"(^|\n)\s*(read|edit|apply_patch)\s*(\n|$)", lowered) is not None
        or "<arg_key>" in lowered
        or "<arg_value>" in lowered
        or "oldtext" in lowered and "newtext" in lowered and "git diff" not in lowered
    )


def repeated_text_fragment(text: str) -> bool:
    text = str(text or "")
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 20]
    paragraphs = [
        " ".join(paragraph.split())
        for paragraph in re.split(r"\n\s*\n", text)
        if len(" ".join(paragraph.split())) >= 60
    ]
    for fragments, threshold in ((lines, 4), (paragraphs, 3)):
        counts: dict[str, int] = {}
        for fragment in fragments:
            counts[fragment] = counts.get(fragment, 0) + 1
            if counts[fragment] >= threshold:
                return True
    return False


def repetitive_noop_answer(text: str) -> bool:
    text = str(text or "")
    lowered = " ".join(text.lower().split())
    if not lowered or not repeated_text_fragment(text):
        return False
    already_correct_fix = (
        "already correct" in lowered
        and ("fix is to change the line to" in lowered or "change the line to:" in lowered)
    )
    if not already_correct_fix:
        return False

    code_blocks = [
        " ".join(match.group(1).split())
        for match in re.finditer(r"```[A-Za-z0-9_-]*\s*(.*?)```", text, flags=re.DOTALL)
        if match.group(1).strip()
    ]
    if not code_blocks:
        return True
    counts: dict[str, int] = {}
    for block in code_blocks:
        counts[block] = counts.get(block, 0) + 1
        if counts[block] >= 2:
            return True
    return False


def false_edit_success_without_diff(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    success_markers = (
        "successfully edited",
        "successfully modified",
        "i have edited",
        "i've edited",
        "the change has been made",
        "replaced `",
        "replaced the",
    )
    edit_targets = (
        "file",
        "src/",
        "lib/",
        "tests/",
        "component",
        ".js",
        ".ts",
        ".py",
        ".java",
        ".go",
        ".rb",
    )
    return any(marker in normalized for marker in success_markers) and any(
        target in normalized for target in edit_targets
    )


def guessed_test_path_failure(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    return (
        ("enoent" in normalized or "no such file or directory" in normalized)
        and "/tests/" in normalized
        and (".test" in normalized or ".spec" in normalized)
    )


def destructive_git_or_history_patch_failure(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    destructive_git = (
        "git checkout",
        "git reset",
        "git restore",
        "git clean",
        "git revert",
    )
    history_patch_markers = (
        "author:",
        "date:",
        "files changed",
        "diff --git",
        "changelog:",
    )
    return any(marker in normalized for marker in destructive_git) or (
        "components/prism-shell-session" in normalized
        and sum(1 for marker in history_patch_markers if marker in normalized) >= 2
    )


def raw_params_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    output = []
    marker = "raw_params="
    start = 0
    while True:
        index = text.find(marker, start)
        if index < 0:
            break
        raw_start = index + len(marker)
        try:
            value, end = decoder.raw_decode(text[raw_start:])
        except json.JSONDecodeError:
            start = raw_start
            continue
        if isinstance(value, dict):
            output.append(value)
        start = raw_start + end
    return output


def intended_new_text_already_present(text: str) -> bool:
    lower_text = text.lower()
    current_marker = "current file contents:"
    if current_marker not in lower_text:
        return False
    current_text = text[lower_text.find(current_marker) + len(current_marker) :]
    normalized_current = " ".join(current_text.split())
    for params in raw_params_objects(text):
        edits = params.get("edits")
        if not isinstance(edits, list):
            continue
        for edit in edits:
            if not isinstance(edit, dict):
                continue
            new_text = edit.get("newText")
            if not isinstance(new_text, str) or not new_text.strip():
                continue
            if new_text in current_text:
                return True
            normalized_new_text = " ".join(new_text.split())
            if normalized_new_text and normalized_new_text in normalized_current:
                return True
    return False


def assistant_excerpt(text: str, limit: int = 1200) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


def repair_reason(
    *,
    delegate_output: dict[str, Any] | None,
    diff_text: str,
    detail: str,
    attempt: int,
) -> str | None:
    parts = []
    if detail:
        parts.append(detail.strip())
        if guessed_test_path_failure(detail):
            parts.append(
                "The previous attempt guessed a nonexistent test/spec path. In the next turn, "
                "first list or search the local parent test directory and use only exact tracked "
                "test files returned by `git ls-files`, directory listing, or local search. Do not "
                "retry the guessed test path or invent another conventional test filename."
            )
        if intended_new_text_already_present(detail):
            parts.append(
                "The failed edit's intended newText already appears in the current file contents. "
                "Treat that exact source change as already applied and do not retry the same edit. "
                "Continue with any remaining relevant files or finish."
            )
        if destructive_git_or_history_patch_failure(detail):
            parts.append(
                "The previous attempt drifted into destructive/history-restoring git commands or copied "
                "historical commit/diff/changelog text. Do not run git checkout/reset/restore/clean/revert "
                "and do not use historical patches. Continue from the current workspace files only, using "
                "small exact edits against current file contents."
            )
    if delegate_output is not None:
        text = assistant_text(delegate_output)
        if schema_echo_without_edit(text):
            parts.append("The assistant echoed the expected-output schema instead of editing repository files.")
        if not diff_text:
            if repetitive_noop_answer(text):
                parts.append(
                    "No git diff was produced because the assistant repeated a no-op answer and proposed "
                    "changing code to identical code. In the next turn, stop repeating that explanation, "
                    "reread the issue and exact current files/tests, and identify the real behavioral delta. "
                    "Do not present `change X to X` as a fix; if the cited line is already correct, inspect "
                    "adjacent logic, call sites, counterpart build/source files, and exact tracked tests until "
                    "there is an actual repository edit.\n\nAssistant text excerpt:\n"
                    + assistant_excerpt(text)
                )
            elif false_edit_success_without_diff(text):
                parts.append(
                    "No git diff was produced even though the assistant claimed a file was edited. "
                    "Treat that claim as false: a repository edit is only successful after a runtime "
                    "tool call succeeds on an existing path and `git diff` becomes non-empty. In the "
                    "next turn, inspect or search the repository tree, use exact tracked paths, and do "
                    "not repeat any failed guessed path or claim success after ENOENT/oldText failures.\n\n"
                    "Assistant text excerpt:\n" + assistant_excerpt(text)
                )
            elif textual_tool_call_without_edit(text):
                parts.append(
                    "No git diff was produced because the assistant wrote a tool call as final text "
                    "instead of emitting a runtime-recognized structured tool call. In the next "
                    "turn it must use the provider/OpenClaw tool-call channel for read/edit/apply_patch, "
                    "not print XML/argument markup or a pseudo tool transcript.\n\nAssistant text excerpt:\n"
                    + assistant_excerpt(text)
                )
            else:
                parts.append(
                    "No git diff was produced in the workspace after the attempt.\n\n"
                    "Assistant text excerpt:\n" + assistant_excerpt(text)
                )
    elif not detail:
        parts.append("The delegate failed before producing normalized output.")
    summary = "\n\n".join(part for part in parts if part)
    if not summary:
        return None
    if diff_text and not repairable_delegate_error(summary) and not schema_echo_without_edit(summary):
        return None
    if not diff_text or repairable_delegate_error(summary) or schema_echo_without_edit(summary):
        return f"Attempt {attempt} failed or stalled.\n\n{summary}"
    return None


def normalize_with_diff(
    delegate_output: dict[str, Any],
    diff_text: str,
    workspace: Path,
    reference: dict[str, Any],
    path_hint_count: int,
    path_hint_sample: list[str],
    self_repair_attempts: int = 0,
    self_repair_reasons: list[str] | None = None,
) -> dict[str, Any]:
    choices = delegate_output.get("choices") if isinstance(delegate_output.get("choices"), list) else []
    message = choices[0].get("message") if choices and isinstance(choices[0], dict) else {}
    assistant_text = str(message.get("content") or "") if isinstance(message, dict) else ""
    if diff_text:
        content = f"```diff\n{diff_text}\n```\n\n{assistant_text}".strip()
    else:
        content = assistant_text

    normalized = {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": choices[0].get("finish_reason") if choices and isinstance(choices[0], dict) else "stop",
            }
        ],
        "usage": delegate_output.get("usage"),
        "openclaw": {
            **(delegate_output.get("openclaw") if isinstance(delegate_output.get("openclaw"), dict) else {}),
            "swebench_real_repo": True,
            "repo": reference.get("repo"),
            "base_commit": reference.get("base_commit"),
            "instance_id": reference.get("instance_id"),
            "head_commit": git_head(workspace),
            "workspace": str(workspace),
            "repo_path_hint_count": path_hint_count,
            "repo_path_hint_sample": path_hint_sample,
            "self_repair_attempts": self_repair_attempts,
            "self_repair_reasons": self_repair_reasons or [],
            "git_status_short": git_status_short(workspace),
            "git_diff_bytes": len(diff_text.encode("utf-8")),
            "git_diff_present": bool(diff_text),
        },
    }
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one SWE-bench task in a real repo workspace through OpenClaw.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--openclaw-bin", default=os.environ.get("OPENCLAW_BIN", "openclaw"))
    parser.add_argument("--agent", default=os.environ.get("OPENCLAW_AGENT", "main"))
    parser.add_argument("--model-ref-template", default=os.environ.get("OPENCLAW_MODEL_REF_TEMPLATE", ""))
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("OPENCLAW_CONFIG", "~/.openclaw/openclaw.json")).expanduser())
    parser.add_argument("--work-root", type=Path, default=Path(os.environ.get("SWEBENCH_WORK_ROOT", "/tmp/vlm-exec-routerbench-swebench")).expanduser())
    parser.add_argument("--fetch-timeout", type=int, default=int(os.environ.get("SWEBENCH_FETCH_TIMEOUT", "900")))
    parser.add_argument(
        "--delegate-retries",
        type=int,
        default=int(os.environ.get("SWEBENCH_DELEGATE_RETRIES", "1")),
        help="Retry the OpenClaw delegate on transient timeout/lock/empty-response failures.",
    )
    parser.add_argument(
        "--delegate-retry-sleep",
        type=float,
        default=float(os.environ.get("SWEBENCH_DELEGATE_RETRY_SLEEP", "5.0")),
    )
    parser.add_argument(
        "--self-repair-rounds",
        type=int,
        default=int(os.environ.get("SWEBENCH_SELF_REPAIR_ROUNDS", "1")),
        help=(
            "Run additional same-session repair turns when the delegate produces no diff, "
            "echoes the schema, or fails with path/edit tool errors."
        ),
    )
    parser.add_argument("--keep-workspace", action=argparse.BooleanOptionalAction, default=os.environ.get("SWEBENCH_KEEP_WORKSPACE", "1") != "0")
    parser.add_argument(
        "--delegate-extra-arg",
        action="append",
        default=[],
        help="Extra argument passed through to run_openclaw_executor.py; repeat for multiple args.",
    )
    parser.add_argument(
        "--multimodal-vision-mode",
        choices=["gateway-edit", "cli"],
        default=os.environ.get("SWEBENCH_MULTIMODAL_VISION_MODE", "gateway-edit"),
        help=(
            "How real-repo SWE-bench tasks with image inputs should be sent to OpenClaw. "
            "gateway-edit makes one Gateway agent call with image attachments and the real "
            "repository workspace; cli preserves the old flattened text path behavior."
        ),
    )
    args = parser.parse_args()

    payload = read_json(args.input)
    reference = verifier_reference(payload)
    repo = str(reference.get("repo") or "")
    base_commit = str(reference.get("base_commit") or "")
    workspace = prepare_repo_workspace(
        repo=repo,
        base_commit=base_commit,
        work_root=args.work_root,
        fetch_timeout=args.fetch_timeout,
    )
    runtime_config = write_runtime_openclaw_config(
        args.config,
        args.work_root,
        str(payload.get("task_id") or "task"),
    )

    delegate_input = args.work_root / "delegate_io" / f"{payload.get('task_id', 'task')}-{uuid.uuid4().hex[:8]}.input.json"
    delegate_output = delegate_input.with_suffix(".output.json")
    delegate_input.parent.mkdir(parents=True, exist_ok=True)
    delegate_session_id = re.sub(
        r"[^A-Za-z0-9_.:-]+",
        "-",
        (
            f"swebench-repair-{payload.get('task_id', 'task')}-{payload.get('model', 'model')}-"
            f"{os.getpid()}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        ),
    )
    delegate_agent = args.agent
    if should_isolate_openclaw_agent():
        delegate_agent = safe_openclaw_id(
            f"{args.agent}-swebench-{payload.get('task_id', 'task')}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )

    cmd = [
        sys.executable,
        str(DELEGATE),
        "--input",
        str(delegate_input),
        "--output",
        str(delegate_output),
        "--openclaw-bin",
        args.openclaw_bin,
        "--agent",
        delegate_agent,
        "--session-id",
        delegate_session_id,
        "--config",
        str(runtime_config),
        "--workspace",
        str(workspace),
    ]
    if args.model_ref_template:
        cmd.extend(["--model-ref-template", args.model_ref_template])
    if payload_has_images(payload):
        cmd.extend(["--vision-mode", args.multimodal_vision_mode])
    cmd.extend(args.delegate_extra_arg)
    delegate_env = os.environ.copy()
    delegate_env["OPENCLAW_RUNTIME_CONFIG_DIR"] = str(args.work_root / "runtime_configs")

    try:
        timeout = int(float((payload.get("generation_config") or {}).get("timeout") or 180))
        delegate_result = None
        diff_text = ""
        path_hint_count = 0
        path_hint_sample: list[str] = []
        self_repair_reasons: list[str] = []
        max_repair_rounds = max(0, args.self_repair_rounds)

        for repair_round in range(max_repair_rounds + 1):
            delegate_payload = inject_real_repo_instruction(payload, workspace)
            if self_repair_reasons:
                delegate_payload = append_self_repair_instruction(
                    delegate_payload,
                    self_repair_reasons[-1],
                    repair_round,
                )
            path_hint_count = int(delegate_payload.pop("_swebench_repo_path_hint_count", 0) or 0)
            path_hint_sample = delegate_payload.pop("_swebench_repo_path_hint_sample", []) or []
            write_json(delegate_input, delegate_payload)

            delegate_result = None
            failure_detail = ""
            for attempt in range(args.delegate_retries + 1):
                if attempt > 0 and repair_round == 0:
                    reset_workspace(workspace, base_commit)
                    delegate_payload = inject_real_repo_instruction(payload, workspace)
                    path_hint_count = int(delegate_payload.pop("_swebench_repo_path_hint_count", 0) or 0)
                    path_hint_sample = delegate_payload.pop("_swebench_repo_path_hint_sample", []) or []
                    write_json(delegate_input, delegate_payload)
                result = run_cmd(cmd, cwd=ROOT, timeout=timeout * 2 + 300, check=False, env=delegate_env)
                if result.returncode == 0:
                    delegate_result = read_json(delegate_output)
                    break
                failure_detail = subprocess_error_detail(
                    result.stdout,
                    result.stderr,
                    f"exit {result.returncode}",
                )
                if attempt >= args.delegate_retries or not retryable_delegate_error(failure_detail):
                    break
                print(
                    f"[run_swebench_openclaw_executor_retry] attempt={attempt + 1} "
                    f"task={payload.get('task_id')} reason={failure_detail[:300]}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(args.delegate_retry_sleep * (attempt + 1))

            if delegate_result is None:
                reason = repair_reason(
                    delegate_output=None,
                    diff_text="",
                    detail=failure_detail,
                    attempt=repair_round + 1,
                )
                if repair_round < max_repair_rounds and reason and repairable_delegate_error(reason):
                    self_repair_reasons.append(reason)
                    print(
                        f"[run_swebench_openclaw_executor_repair] round={repair_round + 1} "
                        f"task={payload.get('task_id')} reason={reason[:300]}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                detail = subprocess_error_detail("", failure_detail or reason or "", "", max_chars=5000)
                raise RuntimeError(f"real-repo OpenClaw delegate failed: {detail}")

            cleanup_untracked_openclaw_bootstrap(workspace)
            diff_text = git_diff(workspace, base_commit)
            reason = repair_reason(
                delegate_output=delegate_result,
                diff_text=diff_text,
                detail="",
                attempt=repair_round + 1,
            )
            if repair_round < max_repair_rounds and reason:
                self_repair_reasons.append(reason)
                print(
                    f"[run_swebench_openclaw_executor_repair] round={repair_round + 1} "
                    f"task={payload.get('task_id')} reason={reason[:300]}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            break

        if delegate_result is None:
            raise RuntimeError("real-repo OpenClaw delegate failed without a result.")
        normalized = normalize_with_diff(
            delegate_result,
            diff_text,
            workspace,
            reference,
            path_hint_count,
            path_hint_sample,
            self_repair_attempts=len(self_repair_reasons),
            self_repair_reasons=self_repair_reasons,
        )
        write_json(args.output, normalized)
    finally:
        if not args.keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[run_swebench_openclaw_executor_error] {exc}", file=sys.stderr)
        raise
