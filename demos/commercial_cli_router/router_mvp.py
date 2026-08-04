#!/usr/bin/env python3
"""Minimal router for Codex, Claude Code, and opencode CLI frontends."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    raise SystemExit("Provide --prompt or --prompt-file.")


def find_tool(config: dict[str, Any], name: str) -> dict[str, Any]:
    for tool in config.get("tools", []):
        if tool.get("name") == name:
            return tool
    raise SystemExit(f"Unknown tool {name!r}.")


def tool_available(tool: dict[str, Any]) -> bool:
    command = command_prefix(tool) + list(tool.get("command") or [])
    return bool(command and shutil.which(command[0]))


def choose_tool(config: dict[str, Any], prompt: str, forced_tool: str | None) -> dict[str, Any]:
    if forced_tool:
        return find_tool(config, forced_tool)

    lower_prompt = prompt.lower()
    for route in config.get("routes", []):
        keywords = [str(item).lower() for item in route.get("keywords", [])]
        if any(keyword in lower_prompt for keyword in keywords):
            return find_tool(config, route["tool"])

    for tool in config.get("tools", []):
        if tool.get("enabled", True) and tool_available(tool):
            return tool

    for tool in config.get("tools", []):
        if tool.get("enabled", True):
            return tool

    raise SystemExit("No enabled tools are configured.")


def render_command(tool: dict[str, Any], prompt: str, workdir: Path) -> list[str]:
    replacements = {
        "prompt": prompt,
        "workdir": str(workdir),
    }
    command_template = list(tool.get("command", []))
    rendered = [
        str(part).format(**replacements)
        for part in command_prefix(tool) + command_template
    ]
    if tool.get("prompt_mode", "append_arg") == "append_arg" and "{prompt}" not in " ".join(
        map(str, command_template)
    ):
        rendered.append(prompt)
    return rendered


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "tool"


def command_prefix(tool: dict[str, Any]) -> list[str]:
    prefix = [str(part) for part in tool.get("command_prefix", [])]
    for env_name in command_prefix_envs(tool):
        value = os.environ.get(env_name)
        if value:
            prefix.extend(shlex.split(value))
            break
    return prefix


def command_prefix_envs(tool: dict[str, Any]) -> list[str]:
    raw = tool.get("command_prefix_env")
    names: list[str] = []
    if isinstance(raw, str):
        names.append(raw)
    elif isinstance(raw, list):
        names.extend(str(item) for item in raw)
    legacy = tool.get("legacy_command_prefix_env")
    if isinstance(legacy, str):
        names.append(legacy)
    elif isinstance(legacy, list):
        names.extend(str(item) for item in legacy)
    return names


def new_run_dir(output_root: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = f"{int((time.time() % 1) * 1000):03d}"
    return output_root / f"{stamp}-{suffix}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def run_tool(
    tool: dict[str, Any],
    command: list[str],
    prompt: str,
    workdir: Path,
    timeout: int,
) -> dict[str, Any]:
    if not command:
        raise RuntimeError(f"Tool {tool.get('name')} has an empty command.")
    if not shutil.which(command[0]):
        raise RuntimeError(f"Command {command[0]!r} is not available on PATH.")

    prompt_mode = tool.get("prompt_mode", "append_arg")
    input_text = prompt if prompt_mode == "stdin" else None
    completed = subprocess.run(
        command,
        cwd=workdir,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "passed": completed.returncode == 0,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    config_path = Path(args.config)
    config = load_json(config_path)
    prompt = read_prompt(args)
    workdir = Path(args.workdir).resolve()
    tool = choose_tool(config, prompt, args.tool)
    command = render_command(tool, prompt, workdir)

    output_root = Path(args.output_dir)
    run_dir = new_run_dir(output_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "route.json",
        {
            "config": str(config_path),
            "dry_run": args.dry_run,
            "selected_tool": tool["name"],
            "tool_available": tool_available(tool),
            "prompt_mode": tool.get("prompt_mode", "append_arg"),
            "workdir": str(workdir),
            "command": command,
        },
    )
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    if args.dry_run:
        print(f"Dry run selected {tool['name']} and wrote route to {run_dir}")
        return 0

    result = run_tool(tool, command, prompt, workdir, args.timeout)
    write_json(run_dir / "result.json", result)
    (run_dir / f"{sanitize_name(tool['name'])}.stdout.txt").write_text(
        result["stdout"], encoding="utf-8"
    )
    (run_dir / f"{sanitize_name(tool['name'])}.stderr.txt").write_text(
        result["stderr"], encoding="utf-8"
    )
    print(f"{tool['name']} exited with {result['returncode']}")
    print(f"Run directory: {run_dir}")
    return int(result["returncode"])


def parse_args(argv: list[str]) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(here / "tools.example.json"))
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--tool", choices=["codex", "claude-code", "opencode"])
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--output-dir", default=str(here / "runs"))
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
