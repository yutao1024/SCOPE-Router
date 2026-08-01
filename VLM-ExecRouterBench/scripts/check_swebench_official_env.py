#!/usr/bin/env python3
"""Preflight checks for running the official SWE-bench harness."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def check_python_module(name: str) -> tuple[bool, str]:
    try:
        __import__(name)
    except Exception as exc:
        return False, repr(exc)
    return True, "ok"


def check_docker_socket() -> tuple[bool, str]:
    try:
        import docker
    except Exception as exc:
        return False, f"cannot import docker SDK: {exc!r}"
    try:
        client = docker.from_env()
        version = client.version()
    except Exception as exc:
        return False, repr(exc)
    return True, str(version.get("Version") or version)


def check_cli(command: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=30, check=False)
    except Exception as exc:
        return False, repr(exc)
    detail = (result.stdout.strip() or result.stderr.strip() or f"exit {result.returncode}")[:1000]
    return result.returncode == 0, detail


def check_modal_auth() -> tuple[bool, str]:
    token_id = os.environ.get("MODAL_TOKEN_ID")
    token_secret = os.environ.get("MODAL_TOKEN_SECRET")
    if token_id and token_secret:
        return True, "MODAL_TOKEN_ID and MODAL_TOKEN_SECRET are set"
    ok, detail = check_cli(["modal", "profile", "current"])
    if ok:
        return True, detail
    return False, "missing Modal token env vars and `modal profile current` failed: " + detail


def count_predictions(path: Path | None) -> tuple[bool, str]:
    if path is None:
        return True, "not checked"
    if not path.exists():
        return False, f"missing: {path}"
    count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                missing = [key for key in ("instance_id", "model_patch", "model_name_or_path") if key not in row]
                if missing:
                    return False, f"line {count + 1} missing {missing}"
                count += 1
    except Exception as exc:
        return False, repr(exc)
    if count == 0:
        return False, "prediction file has 0 rows"
    return True, f"{count} rows"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--modal", action="store_true", help="Check Modal auth instead of local Docker daemon.")
    args = parser.parse_args()

    checks = {
        "swebench_module": check_python_module("swebench"),
        "predictions": count_predictions(args.predictions),
    }
    if args.modal:
        checks.update(
            {
                "modal_module": check_python_module("modal"),
                "modal_auth": check_modal_auth(),
            }
        )
    else:
        checks.update(
            {
                "docker_sdk_module": check_python_module("docker"),
                "docker_sdk_daemon": check_docker_socket(),
                "docker_cli_info": check_cli(["docker", "info"]),
            }
        )
    ok = True
    for name, (passed, detail) in checks.items():
        ok = ok and passed
        status = "ok" if passed else "fail"
        print(f"[{status}] {name}: {detail}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
