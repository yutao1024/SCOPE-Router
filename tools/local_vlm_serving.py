#!/usr/bin/env python3
"""Download and launch local VLM servers for OOD inference."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional

import yaml


DEFAULT_CONFIG = Path("config/local_vlm_serving.yaml")
DEFAULT_MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "local_models/vlm"))


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "models" not in data:
        raise ValueError(f"Invalid serving config: {path}")
    return data


def selected_models(config: Dict[str, Any], names: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    models = config["models"]
    if not names:
        return models
    selected = {}
    missing = []
    for name in names:
        if name in models:
            selected[name] = models[name]
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"Unknown model(s): {', '.join(missing)}")
    return selected


def model_dir(model_root: Path, name: str) -> Path:
    return model_root / name


def merged_model_config(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    defaults = dict(config.get("defaults") or {})
    model = dict(config["models"][name])
    default_extra_args = list(defaults.get("extra_args") or [])
    model_extra_args = list(model.get("extra_args") or [])
    defaults.update(model)
    defaults["extra_args"] = default_extra_args + model_extra_args
    defaults["name"] = name
    return defaults


def download_model(name: str, entry: Dict[str, Any], model_root: Path, revision: Optional[str]) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Missing huggingface_hub. Install it with: pip install -U huggingface_hub") from exc

    repo_id = entry["hf_id"]
    target_dir = model_dir(model_root, name)
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {name}: {repo_id} -> {target_dir}")
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return target_dir


def has_local_model(model_root: Path, name: str) -> bool:
    path = model_dir(model_root, name)
    return path.exists() and any(path.iterdir())


def local_or_remote_model_path(entry: Dict[str, Any], model_root: Path, name: str) -> str:
    path = model_dir(model_root, name)
    if path.exists() and any(path.iterdir()):
        return str(path)
    return str(entry["hf_id"])


def build_vllm_command(
    config: Dict[str, Any],
    name: str,
    model_root: Path,
    host: Optional[str],
    port: Optional[int],
    gpus: Optional[str],
    dtype: Optional[str],
    tp: Optional[int],
    extra_args: List[str],
) -> List[str]:
    entry = merged_model_config(config, name)
    model_path = local_or_remote_model_path(entry, model_root, name)
    tensor_parallel = tp if tp is not None else int(entry.get("tensor_parallel_size", 1))
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--served-model-name",
        name,
        "--host",
        host or str(entry.get("host", "0.0.0.0")),
        "--port",
        str(port if port is not None else int(entry.get("port", 8000))),
        "--dtype",
        dtype or str(entry.get("dtype", "bfloat16")),
        "--tensor-parallel-size",
        str(tensor_parallel),
        "--gpu-memory-utilization",
        str(entry.get("gpu_memory_utilization", 0.90)),
        "--max-model-len",
        str(entry.get("max_model_len", 8192)),
        "--max-num-seqs",
        str(entry.get("max_num_seqs", 4)),
        "--max-num-batched-tokens",
        str(entry.get("max_num_batched_tokens", 8192)),
    ]
    if entry.get("trust_remote_code", True):
        cmd.append("--trust-remote-code")
    cmd.extend(str(item) for item in entry.get("extra_args") or [])
    cmd.extend(extra_args)
    return cmd


def build_sglang_command(
    config: Dict[str, Any],
    name: str,
    model_root: Path,
    host: Optional[str],
    port: Optional[int],
    tp: Optional[int],
    extra_args: List[str],
) -> List[str]:
    entry = merged_model_config(config, name)
    model_path = local_or_remote_model_path(entry, model_root, name)
    tensor_parallel = tp if tp is not None else int(entry.get("tensor_parallel_size", 1))
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--served-model-name",
        name,
        "--host",
        host or str(entry.get("host", "0.0.0.0")),
        "--port",
        str(port if port is not None else int(entry.get("port", 8000))),
        "--tp-size",
        str(tensor_parallel),
    ]
    if entry.get("trust_remote_code", True):
        cmd.append("--trust-remote-code")
    cmd.extend(extra_args)
    return cmd


def build_transformers_command(
    config: Dict[str, Any],
    name: str,
    model_root: Path,
    host: Optional[str],
    port: Optional[int],
    dtype: Optional[str],
    extra_args: List[str],
) -> List[str]:
    entry = merged_model_config(config, name)
    model_path = local_or_remote_model_path(entry, model_root, name)
    cmd = [
        sys.executable,
        "tools/serve_transformers_vlm.py",
        "--model",
        model_path,
        "--host",
        host or str(entry.get("host", "0.0.0.0")),
        "--port",
        str(port if port is not None else int(entry.get("port", 8000))),
        "--dtype",
        dtype or str(entry.get("dtype", "bfloat16")),
    ]
    cmd.extend(extra_args)
    return cmd


def print_shell_command(cmd: List[str], env: Dict[str, str]) -> None:
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    body = " ".join(shlex.quote(item) for item in cmd)
    print(f"{prefix} {body}".strip())


def cmd_list(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    models = selected_models(config, args.models)
    for name, entry in models.items():
        print(
            f"{name}\t{entry['hf_id']}\tbackend={entry.get('backend', 'vllm')}"
            f"\tgpus={entry.get('gpus', '0')}\ttp={entry.get('tensor_parallel_size', 1)}"
        )


def cmd_download(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    models = selected_models(config, args.models)
    args.model_root.mkdir(parents=True, exist_ok=True)
    for name, entry in models.items():
        if has_local_model(args.model_root, name) and not args.force:
            print(f"[download] reuse {name}: {model_dir(args.model_root, name)}")
            continue
        download_model(name, entry, args.model_root, args.revision)


def cmd_serve(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.model not in config["models"]:
        raise ValueError(f"Unknown model: {args.model}")

    entry = merged_model_config(config, args.model)
    backend = args.backend or str(entry.get("backend", "vllm"))
    gpus = args.gpus or str(entry.get("gpus", "0"))
    if backend == "vllm":
        cmd = build_vllm_command(
            config,
            args.model,
            args.model_root,
            args.host,
            args.port,
            gpus,
            args.dtype,
            args.tensor_parallel_size,
            args.extra_arg,
        )
    elif backend == "sglang":
        cmd = build_sglang_command(
            config,
            args.model,
            args.model_root,
            args.host,
            args.port,
            args.tensor_parallel_size,
            args.extra_arg,
        )
    elif backend == "transformers":
        cmd = build_transformers_command(
            config,
            args.model,
            args.model_root,
            args.host,
            args.port,
            args.dtype,
            args.extra_arg,
        )
    else:
        raise ValueError(f"Unsupported backend {backend!r} for {args.model}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["PYTHONNOUSERSITE"] = env.get("PYTHONNOUSERSITE", "1")
    printable_env = {"CUDA_VISIBLE_DEVICES": gpus, "PYTHONNOUSERSITE": env["PYTHONNOUSERSITE"]}
    print_shell_command(cmd, printable_env)
    if args.dry_run:
        return
    os.execvpe(cmd[0], cmd, env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and serve local VLMs for VL-RouterBench")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List configured models")
    list_parser.add_argument("models", nargs="*")
    list_parser.set_defaults(func=cmd_list)

    download_parser = subparsers.add_parser("download", help="Download model snapshots")
    download_parser.add_argument("models", nargs="*")
    download_parser.add_argument("--revision", default=None)
    download_parser.add_argument("--force", action="store_true")
    download_parser.set_defaults(func=cmd_download)

    serve_parser = subparsers.add_parser("serve", help="Launch one model endpoint")
    serve_parser.add_argument("model")
    serve_parser.add_argument("--backend", choices=["vllm", "sglang", "transformers"], default=None)
    serve_parser.add_argument("--host", default=None)
    serve_parser.add_argument("--port", type=int, default=None)
    serve_parser.add_argument("--gpus", default=None, help="CUDA_VISIBLE_DEVICES value, e.g. 0,1")
    serve_parser.add_argument("--tensor-parallel-size", type=int, default=None)
    serve_parser.add_argument("--dtype", default=None)
    serve_parser.add_argument("--extra-arg", action="append", default=[], help="Extra backend argument; repeat as needed")
    serve_parser.add_argument("--dry-run", action="store_true")
    serve_parser.set_defaults(func=cmd_serve)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
