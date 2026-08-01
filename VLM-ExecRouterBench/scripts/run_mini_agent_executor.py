#!/usr/bin/env python3
"""Minimal non-OpenClaw executor for RouterSFT tasks.

This is the private/default lightweight path: keep the public OpenClaw adapter
available, but run data generation through a small executor that preserves the
same input/output contract.

For now this "mini agent" intentionally stays conservative but it is not the
``raw_api`` backend:

* normal text/code/tool-selection tasks run through an executor-owned agent
  call path using the OpenAI-compatible provider APIs already configured for the
  pipeline;
* MM tasks run through a small multimodal tool loop with image inspection,
  crop, zoom, and best-effort local OCR tools;
* BrowseComp-Plus fixed-corpus search is supported as an agent tool loop;
* SWE-bench real-repo tasks are handled by the SWE mini-agent adapter
  through ``generate_router_sft.py`` command routing, not here.
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import json
import os
import re
import shutil
import sys
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import generate_router_sft as router_sft  # noqa: E402


MINI_AGENT_MM_DEFAULT_MAX_TOOL_CALLS = 5
MINI_AGENT_MM_DEFAULT_MAX_MODEL_CALLS = 10
MINI_AGENT_MM_TOOL_NAMES = ("inspect_image", "crop_image", "zoom_image", "ocr_image")
MINI_AGENT_OCR_DEFAULT_BACKEND = "paddle_http"
MINI_AGENT_OCR_DEFAULT_ENDPOINT = "http://127.0.0.1:8766/ocr"
MINI_AGENT_OCR_DEFAULT_API_MAX_TOKENS = 1024
MINI_AGENT_MM_DEFAULT_MIN_RETURNED_IMAGE_SIDE = 16
MINI_AGENT_TRAJECTORY_FORMAT = "mini_agent_executor_trajectory_v1"
MINI_AGENT_DEFAULT_TRAJECTORY_DIR = "/tmp/vlm-exec-routerbench-mini-agent-trajectories"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def env_truthy(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


def trajectory_dir() -> Path:
    return Path(os.environ.get("MINI_AGENT_TRAJECTORY_DIR", MINI_AGENT_DEFAULT_TRAJECTORY_DIR)).expanduser()


def safe_slug(value: Any, fallback: str) -> str:
    text = str(value or fallback)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")
    return slug[:120] or fallback


def save_trajectory(payload: dict[str, Any], *, mode: str, model: str, trajectory: dict[str, Any]) -> dict[str, Any]:
    if not env_truthy("MINI_AGENT_SAVE_TRAJECTORY", "1"):
        return {"trajectory_saved": False, "trajectory_format": MINI_AGENT_TRAJECTORY_FORMAT}
    task_id = payload.get("task_id") or payload.get("source_id") or "task"
    path = (
        trajectory_dir()
        / safe_slug(mode, "mode")
        / f"{safe_slug(task_id, 'task')}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.json"
    )
    document = {
        "schema": MINI_AGENT_TRAJECTORY_FORMAT,
        "mode": mode,
        "task_id": payload.get("task_id"),
        "source_id": payload.get("source_id"),
        "category": payload.get("category"),
        "source_dataset": payload.get("source_dataset"),
        "model": model,
        "created_at_unix": time.time(),
        **trajectory,
    }
    write_json(path, document)
    return {
        "trajectory_saved": True,
        "trajectory_format": MINI_AGENT_TRAJECTORY_FORMAT,
        "trajectory_path": str(path),
    }


def attach_trajectory_metadata(response: dict[str, Any], metadata: dict[str, Any]) -> None:
    for key in ("mini_agent", "openclaw"):
        current = response.get(key) if isinstance(response.get(key), dict) else {}
        response[key] = {**current, **metadata}


def task_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": payload.get("task_id"),
        "category": payload.get("category"),
        "difficulty_prior": payload.get("difficulty_prior"),
        "clawbench_style": payload.get("clawbench_style"),
        "source_dataset": payload.get("source_dataset"),
        "source_ref": payload.get("source_ref"),
        "executor_input": payload.get("executor_input") or {
            "messages": payload.get("messages") or [],
            "tools": payload.get("tools") or [],
            "assets": payload.get("assets") or [],
        },
        "router_view": payload.get("router_view") or {},
        "expected_output_format": payload.get("expected_output_format") or {},
        "verifier": payload.get("verifier") or {},
    }


def normalize_response(response: dict[str, Any], *, payload: dict[str, Any], model: str, started: float) -> dict[str, Any]:
    response = dict(response)
    meta = response.get("openclaw") if isinstance(response.get("openclaw"), dict) else {}
    agent_meta = response.get("mini_agent") if isinstance(response.get("mini_agent"), dict) else {}
    normalized_agent_meta = {
        **agent_meta,
        "backend": "mini_agent",
        "model": model,
        "task_id": payload.get("task_id"),
        "latency_s": round(time.time() - started, 3),
    }
    response["mini_agent"] = normalized_agent_meta
    response["openclaw"] = {
        **meta,
        **normalized_agent_meta,
    }
    return response


def parse_tool_arguments(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {"_raw": arguments}
    return arguments if isinstance(arguments, dict) else {}


def mm_max_tool_calls() -> int:
    configured = os.environ.get("MINI_AGENT_MM_MAX_TOOL_CALLS", str(MINI_AGENT_MM_DEFAULT_MAX_TOOL_CALLS))
    return max(1, int(configured))


def mm_max_model_calls() -> int:
    configured = os.environ.get("MINI_AGENT_MM_MAX_MODEL_CALLS", str(MINI_AGENT_MM_DEFAULT_MAX_MODEL_CALLS))
    return max(1, int(configured))


def mm_min_returned_image_side() -> int:
    configured = os.environ.get(
        "MINI_AGENT_MM_MIN_RETURNED_IMAGE_SIDE",
        str(MINI_AGENT_MM_DEFAULT_MIN_RETURNED_IMAGE_SIDE),
    )
    return max(1, int(configured))


def ocr_backends() -> list[str]:
    configured = os.environ.get("MINI_AGENT_OCR_BACKEND", MINI_AGENT_OCR_DEFAULT_BACKEND)
    backends = [item.strip().lower() for item in configured.split(",") if item.strip()]
    return [backend for backend in backends if backend == "paddle_http"] or ["paddle_http"]


def paddle_ocr_endpoint() -> str:
    return os.environ.get("MINI_AGENT_PADDLE_OCR_URL", MINI_AGENT_OCR_DEFAULT_ENDPOINT).strip() or MINI_AGENT_OCR_DEFAULT_ENDPOINT


def paddle_ocr_timeout() -> float:
    return max(1.0, float(os.environ.get("MINI_AGENT_PADDLE_OCR_TIMEOUT", "30")))


def mm_tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "inspect_image",
                "description": "Inspect image dimensions, mode, format, and file metadata.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_index": {
                            "type": "integer",
                            "description": "Zero-based image index from the prompt.",
                            "default": 0,
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "crop_image",
                "description": "Crop a rectangular pixel region from an image and return it as an attached image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_index": {"type": "integer", "default": 0},
                        "x": {"type": "integer", "description": "Left pixel coordinate."},
                        "y": {"type": "integer", "description": "Top pixel coordinate."},
                        "width": {"type": "integer", "description": "Crop width in pixels."},
                        "height": {"type": "integer", "description": "Crop height in pixels."},
                    },
                    "required": ["x", "y", "width", "height"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "zoom_image",
                "description": (
                    "Crop a rectangular pixel region and upscale it before returning it as an attached image. "
                    "Use this for small text, labels, chart values, signs, or fine visual details."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_index": {"type": "integer", "default": 0},
                        "x": {"type": "integer", "description": "Left pixel coordinate."},
                        "y": {"type": "integer", "description": "Top pixel coordinate."},
                        "width": {"type": "integer", "description": "Crop width in pixels."},
                        "height": {"type": "integer", "description": "Crop height in pixels."},
                        "scale": {
                            "type": "number",
                            "description": "Upscale factor from 1.0 to 4.0.",
                            "default": 2.0,
                        },
                        "resample": {
                            "type": "string",
                            "enum": ["nearest", "bilinear", "bicubic", "lanczos"],
                            "default": "lanczos",
                        },
                    },
                    "required": ["x", "y", "width", "height"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ocr_image",
                "description": (
                    "Extract visible text from a full image or optional pixel crop."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_index": {"type": "integer", "default": 0},
                        "x": {"type": "integer", "description": "Optional left pixel coordinate."},
                        "y": {"type": "integer", "description": "Optional top pixel coordinate."},
                        "width": {"type": "integer", "description": "Optional crop width in pixels."},
                        "height": {"type": "integer", "description": "Optional crop height in pixels."},
                        "lang": {
                            "type": "string",
                            "description": "Optional OCR language code.",
                            "default": "ch",
                        },
                    },
                },
            },
        },
    ]


def image_paths_for_task(task: dict[str, Any]) -> list[Path]:
    paths = []
    for item in router_sft.task_image_content_items(task):
        path = router_sft.resolve_image_path(item["image_path"])
        paths.append(path)
    return paths


def image_index(arguments: dict[str, Any], image_paths: list[Path]) -> int:
    try:
        index = int(arguments.get("image_index", 0))
    except (TypeError, ValueError):
        index = 0
    if index < 0 or index >= len(image_paths):
        raise IndexError(f"image_index_out_of_range:{index}; image_count={len(image_paths)}")
    return index


def open_image(path: Path) -> Any:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(f"pillow_unavailable:{exc.__class__.__name__}") from exc
    image = Image.open(path)
    image.load()
    return image


def crop_box(arguments: dict[str, Any], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    x = int(arguments.get("x", 0))
    y = int(arguments.get("y", 0))
    crop_width = int(arguments.get("width", width))
    crop_height = int(arguments.get("height", height))
    left = max(0, min(width, x))
    top = max(0, min(height, y))
    right = max(left + 1, min(width, left + max(1, crop_width)))
    bottom = max(top + 1, min(height, top + max(1, crop_height)))
    return left, top, right, bottom


def image_data_url_from_pil(image: Any, *, image_format: str = "PNG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime_type = "image/png" if image_format.upper() == "PNG" else f"image/{image_format.lower()}"
    return f"data:{mime_type};base64,{encoded}"


def pad_image_to_min_side(image: Any, min_side: int | None = None) -> tuple[Any, bool]:
    min_side = mm_min_returned_image_side() if min_side is None else max(1, int(min_side))
    if image.width >= min_side and image.height >= min_side:
        return image, False
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(f"pillow_unavailable:{exc.__class__.__name__}") from exc
    canvas_width = max(image.width, min_side)
    canvas_height = max(image.height, min_side)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    pasted = image.convert("RGB") if image.mode != "RGB" else image
    canvas.paste(pasted, ((canvas_width - image.width) // 2, (canvas_height - image.height) // 2))
    return canvas, True


def image_resample_filter(name: Any) -> Any:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(f"pillow_unavailable:{exc.__class__.__name__}") from exc
    filters = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    return filters.get(str(name or "lanczos").strip().lower(), Image.Resampling.LANCZOS)


def inspect_image_tool(arguments: dict[str, Any], image_paths: list[Path]) -> dict[str, Any]:
    index = image_index(arguments, image_paths)
    path = image_paths[index]
    image = open_image(path)
    stat = path.stat() if path.exists() else None
    return {
        "image_index": index,
        "path": str(path),
        "exists": path.exists(),
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format,
        "mime_type": mimetypes.guess_type(str(path))[0],
        "bytes": stat.st_size if stat else None,
    }


def crop_image_tool(arguments: dict[str, Any], image_paths: list[Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    index = image_index(arguments, image_paths)
    path = image_paths[index]
    image = open_image(path)
    box = crop_box(arguments, image.size)
    cropped = image.crop(box)
    returned_image, padded = pad_image_to_min_side(cropped)
    data_url = image_data_url_from_pil(returned_image)
    result = {
        "image_index": index,
        "source_width": image.width,
        "source_height": image.height,
        "crop_box": {"left": box[0], "top": box[1], "right": box[2], "bottom": box[3]},
        "width": cropped.width,
        "height": cropped.height,
        "returned_width": returned_image.width,
        "returned_height": returned_image.height,
        "padded_to_min_side": padded,
        "mime_type": "image/png",
    }
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Here is the cropped image returned by crop_image. "
                    "Use it only if it helps answer the original visual question."
                ),
            },
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }
    return result, user_message


def zoom_image_tool(arguments: dict[str, Any], image_paths: list[Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    index = image_index(arguments, image_paths)
    path = image_paths[index]
    image = open_image(path)
    box = crop_box(arguments, image.size)
    cropped = image.crop(box)
    try:
        scale = float(arguments.get("scale", 2.0))
    except (TypeError, ValueError):
        scale = 2.0
    scale = max(1.0, min(4.0, scale))
    zoomed_size = (
        max(1, int(round(cropped.width * scale))),
        max(1, int(round(cropped.height * scale))),
    )
    resample_name = str(arguments.get("resample") or "lanczos").strip().lower()
    zoomed = cropped.resize(zoomed_size, image_resample_filter(resample_name))
    returned_image, padded = pad_image_to_min_side(zoomed)
    data_url = image_data_url_from_pil(returned_image)
    result = {
        "image_index": index,
        "source_width": image.width,
        "source_height": image.height,
        "crop_box": {"left": box[0], "top": box[1], "right": box[2], "bottom": box[3]},
        "crop_width": cropped.width,
        "crop_height": cropped.height,
        "scale": scale,
        "resample": resample_name,
        "width": zoomed.width,
        "height": zoomed.height,
        "returned_width": returned_image.width,
        "returned_height": returned_image.height,
        "padded_to_min_side": padded,
        "mime_type": "image/png",
    }
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Here is the zoomed image returned by zoom_image. "
                    "Use it for small text, labels, chart values, or fine details in the original visual question."
                ),
            },
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }
    return result, user_message


def ocr_available() -> dict[str, Any]:
    if shutil.which("tesseract") is None:
        return {"available": False, "error": "tesseract_binary_unavailable"}
    try:
        import pytesseract  # noqa: F401
    except Exception as exc:
        return {"available": False, "error": f"pytesseract_unavailable:{exc.__class__.__name__}"}
    return {"available": True}


def ocr_image_for_arguments(arguments: dict[str, Any], image_paths: list[Path]) -> tuple[Any, int, dict[str, int] | None]:
    index = image_index(arguments, image_paths)
    path = image_paths[index]
    image = open_image(path)
    region = None
    if all(key in arguments for key in ("x", "y", "width", "height")):
        box = crop_box(arguments, image.size)
        image = image.crop(box)
        region = {"left": box[0], "top": box[1], "right": box[2], "bottom": box[3]}
    return image, index, region


def local_ocr_image(arguments: dict[str, Any], image_paths: list[Path]) -> dict[str, Any]:
    availability = ocr_available()
    image, index, region = ocr_image_for_arguments(arguments, image_paths)
    if not availability.get("available"):
        return {
            **availability,
            "backend": "local",
            "image_index": index,
            "region": region,
            "text": "",
        }
    try:
        import pytesseract

        lang = str(arguments.get("lang") or "eng").strip() or "eng"
        text = pytesseract.image_to_string(image, lang=lang).strip()
        return {
            "available": True,
            "backend": "local",
            "image_index": index,
            "region": region,
            "lang": lang,
            "text": text,
            "text_preview": re.sub(r"\s+", " ", text)[:500],
        }
    except Exception as exc:
        return {
            "available": False,
            "backend": "local",
            "error": f"ocr_failed:{exc.__class__.__name__}",
            "image_index": index,
            "region": region,
            "text": "",
        }


def paddle_http_ocr_image(arguments: dict[str, Any], image_paths: list[Path]) -> dict[str, Any]:
    image, index, region = ocr_image_for_arguments(arguments, image_paths)
    lang = str(arguments.get("lang") or "ch").strip() or "ch"
    endpoint = paddle_ocr_endpoint()
    try:
        data_url = image_data_url_from_pil(image)
        request_payload = {
            "image_base64": data_url,
            "lang": lang,
            "suffix": ".png",
        }
        raw_body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            endpoint,
            data=raw_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=paddle_ocr_timeout()) as response:
            payload = json.loads(response.read().decode("utf-8"))
        text = str(payload.get("text") or "").strip() if isinstance(payload, dict) else ""
        return {
            "available": bool(isinstance(payload, dict) and payload.get("ok")),
            "backend": "paddle_http",
            "endpoint": endpoint,
            "image_index": index,
            "region": region,
            "lang": lang,
            "text": text,
            "text_preview": re.sub(r"\s+", " ", text)[:500],
            "text_chars": len(text),
            "block_count": payload.get("block_count") if isinstance(payload, dict) else None,
            "blocks": payload.get("blocks") if isinstance(payload, dict) else None,
            "latency_s": payload.get("latency_s") if isinstance(payload, dict) else None,
            "error": payload.get("error") if isinstance(payload, dict) and payload.get("error") else None,
        }
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "backend": "paddle_http",
            "endpoint": endpoint,
            "error": f"paddle_http_ocr_failed:{exc.__class__.__name__}",
            "image_index": index,
            "region": region,
            "lang": lang,
            "text": "",
        }
    except Exception as exc:
        return {
            "available": False,
            "backend": "paddle_http",
            "endpoint": endpoint,
            "error": f"paddle_http_ocr_failed:{exc.__class__.__name__}",
            "image_index": index,
            "region": region,
            "lang": lang,
            "text": "",
        }


def ocr_api_model(default_model: str) -> str:
    return os.environ.get("MINI_AGENT_OCR_API_MODEL", "").strip() or default_model


def ocr_api_max_tokens() -> int:
    configured = os.environ.get("MINI_AGENT_OCR_API_MAX_TOKENS", str(MINI_AGENT_OCR_DEFAULT_API_MAX_TOKENS))
    return max(1, int(configured))


def api_ocr_image(
    arguments: dict[str, Any],
    image_paths: list[Path],
    *,
    model: str,
    timeout: int,
    retries: int,
    retry_sleep: float,
    http_transport: str,
    deepseek_thinking: str,
) -> dict[str, Any]:
    image, index, region = ocr_image_for_arguments(arguments, image_paths)
    api_model = ocr_api_model(model)
    prompt = (
        "Extract the visible text from this image exactly. Return only the OCR text. "
        "Preserve line breaks when useful. If no readable text is visible, return an empty string."
    )
    try:
        response = router_sft.post_chat_completion_messages(
            model=api_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an OCR engine. Return only text visible in the image, with no explanation.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url_from_pil(image)}},
                    ],
                },
            ],
            tools=None,
            temperature=0.0,
            max_tokens=ocr_api_max_tokens(),
            timeout=timeout,
            retries=retries,
            retry_sleep=retry_sleep,
            http_transport=http_transport,
            extra_body=router_sft.extra_body_for_model(api_model, deepseek_thinking),
        )
        text = router_sft.message_text(router_sft.response_message(response)).strip()
        return {
            "available": True,
            "backend": "api",
            "model": api_model,
            "image_index": index,
            "region": region,
            "text": text,
            "text_preview": re.sub(r"\s+", " ", text)[:500],
            "usage": response.get("usage"),
        }
    except Exception as exc:
        return {
            "available": False,
            "backend": "api",
            "model": api_model,
            "error": f"ocr_api_failed:{exc.__class__.__name__}",
            "image_index": index,
            "region": region,
            "text": "",
        }


def ocr_image_tool(
    arguments: dict[str, Any],
    image_paths: list[Path],
    *,
    model: str,
    timeout: int,
    retries: int,
    retry_sleep: float,
    http_transport: str,
    deepseek_thinking: str,
) -> dict[str, Any]:
    backends = ocr_backends()
    attempts = []
    for backend in backends:
        if backend == "paddle_http":
            result = paddle_http_ocr_image(arguments, image_paths)
        elif backend == "local":
            result = local_ocr_image(arguments, image_paths)
        else:
            result = api_ocr_image(
                arguments,
                image_paths,
                model=model,
                timeout=timeout,
                retries=retries,
                retry_sleep=retry_sleep,
                http_transport=http_transport,
                deepseek_thinking=deepseek_thinking,
            )
        attempts.append(result)
        if result.get("available") and str(result.get("text") or "").strip():
            return {**result, "configured_backends": backends, "attempts": attempts}
    if attempts:
        return {**attempts[-1], "configured_backends": backends, "attempts": attempts}
    return {"available": False, "error": "ocr_backend_unavailable", "configured_backends": backends, "text": ""}


def execute_mm_tool_call(
    call: dict[str, Any],
    image_paths: list[Path],
    *,
    model: str,
    timeout: int,
    retries: int,
    retry_sleep: float,
    http_transport: str,
    deepseek_thinking: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = str(function.get("name") or "")
    arguments = parse_tool_arguments(call)
    try:
        if name == "inspect_image":
            return inspect_image_tool(arguments, image_paths), None
        if name == "crop_image":
            return crop_image_tool(arguments, image_paths)
        if name == "zoom_image":
            return zoom_image_tool(arguments, image_paths)
        if name == "ocr_image":
            return ocr_image_tool(
                arguments,
                image_paths,
                model=model,
                timeout=timeout,
                retries=retries,
                retry_sleep=retry_sleep,
                http_transport=http_transport,
                deepseek_thinking=deepseek_thinking,
            ), None
        return {"error": f"unknown_tool:{name}", "arguments": arguments}, None
    except Exception as exc:
        return {"error": f"{name}_failed:{exc}", "arguments": arguments}, None


def append_mm_system_instruction(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    instruction = (
        "You are running as a lightweight multimodal agent. Inspect the original image directly first. "
        "Use inspect_image, crop_image, zoom_image, or ocr_image when metadata, an isolated crop, "
        "a zoomed local view, or text extraction would materially improve the answer. "
        "When ready, output only the final requested answer."
    )
    if messages and messages[0].get("role") == "system":
        messages = [dict(messages[0]), *messages[1:]]
        messages[0]["content"] = f"{messages[0].get('content', '')}\n{instruction}".strip()
        return messages
    return [{"role": "system", "content": instruction}, *messages]


def response_usage_total(usage_items: list[Any]) -> dict[str, Any]:
    totals = {field: 0 for field in router_sft.TOKEN_FIELDS}
    saw_any = False
    for usage in usage_items:
        normalized = router_sft.normalize_usage_tokens(usage)
        for field in router_sft.TOKEN_FIELDS:
            value = normalized.get(field)
            if value is not None:
                saw_any = True
                totals[field] += value
    if not saw_any:
        return {}
    return totals


def run_multimodal_tool_loop(
    *,
    payload: dict[str, Any],
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
    executor_input = router_sft.executor_input_for_task(task)
    messages = append_mm_system_instruction(router_sft.build_messages(executor_input))
    tools = mm_tool_schemas()
    image_paths = image_paths_for_task(task)
    max_tool_calls = mm_max_tool_calls()
    max_model_calls = mm_max_model_calls()
    tool_results = []
    requested_calls = 0
    truncated = False
    assistant_turns = 0
    model_calls = 0
    tool_loop_turns = 0
    usage_items = []
    model_responses = []
    response: dict[str, Any] | None = None

    while True:
        tools_for_call = tools if model_calls + 1 < max_model_calls else None
        if tools_for_call is None and tool_results:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "This is the final allowed multimodal model call. Do not call tools. "
                        "Answer now using the original image and any tool results already returned."
                    ),
                }
            )
        response = router_sft.post_chat_completion_messages(
            model=model,
            messages=messages,
            tools=tools_for_call,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries,
            retry_sleep=retry_sleep,
            http_transport=http_transport,
            extra_body=router_sft.extra_body_for_model(model, deepseek_thinking),
        )
        model_calls += 1
        model_responses.append({"model_call_index": model_calls, "response": response})
        usage_items.append(response.get("usage"))
        assistant_turns += 1
        message = router_sft.response_message(response)
        tool_calls = [
            call
            for call in message.get("tool_calls") or []
            if isinstance(call, dict)
            and isinstance(call.get("function"), dict)
            and call["function"].get("name") in MINI_AGENT_MM_TOOL_NAMES
        ]
        if not tool_calls:
            break

        messages.append(message)
        tool_loop_turns += 1
        requested_calls += len(tool_calls)
        for call in tool_calls:
            if len(tool_results) >= max_tool_calls:
                truncated = True
                break
            call_index = len(tool_results) + 1
            result, followup_user_message = execute_mm_tool_call(
                call,
                image_paths,
                model=model,
                timeout=timeout,
                retries=retries,
                retry_sleep=retry_sleep,
                http_transport=http_transport,
                deepseek_thinking=deepseek_thinking,
            )
            result["call_index"] = call_index
            result["max_calls"] = max_tool_calls
            result["tool"] = call.get("function", {}).get("name")
            for attempt in result.get("attempts") or []:
                if isinstance(attempt, dict) and isinstance(attempt.get("usage"), dict):
                    usage_items.append(attempt["usage"])
            tool_results.append(result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(call.get("id") or ""),
                    "name": str(call.get("function", {}).get("name") or ""),
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
            if followup_user_message:
                messages.append(followup_user_message)

        if truncated or len(tool_results) >= max_tool_calls:
            truncated = requested_calls > len(tool_results)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The multimodal tool budget is exhausted. Do not call tools again. "
                        "Answer now using the original image and any tool results already returned."
                    ),
                }
            )
            response = router_sft.post_chat_completion_messages(
                model=model,
                messages=messages,
                tools=None,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                retries=retries,
                retry_sleep=retry_sleep,
                http_transport=http_transport,
                extra_body=router_sft.extra_body_for_model(model, deepseek_thinking),
            )
            model_calls += 1
            model_responses.append({"model_call_index": model_calls, "response": response})
            usage_items.append(response.get("usage"))
            assistant_turns += 1
            break

    if response is None:
        response = router_sft.post_chat_completion_messages(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries,
            retry_sleep=retry_sleep,
            http_transport=http_transport,
            extra_body=router_sft.extra_body_for_model(model, deepseek_thinking),
        )
        model_calls += 1
        model_responses.append({"model_call_index": model_calls, "response": response})
        usage_items.append(response.get("usage"))
    final_message = router_sft.response_message(response)
    if final_message and not final_message.get("tool_calls") and tool_results:
        final_message["tool_calls"] = [
            {
                "id": f"mini_agent_mm_call_{call.get('call_index')}",
                "type": "function",
                "function": {
                    "name": str(call.get("tool") or ""),
                    "arguments": json.dumps(
                        {
                            "image_index": call.get("image_index"),
                            "crop_box": call.get("crop_box"),
                            "region": call.get("region"),
                        },
                        ensure_ascii=False,
                    ),
                },
            }
            for call in tool_results
        ]
    model_call_budget_reached = model_calls >= max_model_calls
    trajectory_metadata = save_trajectory(
        payload,
        mode="multimodal_tool_loop",
        model=model,
        trajectory={
            "messages": [*messages, final_message] if final_message else list(messages),
            "tools": tools,
            "tool_results": tool_results,
            "requested_calls": requested_calls,
            "max_calls": max_tool_calls,
            "model_calls": model_calls,
            "max_model_calls": max_model_calls,
            "model_call_budget_reached": model_call_budget_reached,
            "truncated": truncated or requested_calls > len(tool_results),
            "assistant_turns": assistant_turns,
            "loop_turns": tool_loop_turns,
            "model_responses": model_responses,
            "final_response": response,
        },
    )
    agent_usage = response_usage_total(usage_items)
    if agent_usage:
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        response["usage"] = {
            **usage,
            "agent_usage": agent_usage,
            "last_call_usage": usage,
        }
    response["mini_agent_multimodal"] = {
        "mode": "multimodal_tool_loop",
        "tools": list(MINI_AGENT_MM_TOOL_NAMES),
        "calls": tool_results,
        "requested_calls": requested_calls,
        "max_calls": max_tool_calls,
        "model_calls": model_calls,
        "max_model_calls": max_model_calls,
        "model_call_budget_reached": model_call_budget_reached,
        "truncated": truncated or requested_calls > len(tool_results),
        "assistant_turns": assistant_turns,
        "loop_turns": tool_loop_turns,
        "image_count": len(image_paths),
        "ocr": {
            "configured_backends": ocr_backends(),
            "paddle_http_endpoint": paddle_ocr_endpoint(),
        },
    }
    attach_trajectory_metadata(response, trajectory_metadata)
    return response


def run_payload(
    *,
    payload: dict[str, Any],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    retry_sleep: float,
    http_transport: str,
    deepseek_thinking: str,
) -> dict[str, Any]:
    task = task_from_payload(payload)
    if router_sft.is_browsecomp_plus_task(task):
        response = router_sft.run_browsecomp_plus_tool_loop(
            task=task,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries,
            retry_sleep=retry_sleep,
            http_transport=http_transport,
            deepseek_thinking=deepseek_thinking,
            capture_trajectory=True,
        )
        trajectory = response.pop("browsecomp_plus_trajectory", None)
        trajectory_metadata = save_trajectory(
            payload,
            mode="browsecomp_plus_tool_loop",
            model=model,
            trajectory=trajectory or {"final_response": response},
        )
        meta = response.get("openclaw") if isinstance(response.get("openclaw"), dict) else {}
        agent_meta = response.get("mini_agent") if isinstance(response.get("mini_agent"), dict) else {}
        response["mini_agent"] = {
            **agent_meta,
            **trajectory_metadata,
            "mode": "tool_loop",
            "tool_loop": "browsecomp_plus",
            "search_tool": "search_browsecomp_plus",
        }
        response["openclaw"] = {
            **meta,
            **trajectory_metadata,
            "mini_agent_mode": "tool_loop",
            "tool_loop": "browsecomp_plus",
        }
        return response

    if router_sft.task_image_content_items(task):
        response = run_multimodal_tool_loop(
            payload=payload,
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
        meta = response.get("openclaw") if isinstance(response.get("openclaw"), dict) else {}
        agent_meta = response.get("mini_agent") if isinstance(response.get("mini_agent"), dict) else {}
        response["mini_agent"] = {
            **agent_meta,
            "mode": "multimodal_tool_loop",
            "supports_multimodal": True,
            "tools_present": True,
            "multimodal": response.get("mini_agent_multimodal"),
        }
        response["openclaw"] = {
            **meta,
            "mini_agent_mode": "multimodal_tool_loop",
            "supports_multimodal": True,
            "tools_present": True,
            "multimodal": response.get("mini_agent_multimodal"),
        }
        return response

    executor_input = router_sft.executor_input_for_task(task)
    response = router_sft.post_chat_completion(
        model=model,
        executor_input=executor_input,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
        retry_sleep=retry_sleep,
        http_transport=http_transport,
        extra_body=router_sft.extra_body_for_model(model, deepseek_thinking),
    )
    trajectory_metadata = save_trajectory(
        payload,
        mode="single_turn_agent",
        model=model,
        trajectory={
            "executor_input": executor_input,
            "messages": router_sft.build_messages(executor_input),
            "tools": executor_input.get("tools") or [],
            "final_response": response,
        },
    )
    meta = response.get("openclaw") if isinstance(response.get("openclaw"), dict) else {}
    agent_meta = response.get("mini_agent") if isinstance(response.get("mini_agent"), dict) else {}
    response["mini_agent"] = {
        **agent_meta,
        **trajectory_metadata,
        "mode": "single_turn_agent",
        "supports_multimodal": bool(router_sft.task_image_content_items(task)),
        "tools_present": bool((task.get("executor_input") or {}).get("tools")),
    }
    response["openclaw"] = {
        **meta,
        **trajectory_metadata,
        "mini_agent_mode": "single_turn_agent",
        "supports_multimodal": bool(router_sft.task_image_content_items(task)),
        "tools_present": bool((task.get("executor_input") or {}).get("tools")),
    }
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one RouterSFT task through the lightweight mini agent backend.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--http-transport", choices=["curl", "urllib"], default=os.environ.get("MINI_AGENT_HTTP_TRANSPORT", "curl"))
    parser.add_argument("--deepseek-thinking", choices=["disabled", "enabled"], default=os.environ.get("MINI_AGENT_DEEPSEEK_THINKING", "disabled"))
    args = parser.parse_args()

    payload = read_json(args.input)
    generation = payload.get("generation_config") if isinstance(payload.get("generation_config"), dict) else {}
    model = args.model or str(payload.get("executor_model_ref") or payload.get("openclaw_model_ref") or payload.get("model") or "")
    if not model:
        raise RuntimeError("--model or payload.executor_model_ref is required.")
    started = time.time()
    response = run_payload(
        payload=payload,
        model=model,
        temperature=args.temperature if args.temperature is not None else float(generation.get("temperature") or 0.0),
        max_tokens=args.max_tokens if args.max_tokens is not None else int(generation.get("max_tokens") or 16384),
        timeout=args.timeout if args.timeout is not None else int(generation.get("timeout") or 600),
        retries=args.retries,
        retry_sleep=args.retry_sleep,
        http_transport=args.http_transport,
        deepseek_thinking=args.deepseek_thinking,
    )
    write_json(args.output, normalize_response(response, payload=payload, model=model, started=started))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[run_mini_agent_executor_error] {exc}", file=sys.stderr)
        raise
