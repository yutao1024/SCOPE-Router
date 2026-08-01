#!/usr/bin/env python3
"""Serve PaddleOCR behind a small local HTTP API.

Run in an environment with PaddleOCR installed, for example:

    CUDA_VISIBLE_DEVICES=4 python scripts/ocr_server.py --host 127.0.0.1 --port 8766 --lang ch

Request:

    POST /ocr
    {"image_path": "/absolute/path/to/image.png", "lang": "ch"}

Response:

    {
      "ok": true,
      "provider": "paddleocr",
      "text": "...",
      "blocks": [{"text": "...", "confidence": 0.98, "bbox": [...]}]
    }
"""

from __future__ import annotations

import argparse
import base64
import http.server
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


OCR_CACHE: dict[tuple[str, str], Any] = {}
OCR_CACHE_LOCK = threading.Lock()


def compact_text(text: str, max_chars: int = 500) -> str:
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


def parse_key_value_specs(values: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected key=value for --ocr-arg, got {value!r}")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty --ocr-arg key in {value!r}")
        raw = raw.strip()
        if raw.lower() in {"true", "false"}:
            parsed[key] = raw.lower() == "true"
        elif raw.lower() in {"none", "null"}:
            parsed[key] = None
        else:
            try:
                parsed[key] = int(raw)
            except ValueError:
                try:
                    parsed[key] = float(raw)
                except ValueError:
                    parsed[key] = raw
    return parsed


def build_paddle_ocr(lang: str, device: str, extra_args: dict[str, Any]) -> Any:
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        raise RuntimeError(
            "PaddleOCR is not importable in this Python environment. "
            "Install it in the OCR environment with `python -m pip install paddleocr`."
        ) from exc

    base_kwargs: dict[str, Any] = {"lang": lang}
    if device:
        base_kwargs["device"] = device
    base_kwargs.update(extra_args)

    constructor_attempts = [
        {
            **base_kwargs,
            "enable_hpi": False,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        },
        {
            **{key: value for key, value in base_kwargs.items() if key != "device"},
            "enable_hpi": False,
            "use_angle_cls": True,
            "show_log": False,
        },
        {key: value for key, value in base_kwargs.items() if key != "device"},
    ]
    errors = []
    for kwargs in constructor_attempts:
        try:
            return PaddleOCR(**kwargs)
        except Exception as exc:
            errors.append(f"{exc.__class__.__name__}: {exc}")
    detail = " | ".join(errors)
    if "set_optimization_level" in detail:
        detail += (
            " | This usually means PaddleOCR/PaddleX is newer than the installed paddlepaddle runtime. "
            "Upgrade paddlepaddle-gpu to the matching current wheel, or pin PaddleOCR/PaddleX to match paddlepaddle."
        )
    raise RuntimeError("Could not initialize PaddleOCR. Constructor errors: " + detail)


def get_ocr(lang: str, device: str, extra_args: dict[str, Any]) -> Any:
    cache_key = (lang, json.dumps({"device": device, "extra_args": extra_args}, sort_keys=True))
    with OCR_CACHE_LOCK:
        if cache_key not in OCR_CACHE:
            OCR_CACHE[cache_key] = build_paddle_ocr(lang, device, extra_args)
        return OCR_CACHE[cache_key]


def coerce_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "json"):
        payload = value.json
        if callable(payload):
            payload = payload()
        return coerce_jsonable(payload)
    if isinstance(value, dict):
        return {str(key): coerce_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [coerce_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return coerce_jsonable(value.tolist())
    return str(value)


def result_to_plain(result: Any) -> Any:
    if hasattr(result, "json"):
        payload = result.json
        if callable(payload):
            payload = payload()
        return coerce_jsonable(payload)
    return coerce_jsonable(result)


def bbox_from_item(item: Any) -> Any:
    if isinstance(item, dict):
        for key in ("bbox", "box", "poly", "points", "dt_poly", "rec_poly", "rec_box"):
            if key in item:
                return item[key]
    if isinstance(item, (list, tuple)) and item:
        first = item[0]
        if isinstance(first, (list, tuple)) and first and isinstance(first[0], (list, tuple, int, float)):
            return first
    return None


def parse_blocks_from_res(res: dict[str, Any]) -> list[dict[str, Any]]:
    texts = res.get("rec_texts") or res.get("texts") or []
    scores = res.get("rec_scores") or res.get("scores") or []
    boxes = (
        res.get("rec_polys")
        or res.get("rec_boxes")
        or res.get("dt_polys")
        or res.get("dt_boxes")
        or res.get("boxes")
        or []
    )
    blocks = []
    for index, text in enumerate(texts):
        block = {"text": str(text)}
        if index < len(scores):
            try:
                block["confidence"] = float(scores[index])
            except (TypeError, ValueError):
                block["confidence"] = scores[index]
        if index < len(boxes):
            block["bbox"] = boxes[index]
        blocks.append(block)
    return blocks


def parse_blocks_old_style(result: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if "res" in value and isinstance(value["res"], dict):
                blocks.extend(parse_blocks_from_res(value["res"]))
                return
            if "text" in value:
                block = {"text": str(value.get("text") or "")}
                if "confidence" in value:
                    block["confidence"] = value["confidence"]
                elif "score" in value:
                    block["confidence"] = value["score"]
                bbox = bbox_from_item(value)
                if bbox is not None:
                    block["bbox"] = bbox
                blocks.append(block)
                return
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (list, tuple)):
            if len(value) >= 2 and isinstance(value[1], (list, tuple)) and value[1]:
                text_score = value[1]
                if isinstance(text_score[0], str):
                    block = {"text": text_score[0]}
                    if len(text_score) > 1:
                        block["confidence"] = text_score[1]
                    bbox = bbox_from_item(value)
                    if bbox is not None:
                        block["bbox"] = bbox
                    blocks.append(block)
                    return
            for item in value:
                visit(item)

    visit(result)
    return blocks


def normalize_blocks(raw_result: Any) -> list[dict[str, Any]]:
    plain = result_to_plain(raw_result)
    blocks = parse_blocks_old_style(plain)
    normalized = []
    for block in blocks:
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        normalized.append(
            {
                key: coerce_jsonable(value)
                for key, value in block.items()
                if key == "text" or value is not None
            }
        )
    return normalized


def run_ocr(image_path: Path, lang: str, device: str, extra_args: dict[str, Any]) -> dict[str, Any]:
    ocr = get_ocr(lang, device, extra_args)
    started = time.time()
    if hasattr(ocr, "predict"):
        raw_result = ocr.predict(str(image_path))
    else:
        raw_result = ocr.ocr(str(image_path), cls=True)
    blocks = normalize_blocks(raw_result)
    text = "\n".join(block["text"] for block in blocks).strip()
    return {
        "ok": True,
        "provider": "paddleocr",
        "lang": lang,
        "device": device or None,
        "image_path": str(image_path),
        "text": text,
        "text_preview": compact_text(text),
        "text_chars": len(text),
        "blocks": blocks,
        "block_count": len(blocks),
        "latency_s": round(time.time() - started, 3),
    }


def image_path_from_request(request: dict[str, Any], max_image_bytes: int) -> tuple[Path, Path | None]:
    if request.get("image_path"):
        path = Path(str(request["image_path"])).expanduser()
        return path.resolve(), None

    encoded = str(request.get("image_base64") or "").strip()
    if not encoded:
        raise ValueError("image_path or image_base64 is required")
    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[-1]
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) > max_image_bytes:
        raise ValueError(f"image_base64 exceeds max_image_bytes={max_image_bytes}")
    suffix = str(request.get("suffix") or ".png")
    if not suffix.startswith("."):
        suffix = "." + suffix
    with tempfile.NamedTemporaryFile(prefix="paddleocr-", suffix=suffix, delete=False) as handle:
        handle.write(raw)
        temp_path = Path(handle.name)
    return temp_path, temp_path


def serve(host: str, port: int, lang: str, device: str, extra_args: dict[str, Any], max_image_bytes: int, warmup: bool) -> None:
    if warmup:
        get_ocr(lang, device, extra_args)

    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "PaddleOCRServer/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            if os.environ.get("OCR_SERVER_LOG_REQUESTS", "0") == "1":
                super().log_message(format, *args)

        def write_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path != "/health":
                self.write_json(404, {"ok": False, "error": "not_found"})
                return
            self.write_json(
                200,
                {
                    "ok": True,
                    "provider": "paddleocr",
                    "pid": os.getpid(),
                    "default_lang": lang,
                    "device": device or None,
                    "loaded_models": len(OCR_CACHE),
                },
            )

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path != "/ocr":
                self.write_json(404, {"ok": False, "error": "not_found"})
                return
            temp_path = None
            try:
                length = int(self.headers.get("Content-Length") or "0")
                raw_body = self.rfile.read(length).decode("utf-8")
                request = json.loads(raw_body) if raw_body else {}
                if not isinstance(request, dict):
                    self.write_json(400, {"ok": False, "error": "JSON object required"})
                    return
                image_path, temp_path = image_path_from_request(request, max_image_bytes)
                if not image_path.exists():
                    self.write_json(400, {"ok": False, "error": f"image not found: {image_path}"})
                    return
                request_lang = str(request.get("lang") or lang).strip() or lang
                request_device = str(request.get("device") or device).strip()
                payload = run_ocr(image_path, request_lang, request_device, extra_args)
                self.write_json(200, payload)
            except Exception as exc:
                self.write_json(500, {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"})
            finally:
                if temp_path is not None:
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass

    server = http.server.ThreadingHTTPServer((host, port), Handler)
    print(
        json.dumps(
            {
                "event": "paddleocr_server_started",
                "host": host,
                "port": port,
                "pid": os.getpid(),
                "lang": lang,
                "device": device or None,
                "warmup": warmup,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local PaddleOCR HTTP service.")
    parser.add_argument("--host", default=os.environ.get("OCR_SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("OCR_SERVER_PORT", "8766")))
    parser.add_argument("--lang", default=os.environ.get("PADDLEOCR_LANG", "ch"))
    parser.add_argument("--device", default=os.environ.get("PADDLEOCR_DEVICE", ""))
    parser.add_argument("--ocr-arg", action="append", default=[], help="Extra PaddleOCR constructor key=value. Repeatable.")
    parser.add_argument("--max-image-bytes", type=int, default=int(os.environ.get("OCR_SERVER_MAX_IMAGE_BYTES", str(20 * 1024 * 1024))))
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=os.environ.get("OCR_SERVER_WARMUP", "1") != "0")
    args = parser.parse_args()
    serve(
        host=args.host,
        port=args.port,
        lang=args.lang,
        device=args.device,
        extra_args=parse_key_value_specs(args.ocr_arg),
        max_image_bytes=args.max_image_bytes,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
