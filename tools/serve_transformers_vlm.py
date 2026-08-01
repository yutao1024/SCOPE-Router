#!/usr/bin/env python3
"""Minimal OpenAI-compatible wrapper for Transformers VLM pipelines."""

import argparse
import base64
import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import uvicorn


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, Any]]
    max_tokens: int = 64
    temperature: float = 0.0


def data_url_to_image(url: str):
    from PIL import Image

    _, payload = url.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def normalize_content(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "text", "text": str(content)}]

    normalized = []
    for item in content:
        if not isinstance(item, dict):
            normalized.append({"type": "text", "text": str(item)})
            continue
        item_type = item.get("type")
        if item_type == "text":
            normalized.append({"type": "text", "text": item.get("text", "")})
        elif item_type == "image_url":
            image_url = item.get("image_url") or {}
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if isinstance(url, str) and url.startswith("data:image/"):
                normalized.append({"type": "image", "image": data_url_to_image(url)})
            elif isinstance(url, str) and Path(url).exists():
                normalized.append({"type": "image", "url": Path(url).resolve().as_uri()})
            else:
                normalized.append({"type": "image", "url": url})
        elif item_type == "image":
            normalized.append(item)
    return normalized


def normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "role": message.get("role", "user"),
            "content": normalize_content(message.get("content", "")),
        }
        for message in messages
    ]


def extract_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list) and output:
        return extract_text(output[0])
    if isinstance(output, dict):
        if "generated_text" in output:
            generated = output["generated_text"]
            if isinstance(generated, list) and generated:
                last = generated[-1]
                if isinstance(last, dict):
                    return str(last.get("content", ""))
            return str(generated)
        if "text" in output:
            return str(output["text"])
    return str(output)


def create_app(model_path: str, dtype: str, device_map: str) -> FastAPI:
    from transformers import pipeline

    torch_dtype = "auto"
    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16

    pipe = pipeline(
        "image-text-to-text",
        model=model_path,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )

    app = FastAPI()

    @app.get("/v1/models")
    def models() -> Dict[str, Any]:
        return {"object": "list", "data": [{"id": model_path, "object": "model"}]}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest) -> Dict[str, Any]:
        started = time.time()
        messages = normalize_messages(req.messages)
        outputs = pipe(
            text=messages,
            max_new_tokens=req.max_tokens,
            do_sample=req.temperature > 0,
            temperature=req.temperature if req.temperature > 0 else None,
        )
        text = extract_text(outputs)
        return {
            "id": f"chatcmpl-transformers-{int(started * 1000)}",
            "object": "chat.completion",
            "created": int(started),
            "model": req.model or model_path,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a Transformers VLM through a minimal OpenAI-compatible API")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16"])
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    app = create_app(args.model, dtype=args.dtype, device_map=args.device_map)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
