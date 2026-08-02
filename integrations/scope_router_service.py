#!/usr/bin/env python3
"""HTTP routing service for using SCOPE-Router inside agent gateways.

This is the integration shape used by cc-switch / Claude Code Router style
gateways:

    agent request -> gateway collects candidate models
    gateway POSTs request summary + candidates to this service
    SCOPE-Router returns one candidate
    gateway rewrites body.model before forwarding upstream

The service does not execute Codex, Claude Code, OpenClaw, or any other agent.
Those frameworks keep their normal execution path; SCOPE-Router only chooses
the backend model.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if False:  # pragma: no cover - typing only.
    from routers.features.text_encoder import TextEncoder
    from routers.features.vision_encoder import VisionEncoder


def compact_json(value: Any, limit: int = 12000) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def text_from_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") in {"image", "image_url", "image_path"}:
                    parts.append(f"[{item.get('type')}]")
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return compact_json(value, limit=2000)


def prompt_from_payload(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("prompt"), str) and payload["prompt"].strip():
        return payload["prompt"]

    body = payload.get("body")
    if not isinstance(body, dict):
        return compact_json(payload, limit=12000)

    pieces = []
    if body.get("system") is not None:
        pieces.append("System:\n" + text_from_content(body.get("system")))
    if isinstance(body.get("messages"), list):
        for message in body["messages"]:
            if not isinstance(message, dict):
                continue
            role = message.get("role", "message")
            pieces.append(f"{role}:\n{text_from_content(message.get('content'))}")
    if body.get("input") is not None:
        pieces.append("Input:\n" + text_from_content(body.get("input")))
    if body.get("tools") is not None:
        tool_names = []
        for tool in body.get("tools") or []:
            if isinstance(tool, dict):
                name = tool.get("name") or tool.get("function", {}).get("name")
                if name:
                    tool_names.append(str(name))
        if tool_names:
            pieces.append("Tools:\n" + ", ".join(tool_names[:50]))
    return "\n\n".join(pieces).strip() or compact_json(body, limit=12000)


def normalize_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("candidates") or payload.get("models") or []
    candidates = []
    for item in raw:
        if isinstance(item, str):
            candidates.append({"model": item, "selector": item})
        elif isinstance(item, dict):
            model = item.get("model") or item.get("name") or item.get("id")
            selector = item.get("selector") or item.get("route") or item.get("model")
            if model or selector:
                candidate = dict(item)
                candidate["model"] = str(model or selector)
                candidate["selector"] = str(selector or model)
                candidates.append(candidate)
    return candidates


def candidate_keys(candidate: dict[str, Any]) -> set[str]:
    keys = {
        str(candidate.get("model", "")),
        str(candidate.get("selector", "")),
        str(candidate.get("name", "")),
        str(candidate.get("id", "")),
    }
    provider = str(candidate.get("provider", "") or candidate.get("providerName", ""))
    model = str(candidate.get("model", ""))
    if provider and model:
        keys.add(f"{provider}/{model}")
    aliases = candidate.get("aliases")
    if isinstance(aliases, list):
        keys.update(str(item) for item in aliases)
    return {key for key in keys if key}


def build_candidate_lookup(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup = {}
    for candidate in candidates:
        for key in candidate_keys(candidate):
            lookup[key.lower()] = candidate
    return lookup


class ScopeRoutingEngine:
    def __init__(
        self,
        router_path: Path,
        text_encoder_name: str,
        vision_encoder_name: str,
        batch_size: int,
        device: str | None,
    ):
        from routers.scope_router.router import ScopeRouter

        self.router = ScopeRouter.load(str(router_path))
        self.router.verbose = 0
        self.text_encoder_name = text_encoder_name
        self.vision_encoder_name = vision_encoder_name
        self.batch_size = batch_size
        self.device = device
        self._text_encoder = None
        self._vision_encoder = None
        self._lock = threading.Lock()

    @property
    def model_names(self) -> list[str]:
        return [str(name) for name in self.router.model_names]

    def text_encoder(self) -> TextEncoder:
        if self._text_encoder is None:
            from routers.features.text_encoder import TextEncoder

            self._text_encoder = TextEncoder(
                model_name=self.text_encoder_name,
                device=self.device,
                batch_size=self.batch_size,
                show_progress=False,
            )
        return self._text_encoder

    def vision_encoder(self) -> VisionEncoder:
        if self._vision_encoder is None:
            from routers.features.vision_encoder import VisionEncoder

            self._vision_encoder = VisionEncoder(
                model_name=self.vision_encoder_name,
                device=self.device,
                batch_size=self.batch_size,
            )
        return self._vision_encoder

    def embed(self, prompt: str, image_paths: list[str]) -> tuple[np.ndarray | None, np.ndarray | None]:
        text_emb = None
        vision_emb = None
        expected_text_dim = getattr(self.router, "_last_text_dim", None)
        expected_vision_dim = getattr(self.router, "_last_vision_dim", None)

        if expected_text_dim is None or expected_text_dim > 0:
            text_emb = self.text_encoder().extract([prompt]).astype(np.float32)
        if expected_vision_dim is not None and expected_vision_dim > 0:
            if image_paths:
                image_embs = self.vision_encoder().extract(image_paths).astype(np.float32)
                pooled = image_embs.mean(axis=0, keepdims=True)
                norm = np.linalg.norm(pooled, axis=1, keepdims=True)
                vision_emb = pooled / np.maximum(norm, 1e-8)
            else:
                vision_emb = np.zeros((1, int(expected_vision_dim)), dtype=np.float32)
        return text_emb, vision_emb

    def route(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = prompt_from_payload(payload)
        image_paths = [str(Path(path).expanduser()) for path in payload.get("image_paths", [])]
        candidates = normalize_candidates(payload)
        fallback = payload.get("fallback_model") or payload.get("fallback")

        with self._lock:
            text_emb, vision_emb = self.embed(prompt, image_paths)
            scores = self.router.predict_proba(X_text=text_emb, X_vision=vision_emb)[0]

        ranked_all = sorted(
            (
                {
                    "model_name": self.model_names[index],
                    "score": float(scores[index]),
                }
                for index in range(len(self.model_names))
            ),
            key=lambda row: row["score"],
            reverse=True,
        )

        if candidates:
            lookup = build_candidate_lookup(candidates)
            for row in ranked_all:
                candidate = lookup.get(row["model_name"].lower())
                if candidate is not None:
                    return self._response(row, candidate, ranked_all, routed=True)
            if fallback:
                for candidate in candidates:
                    if str(candidate.get("selector")) == str(fallback) or str(candidate.get("model")) == str(fallback):
                        return self._response(
                            {"model_name": str(fallback), "score": 0.0},
                            candidate,
                            ranked_all,
                            routed=False,
                            reason="fallback-no-scope-candidate-match",
                        )
            first = candidates[0]
            return self._response(
                {"model_name": str(first.get("model") or first.get("selector")), "score": 0.0},
                first,
                ranked_all,
                routed=False,
                reason="fallback-first-candidate-no-scope-candidate-match",
            )

        top = ranked_all[0]
        candidate = {"model": top["model_name"], "selector": top["model_name"]}
        return self._response(top, candidate, ranked_all, routed=True)

    def _response(
        self,
        row: dict[str, Any],
        candidate: dict[str, Any],
        ranked_all: list[dict[str, Any]],
        routed: bool,
        reason: str = "scope-router",
    ) -> dict[str, Any]:
        selector = str(candidate.get("selector") or candidate.get("model") or row["model_name"])
        model = str(candidate.get("model") or selector)
        return {
            "model": selector,
            "selected_model": model,
            "selector": selector,
            "score": float(row["score"]),
            "routed": bool(routed),
            "reason": reason,
            "top": ranked_all[:10],
        }


class MockRoutingEngine:
    """Small checkpoint-free router for testing gateway integration hooks."""

    @property
    def model_names(self) -> list[str]:
        return ["mock-cheap-model", "mock-strong-model"]

    def route(self, payload: dict[str, Any]) -> dict[str, Any]:
        candidates = normalize_candidates(payload)
        if candidates:
            candidate = candidates[-1]
            selector = str(candidate.get("selector") or candidate.get("model"))
            model = str(candidate.get("model") or selector)
        else:
            selector = "mock-strong-model"
            model = selector
        print(
            "[scope-router] mock route "
            f"candidates={len(candidates)} selected={selector}",
            flush=True,
        )
        return {
            "model": selector,
            "selected_model": model,
            "selector": selector,
            "score": 1.0,
            "routed": True,
            "reason": "mock-scope",
            "top": [{"model_name": selector, "score": 1.0}],
        }


def make_handler(engine: ScopeRoutingEngine | MockRoutingEngine):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ScopeRouterService/0.1"

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/health":
                self.write_json({"ok": True, "models": engine.model_names})
                return
            self.send_error(404, "Not found")

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/route":
                self.send_error(404, "Not found")
                return
            try:
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("JSON body must be an object")
                self.write_json(engine.route(payload))
            except Exception as exc:  # noqa: BLE001 - service returns JSON errors.
                self.write_json({"error": str(exc), "routed": False}, status=500)

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("[scope-router] " + fmt % args + "\n")

        def write_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> None:
    args = parse_args()
    if args.mock:
        engine: ScopeRoutingEngine | MockRoutingEngine = MockRoutingEngine()
    else:
        if not args.router:
            raise SystemExit("--router is required unless --mock is set")
        engine = ScopeRoutingEngine(
            router_path=Path(args.router).expanduser(),
            text_encoder_name=args.text_encoder,
            vision_encoder_name=args.vision_encoder,
            batch_size=args.batch_size,
            device=args.device,
        )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(f"SCOPE-Router service listening on http://{args.host}:{args.port}")
    print("Routes: GET /health, POST /route")
    if args.mock:
        print("Mock mode: selecting the last candidate from each request")
    server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router", help="Path to a trained SCOPE-Router .pkl checkpoint")
    parser.add_argument("--mock", action="store_true", help="Run a checkpoint-free mock router for integration tests")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8760)
    parser.add_argument("--text-encoder", default="BAAI/bge-m3")
    parser.add_argument("--vision-encoder", default="facebook/dinov2-large")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
