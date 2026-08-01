#!/usr/bin/env python3
"""Estimate OpenRouter cost from RouterSFT executor result JSONL.

Prices are loaded from OpenRouter's public /api/v1/models catalog by default.
The catalog reports prices in USD per token, so this script multiplies those
prices by the normalized token fields written by generate_router_sft.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_RESULTS = "outputs/smoke_2_per_source/executor_results.jsonl"
DEFAULT_PRICE_CACHE = "runs/openrouter_model_prices.json"

TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "total_tokens",
)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


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
    parts = [input_tokens, output_tokens, reasoning_tokens]
    if any(part is not None for part in parts):
        summed_total = sum(part or 0 for part in parts)
        if total_tokens is None or total_tokens < summed_total:
            total_tokens = summed_total

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cache_read_tokens": cache_read_tokens,
        "total_tokens": total_tokens,
    }


def row_token_fields(row: dict[str, Any], prefix: str = "") -> dict[str, int]:
    if not prefix:
        trajectory = {}
        for field in TOKEN_FIELDS:
            trajectory[field] = int_token(row.get("trajectory_" + field))
        if any(value is not None for value in trajectory.values()):
            return {field: trajectory[field] or 0 for field in TOKEN_FIELDS}
        usage = row.get("usage")
        if isinstance(usage, dict) and isinstance(usage.get("agent_usage"), dict):
            normalized = normalize_usage_tokens(usage["agent_usage"])
            return {field: normalized.get(field) or 0 for field in TOKEN_FIELDS}

    output = {}
    for field in TOKEN_FIELDS:
        value = int_token(row.get(prefix + field))
        output[field] = value or 0
    if any(output.values()):
        return output
    normalized = normalize_usage_tokens(row.get("usage"))
    return {field: normalized.get(field) or 0 for field in TOKEN_FIELDS}


def decimal_price(value: Any) -> Decimal:
    if value is None or value == "":
        return Decimal("0")
    return Decimal(str(value))


def money(value: Decimal, places: str = "0.000001") -> str:
    return f"${value.quantize(Decimal(places), rounding=ROUND_HALF_UP)}"


def fetch_openrouter_catalog(timeout: int) -> dict[str, Any]:
    headers = {"User-Agent": "vlm-exec-routerbench-cost-estimator/1.0"}
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(OPENROUTER_MODELS_URL, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def load_price_catalog(cache_path: Path, *, refresh: bool, offline: bool, timeout: int) -> dict[str, dict[str, Decimal]]:
    if offline or (cache_path.exists() and not refresh):
        if not cache_path.exists():
            raise RuntimeError(f"Price cache does not exist: {cache_path}")
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        try:
            raw = fetch_openrouter_catalog(timeout)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                "Failed to fetch OpenRouter model prices from "
                f"{OPENROUTER_MODELS_URL}: {exc!r}. Rerun with network access, "
                "or pass --offline --price-cache PATH after creating a cached catalog."
            ) from exc
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    prices = {}
    for item in raw.get("data") or []:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        pricing = item.get("pricing") if isinstance(item.get("pricing"), dict) else {}
        prices[str(item["id"])] = {key: decimal_price(value) for key, value in pricing.items()}
    return prices


def strip_openrouter_prefix(model: str) -> str:
    if model.startswith("openrouter/"):
        return model[len("openrouter/") :]
    return model


def model_aliases(row: dict[str, Any], model: str) -> list[str]:
    aliases = []
    for value in (
        model,
        strip_openrouter_prefix(model),
        row.get("openclaw_model_ref"),
        strip_openrouter_prefix(str(row.get("openclaw_model_ref") or "")),
    ):
        value = str(value or "").strip()
        if value and value not in aliases:
            aliases.append(value)

    if "/" not in model:
        for value in (f"openai/{model}", f"qwen/{model}"):
            if value not in aliases:
                aliases.append(value)
    return aliases


def resolve_price_model(
    *,
    row: dict[str, Any],
    model: str,
    prices: dict[str, dict[str, Decimal]],
    explicit_aliases: dict[str, str],
) -> str | None:
    if model in explicit_aliases:
        return explicit_aliases[model]
    for alias in model_aliases(row, model):
        if alias in explicit_aliases:
            return explicit_aliases[alias]
        if alias in prices:
            return alias
    return None


def calculate_cost(tokens: dict[str, int], pricing: dict[str, Decimal], *, bill_reasoning: str) -> dict[str, Decimal]:
    input_tokens = Decimal(tokens.get("input_tokens") or 0)
    output_tokens = Decimal(tokens.get("output_tokens") or 0)
    reasoning_tokens = Decimal(tokens.get("reasoning_tokens") or 0)
    cache_read_tokens = Decimal(tokens.get("cache_read_tokens") or 0)

    prompt_price = pricing.get("prompt", Decimal("0"))
    completion_price = pricing.get("completion", Decimal("0"))
    cache_read_price = pricing.get("input_cache_read")
    reasoning_price = pricing.get("internal_reasoning", completion_price)

    if cache_read_price is not None and cache_read_tokens:
        billable_prompt_tokens = max(input_tokens - cache_read_tokens, Decimal("0"))
        prompt_cost = billable_prompt_tokens * prompt_price
        cache_read_cost = cache_read_tokens * cache_read_price
    else:
        prompt_cost = input_tokens * prompt_price
        cache_read_cost = Decimal("0")

    completion_cost = output_tokens * completion_price
    if bill_reasoning == "separate":
        reasoning_cost = reasoning_tokens * reasoning_price
    elif bill_reasoning == "auto" and "internal_reasoning" in pricing:
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


def add_numbers(target: dict[str, Any], values: dict[str, int | Decimal]) -> None:
    for key, value in values.items():
        if isinstance(value, Decimal):
            target[key] = target.get(key, Decimal("0")) + value
        else:
            target[key] = int(target.get(key, 0)) + int(value or 0)


def empty_bucket() -> dict[str, Any]:
    return {
        "rows": 0,
        "passed": 0,
        "latency_s": Decimal("0"),
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "total_tokens": 0,
        "prompt_cost": Decimal("0"),
        "completion_cost": Decimal("0"),
        "cache_read_cost": Decimal("0"),
        "reasoning_cost": Decimal("0"),
        "total_cost": Decimal("0"),
    }


def serializable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serializable(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate OpenRouter spend from executor_results.jsonl.")
    parser.add_argument("--results", type=Path, default=Path(DEFAULT_RESULTS))
    parser.add_argument("--price-cache", type=Path, default=Path(DEFAULT_PRICE_CACHE))
    parser.add_argument("--refresh-prices", action="store_true", help="Fetch fresh prices from OpenRouter even if cache exists.")
    parser.add_argument("--offline", action="store_true", help="Use --price-cache only; do not fetch OpenRouter.")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument(
        "--model-alias",
        action="append",
        default=[],
        metavar="FROM=TO",
        help="Map a result model name to an OpenRouter model id. Repeat as needed.",
    )
    parser.add_argument(
        "--bill-reasoning",
        choices=["included", "separate", "auto"],
        default="included",
        help=(
            "How to bill reasoning_tokens. Default assumes reasoning is included in output_tokens. "
            "'auto' adds reasoning only when OpenRouter pricing exposes internal_reasoning."
        ),
    )
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    results_path = args.results if args.results.is_absolute() else ROOT / args.results
    cache_path = args.price_cache if args.price_cache.is_absolute() else ROOT / args.price_cache
    if not results_path.exists():
        raise SystemExit(f"Missing results file: {results_path}")

    explicit_aliases = {}
    for item in args.model_alias:
        if "=" not in item:
            raise SystemExit(f"--model-alias must be FROM=TO, got: {item}")
        left, right = item.split("=", 1)
        explicit_aliases[left.strip()] = right.strip()

    prices = load_price_catalog(cache_path, refresh=args.refresh_prices, offline=args.offline, timeout=args.timeout)
    by_model: dict[str, dict[str, Any]] = defaultdict(empty_bucket)
    by_price_model: dict[str, dict[str, Any]] = defaultdict(empty_bucket)
    judge_by_model: dict[str, dict[str, Any]] = defaultdict(empty_bucket)
    totals = empty_bucket()
    missing_models: dict[str, list[str]] = defaultdict(list)

    for row in read_jsonl(results_path):
        model = str(row.get("candidate_model") or "unknown")
        bucket = by_model[model]
        bucket["rows"] += 1
        bucket["passed"] += int(row.get("passed") is True)
        latency = Decimal(str(row.get("latency_s") or 0))
        bucket["latency_s"] += latency
        totals["rows"] += 1
        totals["passed"] += int(row.get("passed") is True)
        totals["latency_s"] += latency

        tokens = row_token_fields(row)
        add_numbers(bucket, tokens)
        add_numbers(totals, tokens)
        price_model = resolve_price_model(row=row, model=model, prices=prices, explicit_aliases=explicit_aliases)
        if price_model is None:
            missing_models[model].append(str(row.get("task_id") or ""))
        else:
            cost = calculate_cost(tokens, prices[price_model], bill_reasoning=args.bill_reasoning)
            add_numbers(bucket, cost)
            add_numbers(totals, cost)
            price_bucket = by_price_model[price_model]
            price_bucket["rows"] += 1
            price_bucket["passed"] += int(row.get("passed") is True)
            price_bucket["latency_s"] += latency
            add_numbers(price_bucket, tokens)
            add_numbers(price_bucket, cost)

        for judge in row.get("judge_usage") or []:
            if not isinstance(judge, dict):
                continue
            judge_model = str(judge.get("judge_model") or "unknown")
            judge_row = {"usage": judge.get("usage"), "openclaw_model_ref": None}
            judge_tokens = row_token_fields(judge_row)
            judge_bucket = judge_by_model[judge_model]
            judge_bucket["rows"] += 1
            add_numbers(judge_bucket, judge_tokens)

            judge_price_model = resolve_price_model(
                row=judge_row,
                model=judge_model,
                prices=prices,
                explicit_aliases=explicit_aliases,
            )
            if judge_price_model is None:
                missing_models[judge_model].append(f"judge:{row.get('task_id')}")
                continue
            judge_cost = calculate_cost(judge_tokens, prices[judge_price_model], bill_reasoning=args.bill_reasoning)
            add_numbers(judge_bucket, judge_cost)
            add_numbers(totals, {f"judge_{key}": value for key, value in judge_cost.items()})

    candidate_cost = totals["total_cost"]
    judge_cost = totals.get("judge_total_cost", Decimal("0"))
    grand_total = candidate_cost + judge_cost

    print(f"results: {results_path}")
    print(f"price_cache: {cache_path}")
    print(f"rows: {totals['rows']} passed: {totals['passed']}")
    print(f"latency_s_sum: {totals['latency_s']:.3f} latency_h_sum: {totals['latency_s'] / Decimal('3600'):.3f}")
    print(f"candidate_cost: {money(candidate_cost)}")
    print(f"judge_cost: {money(judge_cost)}")
    print(f"grand_total: {money(grand_total)}")

    print("\nBY RESULT MODEL")
    for model, item in sorted(by_model.items(), key=lambda pair: pair[1]["total_cost"], reverse=True):
        print(
            f"{model}\n"
            f"  rows={item['rows']} passed={item['passed']} latency_h={item['latency_s'] / Decimal('3600'):.3f}\n"
            f"  cost={money(item['total_cost'])} prompt={money(item['prompt_cost'])} "
            f"completion={money(item['completion_cost'])} cache_read={money(item['cache_read_cost'])} "
            f"reasoning={money(item['reasoning_cost'])}\n"
            f"  tokens input={item['input_tokens']} output={item['output_tokens']} "
            f"reasoning={item['reasoning_tokens']} cache_read={item['cache_read_tokens']} total={item['total_tokens']}"
        )

    if judge_by_model:
        print("\nJUDGE COST BY MODEL")
        for model, item in sorted(judge_by_model.items(), key=lambda pair: pair[1]["total_cost"], reverse=True):
            print(
                f"{model}\n"
                f"  calls={item['rows']} cost={money(item['total_cost'])} "
                f"tokens input={item['input_tokens']} output={item['output_tokens']} "
                f"reasoning={item['reasoning_tokens']} cache_read={item['cache_read_tokens']} total={item['total_tokens']}"
            )

    if missing_models:
        print("\nUNPRICED MODELS")
        for model, examples in sorted(missing_models.items()):
            sample = ", ".join(example for example in examples[:5] if example)
            print(f"  {model}: {len(examples)} row(s)" + (f" sample={sample}" if sample else ""))
        print("Use --model-alias FROM=TO if a local name maps to a different OpenRouter id.")

    if args.out_json:
        out_path = args.out_json if args.out_json.is_absolute() else ROOT / args.out_json
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "results": str(results_path),
            "price_cache": str(cache_path),
            "bill_reasoning": args.bill_reasoning,
            "totals": {
                **totals,
                "candidate_cost": candidate_cost,
                "judge_cost": judge_cost,
                "grand_total": grand_total,
            },
            "by_model": dict(by_model),
            "by_price_model": dict(by_price_model),
            "judge_by_model": dict(judge_by_model),
            "missing_models": {key: value[:20] for key, value in missing_models.items()},
        }
        out_path.write_text(json.dumps(serializable(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
