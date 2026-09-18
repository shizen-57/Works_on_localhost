#!/usr/bin/env python3
"""Benchmark SleepyAI models on frozen GridWise semantic fixtures."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm_interpreter import interpret_notes
from app.providers.sleepyai_catalog import discover_models
from app.providers.sleepyai_provider import SleepyAILLMClient, SleepyAIUsage
from app.schemas import BatteryConfig

FIXTURES_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "semantic_cases.json"
BUNDLES_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "semantic_bundles.json"
TOL = 0.01
HIGH_RISK_IDS = {
    "SEM-019",
    "SEM-020",
    "SEM-025",
    "SEM-030",
    "SEM-034",
    "SEM-037",
    "SEM-042",
    "SEM-046",
    "SEM-048",
    "SEM-050",
    "SEM-051",
    "SEM-054",
    "SEM-056",
    "SEM-060",
    "SEM-064",
    "SEM-065",
    "SEM-066",
    "SEM-067",
    "SEM-068",
    "SEM-070",
    "SEM-071",
    "SEM-072",
    "BUNDLE-006",
    "BUNDLE-012",
}


def _notes_for(case: dict) -> list[str]:
    return case["notes"] if "notes" in case else [case["note"]]


def _expectations_for(case: dict) -> list[dict]:
    expected = case["expected"]
    return expected if isinstance(expected, list) else [expected]


def _fields_match(expected: dict, got: dict) -> dict[str, bool]:
    result = {
        "directive_type": got.get("directive_type") == expected["directive_type"],
        "applies": got.get("applies") == expected["applies"],
    }
    if expected["directive_type"] == "no_op":
        result.update(hours=True, value=True)
        return result
    adjustment = got.get("structured_adjustment") or {}
    result["hours"] = adjustment.get("hours") == expected["hours"]
    value_key = expected["value_key"]
    if value_key is None:
        result["value"] = True
    else:
        value = adjustment.get(value_key)
        result["value"] = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and abs(value - expected["value"]) <= TOL
        )
    return result


async def _run_one(
    client: SleepyAILLMClient,
    case: dict,
    rep: int,
    battery: BatteryConfig,
    semaphore: asyncio.Semaphore,
) -> tuple[dict, int, list[dict] | None, float, Exception | None]:
    async with semaphore:
        start = time.monotonic()
        try:
            directives = await interpret_notes(client, _notes_for(case), battery)
        except Exception as exc:  # noqa: BLE001 - benchmark records each failure
            return case, rep, None, time.monotonic() - start, exc
        return case, rep, directives, time.monotonic() - start, None


def _price(model: dict[str, Any], key: str) -> float | None:
    value = model.get(key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


async def evaluate_model(
    model_id: str,
    model_metadata: dict[str, Any],
    api_key: str,
    base_url: str,
    cases: list[dict],
    battery: BatteryConfig,
    repeats: int,
    high_risk_repeats: int,
    concurrency: int,
    timeout: float,
) -> dict[str, Any]:
    usages: list[SleepyAIUsage] = []
    request_count = 0

    def count_request() -> None:
        nonlocal request_count
        request_count += 1

    client = SleepyAILLMClient(
        api_key=api_key,
        model=model_id,
        base_url=base_url,
        timeout_s=timeout,
        usage_observer=usages.append,
        request_observer=count_request,
    )
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        _run_one(client, case, rep, battery, semaphore)
        for case in cases
        for rep in range(high_risk_repeats if case["id"] in HIGH_RISK_IDS else repeats)
    ]
    total_runs = exact_correct = failures = 0
    field_total = {key: 0 for key in ("directive_type", "applies", "hours", "value")}
    field_correct = dict.fromkeys(field_total, 0)
    type_total: dict[str, int] = {}
    type_correct: dict[str, int] = {}
    latencies: list[float] = []
    mismatches: list[str] = []
    try:
        for task in asyncio.as_completed(tasks):
            case, rep, directives, elapsed, error = await task
            total_runs += 1
            latencies.append(elapsed)
            expectations = _expectations_for(case)
            for expected in expectations:
                dtype = expected["directive_type"]
                type_total[dtype] = type_total.get(dtype, 0) + 1
            if error is not None or directives is None:
                failures += 1
                mismatches.append(f"{case['id']} rep{rep}: {type(error).__name__}")
                continue
            matches_by_note = [
                _fields_match(expected, directive) for expected, directive in zip(expectations, directives, strict=True)
            ]
            for expected, matches in zip(expectations, matches_by_note, strict=True):
                for field, matches_field in matches.items():
                    field_total[field] += 1
                    field_correct[field] += int(matches_field)
                dtype = expected["directive_type"]
                type_correct[dtype] = type_correct.get(dtype, 0) + int(all(matches.values()))
            if all(all(matches.values()) for matches in matches_by_note):
                exact_correct += 1
            else:
                wrong = [
                    f"note{index}:{field}"
                    for index, matches in enumerate(matches_by_note)
                    for field, ok in matches.items()
                    if not ok
                ]
                mismatches.append(f"{case['id']} rep{rep}: wrong={wrong}")
    finally:
        await client.aclose()

    latencies.sort()
    p50 = statistics.median(latencies) if latencies else None
    p95 = latencies[min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)] if latencies else None
    max_latency = max(latencies) if latencies else None
    prompt_tokens = sum(item.prompt_tokens for item in usages)
    completion_tokens = sum(item.completion_tokens for item in usages)
    input_price = _price(model_metadata, "inputPrice")
    output_price = _price(model_metadata, "outputPrice")
    estimated_cost = None
    if input_price is not None and output_price is not None:
        estimated_cost = (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
    eligible = (
        total_runs > 0
        and exact_correct == total_runs
        and failures == 0
        and p95 is not None
        and p95 <= 5
        and max_latency is not None
        and max_latency < 25
    )
    return {
        "model": model_id,
        "eligible": eligible,
        "total_runs": total_runs,
        "exact_correct": exact_correct,
        "exact_accuracy": exact_correct / total_runs if total_runs else 0.0,
        "call_failures": failures,
        "call_failure_rate": failures / total_runs if total_runs else 0.0,
        "request_attempts": request_count,
        "retry_count": max(0, request_count - total_runs),
        "retry_rate": max(0, request_count - total_runs) / total_runs if total_runs else 0.0,
        "per_field_accuracy": {key: field_correct[key] / field_total[key] for key in field_total if field_total[key]},
        "per_type_accuracy": {key: type_correct.get(key, 0) / total for key, total in sorted(type_total.items())},
        "latency_s": {"p50": p50, "p95": p95, "max": max_latency},
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "estimated_cost": estimated_cost,
        },
        "mismatches": mismatches[:25],
    }


def _rank_key(result: dict[str, Any]) -> tuple:
    latency = result["latency_s"]["p95"] or float("inf")
    cost = result["usage"]["estimated_cost"]
    if result["eligible"]:
        # Once every mandatory gate passes, cost is the deciding criterion.
        return (False, float("inf") if cost is None else cost, result["model"])
    return (
        True,
        -result["exact_accuracy"],
        result["call_failures"],
        latency,
        float("inf") if cost is None else cost,
        result["model"],
    )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://www.sleepyai.org/api/v1")
    parser.add_argument("--models", nargs="*", help="model IDs; omit to evaluate every accessible active model")
    parser.add_argument("--fixtures", type=Path, default=FIXTURES_PATH)
    parser.add_argument("--bundles", type=Path, default=BUNDLES_PATH)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--high-risk-repeats",
        type=int,
        default=3,
        help="total runs for the frozen 24-case high-risk subset",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--limit-models", type=int)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.high_risk_repeats < args.repeats or args.concurrency < 1:
        parser.error("repeats/concurrency must be positive and high-risk repeats cannot be lower than repeats")
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        print("ERROR: set LLM_API_KEY in the environment; never pass it on the command line.")
        return 2

    catalog = await discover_models(api_key, args.base_url)
    by_id = {item["id"]: item for item in catalog}
    model_ids = args.models or [
        item["id"]
        for item in catalog
        if item.get("active", True) is not False and item.get("available", True) is not False
    ]
    if args.limit_models:
        model_ids = model_ids[: args.limit_models]
    if not model_ids:
        print("ERROR: no candidate models selected or discovered")
        return 2

    fixture_data = json.loads(args.fixtures.read_text())
    bundle_data = json.loads(args.bundles.read_text()) if args.bundles.exists() else {"cases": []}
    all_cases = fixture_data["cases"] + bundle_data["cases"]
    cases = all_cases[: args.limit] if args.limit else all_cases
    battery = BatteryConfig(**fixture_data["_meta"]["battery_default"])
    results = []
    for model_id in model_ids:
        print(f"\n=== {model_id} ===", flush=True)
        result = await evaluate_model(
            model_id,
            by_id.get(model_id, {}),
            api_key,
            args.base_url,
            cases,
            battery,
            args.repeats,
            args.high_risk_repeats,
            args.concurrency,
            args.timeout,
        )
        results.append(result)
        print(
            f"exact={result['exact_correct']}/{result['total_runs']} "
            f"failures={result['call_failures']} retries={result['retry_count']} "
            f"p95={result['latency_s']['p95']!s} eligible={result['eligible']}"
        )

    ranked = sorted(results, key=_rank_key)
    selected = ranked[0]["model"] if ranked and ranked[0]["eligible"] else None
    print("\n=== Ranking ===")
    for index, result in enumerate(ranked, 1):
        print(f"{index}. {result['model']} accuracy={result['exact_accuracy']:.1%} eligible={result['eligible']}")
    print(f"Selected model: {selected or 'NONE (no candidate passed every gate)'}")

    report_path = args.report
    if report_path is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        report_path = Path("eval_output") / f"model-comparison-{stamp}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "base_url": args.base_url,
                "fixtures": str(args.fixtures),
                "bundles": str(args.bundles),
                "accessible_models": catalog,
                "high_risk_ids": sorted(HIGH_RISK_IDS),
                "base_repeats": args.repeats,
                "high_risk_repeats": args.high_risk_repeats,
                "selected_model": selected,
                "ranking": ranked,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Report written to {report_path}")
    return 0 if selected else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
