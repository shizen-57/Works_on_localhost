#!/usr/bin/env python3
"""Compare interpretation accuracy across several candidate models on the
same LLMClient implementation (currently: app.providers.sleepyai_provider),
using tests/fixtures/semantic_cases.json. Prints a ranked table and writes
a JSON report.

This is a dev/selection tool, not part of the judged service -- it exists
to answer "which of these free models should LLM_MODEL be set to" before
locking in a submission config.

Usage:
    LLM_API_KEY=... python3 scripts/compare_models.py \\
        --base-url https://www.sleepyai.org/api \\
        --models deepseek-v4.1-flash:free glm-5.3-flash:free grok-4.6:free \\
        --repeats 1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.providers.sleepyai_provider import SleepyAILLMClient
from app.schemas import BatteryConfig

FIXTURES_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "semantic_cases.json"
TOL = 0.01


def _fields_match(expected: dict, got: dict) -> dict[str, bool]:
    result = {"directive_type": got.get("directive_type") == expected["directive_type"]}
    result["applies"] = got.get("applies") == expected["applies"]
    if expected["directive_type"] == "no_op":
        result["hours"] = True
        result["value"] = True
        return result
    adj = got.get("structured_adjustment") or {}
    result["hours"] = adj.get("hours") == expected["hours"]
    if expected["value_key"] is None:
        result["value"] = True
    else:
        v = adj.get(expected["value_key"])
        result["value"] = isinstance(v, (int, float)) and abs(v - expected["value"]) <= TOL
    return result


async def _run_one(client: SleepyAILLMClient, case: dict, rep: int, battery: BatteryConfig, sem: asyncio.Semaphore):
    async with sem:
        start = time.monotonic()
        try:
            raw = await client.interpret([case["note"]], battery)
        except Exception as exc:  # noqa: BLE001
            return case, rep, None, None, exc
        return case, rep, raw, time.monotonic() - start, None


async def evaluate_model(
    client: SleepyAILLMClient, cases: list[dict], battery: BatteryConfig, repeats: int, concurrency: int = 4
) -> dict:
    per_field_totals = {"directive_type": 0, "applies": 0, "hours": 0, "value": 0}
    per_field_correct = {k: 0 for k in per_field_totals}
    exact_correct = 0
    total_runs = 0
    call_failures = 0
    latencies: list[float] = []
    mismatches: list[str] = []

    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _run_one(client, case, rep, battery, sem)
        for case in cases for rep in range(repeats)
    ]
    for coro in asyncio.as_completed(tasks):
        case, rep, raw, elapsed, exc = await coro
        total_runs += 1

        if exc is not None:
            call_failures += 1
            mismatches.append(f"{case['id']} rep{rep}: CALL FAILED: {exc}")
            continue
        latencies.append(elapsed)

        if len(raw) != 1 or not isinstance(raw[0], dict):
            call_failures += 1
            mismatches.append(f"{case['id']} rep{rep}: bad shape, got {raw!r}"[:200])
            continue

        matches = _fields_match(case["expected"], raw[0])
        for field, ok in matches.items():
            per_field_totals[field] += 1
            per_field_correct[field] += int(ok)
        all_ok = all(matches.values())
        exact_correct += int(all_ok)
        if not all_ok:
            wrong = [f for f, ok in matches.items() if not ok]
            mismatches.append(f"{case['id']} rep{rep}: wrong={wrong} got={raw[0]}"[:200])

    latencies.sort()
    return dict(
        total_runs=total_runs,
        call_failures=call_failures,
        exact_correct=exact_correct,
        exact_accuracy=exact_correct / total_runs if total_runs else 0.0,
        per_field=(
            {f: per_field_correct[f] / per_field_totals[f] for f in per_field_totals if per_field_totals[f]}
        ),
        p50_s=statistics.median(latencies) if latencies else None,
        p95_s=(latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else None),
        mismatches=mismatches[:10],  # cap for readability
    )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="only use the first N fixture cases (fast triage)")
    parser.add_argument("--timeout", type=float, default=15.0, help="per-call timeout in seconds")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        print("ERROR: set LLM_API_KEY in the environment (never pass it on the command line).")
        return 2

    fixtures = json.loads(FIXTURES_PATH.read_text())
    battery = BatteryConfig(**fixtures["_meta"]["battery_default"])
    cases = fixtures["cases"][: args.limit] if args.limit else fixtures["cases"]

    results = {}
    for model_id in args.models:
        print(f"\n=== {model_id} ===", flush=True)
        client = SleepyAILLMClient(api_key=api_key, model=model_id, base_url=args.base_url, timeout_s=args.timeout)
        result = await evaluate_model(client, cases, battery, args.repeats)
        results[model_id] = result
        print(f"  exact accuracy: {result['exact_correct']}/{result['total_runs']} "
              f"({100*result['exact_accuracy']:.1f}%)  call_failures={result['call_failures']}")
        for field, acc in result["per_field"].items():
            print(f"    {field}: {100*acc:.1f}%")
        if result["p50_s"] is not None:
            print(f"  latency p50={result['p50_s']:.2f}s p95={result['p95_s']:.2f}s")
        for m in result["mismatches"]:
            print(f"    - {m}")

    print("\n=== Ranking (exact accuracy) ===")
    for model_id, r in sorted(results.items(), key=lambda kv: -kv[1]["exact_accuracy"]):
        print(f"  {r['exact_accuracy']*100:5.1f}%  {model_id}  (failures={r['call_failures']}, "
              f"p50={r['p50_s']:.2f}s)" if r["p50_s"] else f"  {r['exact_accuracy']*100:5.1f}%  {model_id}")

    if args.report:
        args.report.write_text(json.dumps(results, indent=2))
        print(f"\nFull report written to {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
