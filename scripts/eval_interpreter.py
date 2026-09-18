#!/usr/bin/env python3
"""Real-model semantic evaluation harness (plan Sec. "Real model, cannot be
skipped"). Calls the CONFIGURED LLMClient directly -- not the HTTP API --
so this isolates interpretation accuracy from the rest of the pipeline.

Reads tests/fixtures/semantic_cases.json (a starter set -- see that file's
_meta.purpose; expand toward >=60 reviewed cases before treating a run of
this script as the final release gate) and reports exact-note accuracy,
per-field accuracy, per-directive-type accuracy, latency percentiles, and
failure/retry counts, repeating each case --repeats times to expose
nondeterminism.

This directly needs LLM_PROVIDER/LLM_MODEL/LLM_API_KEY set to a real
provider (not placeholder) -- see .env.example.

Usage:
    LLM_PROVIDER=anthropic LLM_MODEL=claude-opus-5 LLM_API_KEY=sk-... \\
        python scripts/eval_interpreter.py --repeats 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_settings
from app.main import build_llm_client
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


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fixtures", type=Path, default=FIXTURES_PATH)
    args = parser.parse_args()

    settings = load_settings()
    if settings.using_stub_interpreter:
        print("ERROR: LLM_PROVIDER=placeholder cannot be semantically evaluated "
              "-- set a real provider (see .env.example) before running this script.")
        return 2
    client = build_llm_client(settings)

    fixtures = json.loads(args.fixtures.read_text())
    battery = BatteryConfig(**fixtures["_meta"]["battery_default"])
    cases = fixtures["cases"]

    per_field_totals = {"directive_type": 0, "applies": 0, "hours": 0, "value": 0}
    per_field_correct = {k: 0 for k in per_field_totals}
    exact_correct = 0
    total_runs = 0
    failures = 0
    latencies: list[float] = []
    per_type_totals: dict[str, int] = {}
    per_type_correct: dict[str, int] = {}

    for case in cases:
        dtype = case["expected"]["directive_type"]
        per_type_totals[dtype] = per_type_totals.get(dtype, 0) + args.repeats

        for rep in range(args.repeats):
            total_runs += 1
            start = time.monotonic()
            try:
                raw = await client.interpret([case["note"]], battery)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL-CALL {case['id']} rep={rep}: {exc}")
                continue
            latencies.append(time.monotonic() - start)

            if len(raw) != 1:
                failures += 1
                print(f"FAIL-SHAPE {case['id']} rep={rep}: expected 1 entry, got {len(raw)}")
                continue

            matches = _fields_match(case["expected"], raw[0])
            for field, ok in matches.items():
                per_field_totals[field] += 1
                per_field_correct[field] += int(ok)

            all_ok = all(matches.values())
            exact_correct += int(all_ok)
            per_type_correct[dtype] = per_type_correct.get(dtype, 0) + int(all_ok)

            if not all_ok:
                wrong = [f for f, ok in matches.items() if not ok]
                print(f"MISMATCH {case['id']} rep={rep}: wrong fields={wrong} got={raw[0]}")

    print("\n=== Summary ===")
    print(f"Total runs: {total_runs}  Failures (call/shape errors): {failures}")
    print(f"Exact-note accuracy: {exact_correct}/{total_runs} ({100*exact_correct/max(1,total_runs):.1f}%)")
    for field, total in per_field_totals.items():
        correct = per_field_correct[field]
        print(f"  {field}: {correct}/{total} ({100*correct/max(1,total):.1f}%)")
    print("Per-directive-type exact accuracy:")
    for dtype, total in sorted(per_type_totals.items()):
        correct = per_type_correct.get(dtype, 0)
        print(f"  {dtype}: {correct}/{total} ({100*correct/max(1,total):.1f}%)")

    if latencies:
        latencies.sort()
        p50 = statistics.median(latencies)
        p95 = latencies[min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)]
        print(f"Latency: p50={p50:.2f}s p95={p95:.2f}s max={latencies[-1]:.2f}s")

    return 1 if failures or exact_correct < total_runs else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
