#!/usr/bin/env python3
"""Evaluate labeled note semantics through a deployed GridWise API.

Unlike eval_interpreter.py, this needs no provider credentials: it sends
safe synthetic energy scenarios to the public endpoint. It checks both the
returned interpretation and the schedule against the fixture's expected
directive (organizer-ground-truth style), so a confidently wrong directive
cannot pass merely because the service replayed its own interpretation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_public_samples import TOL, _replay

FIXTURES_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "semantic_cases.json"
BUNDLES_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "semantic_bundles.json"


def _notes_for(case: dict) -> list[str]:
    return case["notes"] if "notes" in case else [case["note"]]


def _expectations_for(case: dict) -> list[dict]:
    expected = case["expected"]
    return expected if isinstance(expected, list) else [expected]


def _request_for(case: dict, battery: dict) -> dict:
    # Moderate demand and generous storage make every labeled hard directive
    # feasible. In particular, post-window demand must be high enough to let a
    # 75%-reserve case return to its lower end-of-day initial state without
    # assuming battery export or energy dumping.
    return {
        "scenario_id": f"LIVE-{case['id']}",
        "operator_notes": _notes_for(case),
        "hours": [
            {
                "hour": hour,
                "demand_kwh": 30.0,
                "solar_kwh": 20.0 if 6 <= hour < 18 else 0.0,
                "tariff_bdt_per_kwh": float(5 + hour % 6),
            }
            for hour in range(24)
        ],
        "battery": battery,
    }


def _directive_from_expected(expected: dict, note_index: int) -> dict:
    if expected["directive_type"] == "no_op":
        adjustment = None
    else:
        adjustment = {"hours": expected["hours"]}
        if expected["value_key"] is not None:
            adjustment[expected["value_key"]] = expected["value"]
    return {
        "note_index": note_index,
        "applies": expected["applies"],
        "directive_type": expected["directive_type"],
        "structured_adjustment": adjustment,
        "explanation": "fixture ground truth",
    }


def _expected_directives(case: dict) -> list[dict]:
    return [
        _directive_from_expected(expected, note_index) for note_index, expected in enumerate(_expectations_for(case))
    ]


def _expected_directive(case: dict) -> dict:
    """Backward-compatible single-case helper used by unit tests."""
    return _expected_directives(case)[0]


def _interpretation_errors(case: dict, response: dict) -> list[str]:
    got_entries = response.get("directive_interpretation")
    expected_entries = _expected_directives(case)
    if not isinstance(got_entries, list) or len(got_entries) != len(expected_entries):
        return [f"expected {len(expected_entries)} directive entries, got {got_entries!r}"]
    errors = []
    expectations = _expectations_for(case)
    for index, (got, expected, fixture_expected) in enumerate(
        zip(got_entries, expected_entries, expectations, strict=True)
    ):
        if not isinstance(got, dict):
            errors.append(f"note {index}: directive entry must be an object")
            continue
        for key in ("note_index", "applies", "directive_type"):
            if got.get(key) != expected[key]:
                errors.append(f"note {index}: {key}={got.get(key)!r}, expected {expected[key]!r}")
        if expected["structured_adjustment"] is None:
            if got.get("structured_adjustment") is not None:
                errors.append(f"note {index}: structured_adjustment must be null")
            continue
        got_adj = got.get("structured_adjustment") or {}
        exp_adj = expected["structured_adjustment"]
        if got_adj.get("hours") != exp_adj["hours"]:
            errors.append(f"note {index}: hours={got_adj.get('hours')!r}, expected {exp_adj['hours']!r}")
        value_key = fixture_expected["value_key"]
        if value_key is not None:
            value = got_adj.get(value_key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or abs(value - exp_adj[value_key]) > TOL
            ):
                errors.append(f"note {index}: {value_key}={value!r}, expected {exp_adj[value_key]!r}")
    return errors


async def _run_case(
    client: httpx.AsyncClient,
    case: dict,
    battery: dict,
    timeout: float,
    semaphore: asyncio.Semaphore,
) -> tuple[str, float, list[str]]:
    request = _request_for(case, battery)
    async with semaphore:
        start = time.monotonic()
        try:
            result = await client.post("/optimize-energy", json=request, timeout=timeout)
        except httpx.HTTPError as exc:
            return case["id"], time.monotonic() - start, [f"request failed: {exc}"]
    elapsed = time.monotonic() - start
    if result.status_code != 200:
        return case["id"], elapsed, [f"HTTP {result.status_code}: {result.text[:300]}"]
    try:
        response = result.json()
    except ValueError as exc:
        return case["id"], elapsed, [f"invalid JSON response: {exc}"]

    errors = _interpretation_errors(case, response)
    try:
        errors.extend(_replay(request, _expected_directives(case), response, TOL))
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"malformed response during ground-truth replay: {exc}")
    return case["id"], elapsed, errors


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--fixtures", type=Path, default=FIXTURES_PATH)
    parser.add_argument("--bundles", type=Path, default=BUNDLES_PATH)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--case", action="append", dest="case_ids", help="run only this case ID (repeatable)")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    fixture_data = json.loads(args.fixtures.read_text())
    bundle_data = json.loads(args.bundles.read_text()) if args.bundles.exists() else {"cases": []}
    cases = fixture_data["cases"] + bundle_data["cases"]
    if args.case_ids:
        requested = set(args.case_ids)
        cases = [case for case in cases if case["id"] in requested]
        missing = requested - {case["id"] for case in cases}
        if missing:
            parser.error(f"unknown case IDs: {sorted(missing)}")
    if not cases:
        parser.error("fixture file must contain at least one case")
    battery = fixture_data["_meta"]["battery_default"]
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(base_url=args.base_url) as client:
        tasks = [_run_case(client, case, battery, args.timeout, semaphore) for case in cases]
        results = await asyncio.gather(*tasks)

    failures = 0
    latencies = []
    case_results = []
    for case_id, elapsed, errors in results:
        latencies.append(elapsed)
        case_results.append({"id": case_id, "latency_s": elapsed, "errors": errors})
        if errors:
            failures += 1
            print(f"FAIL {case_id} ({elapsed:.2f}s)")
            for error in errors:
                print(f"    - {error}")
        else:
            print(f"PASS {case_id} ({elapsed:.2f}s)")

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)]
    print(f"\n{len(cases) - failures}/{len(cases)} semantic cases passed; p50={p50:.2f}s p95={p95:.2f}s")
    if args.report:
        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "base_url": args.base_url,
            "fixtures_version": fixture_data.get("_meta", {}).get("version"),
            "bundle_version": bundle_data.get("_meta", {}).get("version"),
            "passed": failures == 0,
            "summary": {
                "cases": len(cases),
                "successes": len(cases) - failures,
                "failures": failures,
                "latency_s": {"p50": p50, "p95": p95, "max": max(latencies)},
            },
            "results": case_results,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Report written to {args.report}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
