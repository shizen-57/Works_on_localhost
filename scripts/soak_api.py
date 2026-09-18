#!/usr/bin/env python3
"""Release-gate reliability and latency soak for a deployed GridWise API."""

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
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_public_samples import TOL, _replay

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "public_sample_cases.json"


async def _one_request(
    client: httpx.AsyncClient, payload: dict, timeout: float
) -> tuple[bool, bool, float, str | None]:
    start = time.monotonic()
    try:
        response = await client.post("/optimize-energy", json=payload, timeout=timeout)
        elapsed = time.monotonic() - start
    except httpx.TimeoutException:
        return False, True, time.monotonic() - start, "timeout"
    except httpx.HTTPError as exc:
        return False, False, time.monotonic() - start, type(exc).__name__
    if response.status_code != 200:
        return False, False, elapsed, f"HTTP {response.status_code}"
    try:
        body = response.json()
    except ValueError:
        return False, False, elapsed, "invalid JSON"
    if not isinstance(body, dict) or not isinstance(body.get("directive_interpretation"), list):
        return False, False, elapsed, "invalid response shape"
    try:
        replay_errors = _replay(payload, body["directive_interpretation"], body, TOL)
    except (KeyError, TypeError, ValueError) as exc:
        return False, False, elapsed, f"replay error: {type(exc).__name__}"
    if replay_errors:
        return False, False, elapsed, f"replay failed: {replay_errors[0]}"
    return True, False, elapsed, None


async def run_level(
    base_url: str,
    payloads: list[dict],
    concurrency: int,
    timeout: float,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)
    health_latencies: list[float] = []
    health_failures = 0

    async with httpx.AsyncClient(base_url=base_url) as client:

        async def bounded(payload: dict):
            async with semaphore:
                return await _one_request(client, payload, timeout)

        batch = asyncio.gather(*(bounded(payload) for payload in payloads))
        while not batch.done():
            started = time.monotonic()
            try:
                health = await client.get("/health", timeout=2)
                if health.status_code != 200 or health.json().get("status") != "ok":
                    health_failures += 1
                else:
                    health_latencies.append(time.monotonic() - started)
            except (httpx.HTTPError, ValueError):
                health_failures += 1
            await asyncio.sleep(0.5)
        results = await batch

    successes = sum(1 for ok, _, _, _ in results if ok)
    timeouts = sum(1 for _, timed_out, _, _ in results if timed_out)
    failures = len(results) - successes
    latencies = sorted(elapsed for ok, _, elapsed, _ in results if ok)
    reasons: dict[str, int] = {}
    for ok, _, _, reason in results:
        if not ok and reason:
            reasons[reason] = reasons.get(reason, 0) + 1

    def percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        return values[min(len(values) - 1, math.ceil(fraction * len(values)) - 1)]

    report = {
        "concurrency": concurrency,
        "requests": len(results),
        "successes": successes,
        "failures": failures,
        "timeouts": timeouts,
        "failure_reasons": reasons,
        "latency_s": {
            "p50": statistics.median(latencies) if latencies else None,
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "health": {
            "checks": len(health_latencies) + health_failures,
            "failures": health_failures,
            "max_latency_s": max(health_latencies) if health_latencies else None,
        },
    }
    print(f"\n--- concurrency={concurrency} n={len(results)} ---")
    print(f"successes={successes} failures={failures} timeouts={timeouts}")
    if latencies:
        print(
            f"latency: p50={report['latency_s']['p50']:.2f}s "
            f"p95={report['latency_s']['p95']:.2f}s max={report['latency_s']['max']:.2f}s"
        )
    print(
        f"health checks={report['health']['checks']} failures={health_failures} max={report['health']['max_latency_s']}"
    )
    return report


def _passes(report: dict[str, Any], max_p95: float, max_request: float, max_health: float) -> bool:
    latency = report["latency_s"]
    health = report["health"]
    return bool(
        report["failures"] == 0
        and latency["p95"] is not None
        and latency["p95"] <= max_p95
        and latency["max"] is not None
        and latency["max"] < max_request
        and health["failures"] == 0
        and health["max_latency_s"] is not None
        and health["max_latency_s"] <= max_health
    )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--burst-count", type=int, default=20)
    parser.add_argument("--burst-concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-p95", type=float, default=5.0)
    parser.add_argument("--max-request", type=float, default=30.0)
    parser.add_argument("--max-health", type=float, default=1.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.count < 1 or args.burst_count < 0:
        parser.error("request counts must be positive (burst count may be zero)")
    if any(level < 1 for level in [*args.concurrency, args.burst_concurrency]):
        parser.error("every concurrency value must be at least 1")

    cases = json.loads(DATA_PATH.read_text())["cases"]

    def payloads(count: int) -> list[dict]:
        return [cases[index % len(cases)]["input"] for index in range(count)]

    reports = [await run_level(args.base_url, payloads(args.count), level, args.timeout) for level in args.concurrency]
    if args.burst_count:
        reports.append(
            await run_level(
                args.base_url,
                payloads(args.burst_count),
                args.burst_concurrency,
                args.timeout,
            )
        )
    passed = all(_passes(report, args.max_p95, args.max_request, args.max_health) for report in reports)
    output = {
        "generated_at": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "thresholds": {
            "max_p95_s": args.max_p95,
            "max_request_s": args.max_request,
            "max_health_s": args.max_health,
        },
        "passed": passed,
        "levels": reports,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(output, indent=2) + "\n")
        print(f"Report written to {args.report}")
    print(f"\nRelease soak: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
