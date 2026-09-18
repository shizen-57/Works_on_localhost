#!/usr/bin/env python3
"""Reliability soak test (Participant Guide Sec. 08 "Failure rate" /
"p95 latency"; plan Sec. "Deployment" -- "soak 100 valid requests at
concurrency 1 and 4, counting failures and timeouts, not just successful
latency").

Sends N valid requests (cycling through the public sample cases) against a
live deployment at each of the given concurrency levels and reports
success/failure/timeout counts and latency percentiles per level. A
passing run here is a release gate, not a nice-to-have -- a service that
is only tested from the dev machine at concurrency 1 has not demonstrated
what the plan calls for.

Usage:
    python scripts/soak_api.py --base-url https://your-deployment.example.com \\
        --count 100 --concurrency 1 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "public_sample_cases.json"


async def _one_request(client: httpx.AsyncClient, payload: dict, timeout: float) -> tuple[bool, bool, float]:
    """Returns (success, timed_out, elapsed_seconds)."""
    start = time.monotonic()
    try:
        r = await client.post("/optimize-energy", json=payload, timeout=timeout)
        elapsed = time.monotonic() - start
        return (r.status_code == 200, False, elapsed)
    except httpx.TimeoutException:
        return (False, True, time.monotonic() - start)
    except httpx.HTTPError:
        return (False, False, time.monotonic() - start)


async def run_level(base_url: str, payloads: list[dict], concurrency: int, timeout: float) -> None:
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(client: httpx.AsyncClient, payload: dict):
        async with sem:
            return await _one_request(client, payload, timeout)

    async with httpx.AsyncClient(base_url=base_url) as client:
        results = await asyncio.gather(*[_bounded(client, p) for p in payloads])

    successes = sum(1 for ok, _, _ in results if ok)
    timeouts = sum(1 for _, t, _ in results if t)
    failures = len(results) - successes
    latencies = sorted(e for ok, _, e in results if ok)

    print(f"\n--- concurrency={concurrency} n={len(results)} ---")
    print(f"successes={successes} failures={failures} timeouts={timeouts}")
    if latencies:
        p50 = statistics.median(latencies)
        p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
        print(f"latency (successful only): p50={p50:.2f}s p95={p95:.2f}s max={latencies[-1]:.2f}s")
    if failures:
        print(f"WARNING: {failures} non-2xx/exception responses out of {len(results)}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--timeout", type=float, default=35.0)
    args = parser.parse_args()

    cases = json.loads(DATA_PATH.read_text())["cases"]
    payloads = [cases[i % len(cases)]["input"] for i in range(args.count)]

    for level in args.concurrency:
        await run_level(args.base_url, payloads, level, args.timeout)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
