from __future__ import annotations

import httpx
import pytest

from scripts.soak_api import _one_request, _passes


@pytest.mark.asyncio
async def test_soak_request_requires_replay_valid_response(public_cases):
    case = public_cases[0]
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=case["expected_output"]))
    async with httpx.AsyncClient(base_url="https://example.test", transport=transport) as client:
        ok, timed_out, _, reason = await _one_request(client, case["input"], 1)
    assert ok is True
    assert timed_out is False
    assert reason is None


@pytest.mark.asyncio
async def test_soak_request_rejects_invalid_json(public_cases):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"not-json"))
    async with httpx.AsyncClient(base_url="https://example.test", transport=transport) as client:
        ok, _, _, reason = await _one_request(client, public_cases[0]["input"], 1)
    assert ok is False
    assert reason == "invalid JSON"


def test_release_threshold_gate():
    report = {
        "failures": 0,
        "latency_s": {"p95": 4.9, "max": 10.0},
        "health": {"failures": 0, "max_latency_s": 0.2},
    }
    assert _passes(report, 5, 30, 1)
    report["latency_s"]["p95"] = 5.1
    assert not _passes(report, 5, 30, 1)
