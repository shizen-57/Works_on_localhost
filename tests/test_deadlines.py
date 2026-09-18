from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient


class HangingLLMClient:
    """Never returns within the request deadline -- proves a stuck
    provider call cannot hang the whole request past the configured
    budget, and that timing out maps to a controlled 500, not a hang or
    crash."""

    async def interpret(self, notes, battery, correction_feedback=None) -> list[dict]:
        await asyncio.sleep(10)
        return []  # pragma: no cover -- never reached within the test's short deadline


def test_slow_llm_call_hits_internal_deadline_not_the_caller(monkeypatch, public_cases):
    monkeypatch.setenv("REQUEST_DEADLINE_S", "1")
    from app.main import app  # import after env is set so lifespan reads the short deadline

    with TestClient(app) as client:
        app.state.llm_client = HangingLLMClient()
        start = time.monotonic()
        r = client.post("/optimize-energy", json=public_cases[0]["input"])
        elapsed = time.monotonic() - start

    assert r.status_code == 500
    assert r.json()["error"] == "internal_deadline_exceeded"
    # Should return close to the 1s deadline, not wait out the 10s sleep.
    assert elapsed < 5, f"took {elapsed:.2f}s -- deadline was not enforced"
