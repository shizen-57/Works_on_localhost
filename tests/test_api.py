from __future__ import annotations

import asyncio
import copy
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.llm_client import LLMTransientError
from app.main import app


class FixtureLLMClient:
    """Test-only LLMClient that returns a pre-baked directive list,
    ignoring notes/battery -- used to exercise the full HTTP contract
    (schema, echoing, replay) with organizer-correct semantics, without a
    live provider key."""

    def __init__(self, directives: list[dict]):
        self._directives = directives

    async def interpret(self, notes, battery, correction_feedback=None) -> list[dict]:
        return copy.deepcopy(self._directives)


class AlwaysFailingLLMClient:
    async def interpret(self, notes, battery, correction_feedback=None) -> list[dict]:
        raise LLMTransientError("simulated provider outage")


class TransientThenGoodLLMClient:
    def __init__(self, good_directives: list[dict]):
        self._good = good_directives
        self.calls = 0

    async def interpret(self, notes, battery, correction_feedback=None) -> list[dict]:
        self.calls += 1
        if self.calls == 1:
            raise LLMTransientError("simulated one-off provider outage")
        return copy.deepcopy(self._good)


class SlowLLMClient:
    def __init__(self, directives: list[dict], started: threading.Event):
        self._directives = directives
        self._started = started

    async def interpret(self, notes, battery, correction_feedback=None) -> list[dict]:
        self._started.set()
        await asyncio.sleep(0.5)
        return copy.deepcopy(self._directives)


class BadThenGoodLLMClient:
    """Returns an unsupported directive_type on the first call, then a
    valid response on the bounded correction retry -- exercises
    app.llm_interpreter's one-retry policy end to end."""

    def __init__(self, good_directives: list[dict]):
        self._good = good_directives
        self.calls = 0

    async def interpret(self, notes, battery, correction_feedback=None) -> list[dict]:
        self.calls += 1
        if correction_feedback is None:
            return [
                dict(
                    note_index=i,
                    applies=True,
                    directive_type="not_a_real_type",
                    structured_adjustment=dict(hours=[0]),
                    explanation="bad",
                )
                for i in range(len(notes))
            ]
        return copy.deepcopy(self._good)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_optimize_energy_matches_organizer_cost_through_full_http_contract(client, public_cases):
    for case in public_cases:
        app.state.llm_client = FixtureLLMClient(case["expected_output"]["directive_interpretation"])
        r = client.post("/optimize-energy", json=case["input"])
        assert r.status_code == 200, (case["id"], r.text)
        body = r.json()

        assert body["scenario_id"] == case["input"]["scenario_id"]
        assert len(body["hourly_plan"]) == 24
        assert len(body["directive_interpretation"]) == len(case["input"]["operator_notes"])
        gap = abs(body["total_cost_bdt"] - case["expected_output"]["total_cost_bdt"])
        assert gap < 0.01, (case["id"], gap)


@pytest.mark.parametrize(
    "body,content_type",
    [
        (b"{not valid json", "application/json"),
    ],
)
def test_malformed_json_returns_400(client, body, content_type):
    r = client.post("/optimize-energy", content=body, headers={"content-type": content_type})
    assert r.status_code == 400
    assert "error" in r.json()


def test_duplicate_json_key_returns_400(client):
    body = b'{"scenario_id":"x","scenario_id":"y","operator_notes":["a"],"hours":[],"battery":{}}'
    r = client.post("/optimize-energy", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_nonfinite_literal_returns_400(client):
    body = b'{"scenario_id":"x","operator_notes":["a"],"hours":[],"battery":{},"x":NaN}'
    r = client.post("/optimize-energy", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_missing_fields_returns_400(client):
    r = client.post("/optimize-energy", json={"scenario_id": "x"})
    assert r.status_code == 400


def test_wrong_hours_count_returns_400(client, public_cases):
    req = copy.deepcopy(public_cases[0]["input"])
    req["hours"] = req["hours"][:23]
    r = client.post("/optimize-energy", json=req)
    assert r.status_code == 400


@pytest.mark.parametrize("body", [b"", b"null", b"[]", b'"text"', b"123", b"true", b"\xff\xfe"])
def test_non_object_or_invalid_utf8_bodies_return_controlled_400(client, body):
    r = client.post("/optimize-energy", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert set(r.json()) == {"error", "detail"}


def test_nested_duplicate_json_key_returns_400(client):
    body = b'{"scenario_id":"x","operator_notes":["a"],"hours":[],"battery":{"capacity_kwh":1,"capacity_kwh":2}}'
    r = client.post("/optimize-energy", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert r.json()["error"] == "malformed_json"


def test_request_body_limit_returns_controlled_400(client):
    body = b" " * (256 * 1024 + 1)
    r = client.post("/optimize-energy", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert r.json()["error"] == "request_too_large"


@pytest.mark.parametrize(
    "field,value",
    [
        ("scenario_id", "s" * 257),
        ("operator_notes", ["n" * 4001]),
    ],
)
def test_string_resource_limits_return_400(client, public_cases, field, value):
    request = copy.deepcopy(public_cases[0]["input"])
    request[field] = value
    r = client.post("/optimize-energy", json=request)
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_request"


def test_health_remains_responsive_during_slow_model_call(client, public_cases):
    case = public_cases[0]
    started = threading.Event()
    app.state.llm_client = SlowLLMClient(case["expected_output"]["directive_interpretation"], started)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(client.post, "/optimize-energy", json=case["input"])
        assert started.wait(timeout=2)
        health = client.get("/health")
        result = future.result(timeout=5)
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert result.status_code == 200


def test_llm_persistent_failure_returns_controlled_500_no_leakage(client, public_cases):
    app.state.llm_client = AlwaysFailingLLMClient()
    r = client.post("/optimize-energy", json=public_cases[0]["input"])
    assert r.status_code == 500
    text = r.text
    assert "Traceback" not in text
    assert "simulated provider outage" not in text  # internal detail must not leak to the client
    body = r.json()
    assert "error" in body and "detail" in body


def test_bounded_correction_retry_recovers_from_bad_first_attempt(client, public_cases):
    case = public_cases[0]
    fake = BadThenGoodLLMClient(case["expected_output"]["directive_interpretation"])
    app.state.llm_client = fake
    r = client.post("/optimize-energy", json=case["input"])
    assert r.status_code == 200, r.text
    assert fake.calls == 2
    gap = abs(r.json()["total_cost_bdt"] - case["expected_output"]["total_cost_bdt"])
    assert gap < 0.01


def test_bounded_retry_recovers_from_one_transient_provider_failure(client, public_cases):
    case = public_cases[0]
    fake = TransientThenGoodLLMClient(case["expected_output"]["directive_interpretation"])
    app.state.llm_client = fake
    r = client.post("/optimize-energy", json=case["input"])
    assert r.status_code == 200, r.text
    assert fake.calls == 2
