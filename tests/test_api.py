from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from app.llm_client import LLMTransientError
from app.main import app
from app.schemas import BatteryConfig


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
                dict(note_index=i, applies=True, directive_type="not_a_real_type",
                     structured_adjustment=dict(hours=[0]), explanation="bad")
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


@pytest.mark.parametrize("body,content_type", [
    (b"{not valid json", "application/json"),
])
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
