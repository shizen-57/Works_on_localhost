from __future__ import annotations

import json
import math
import random

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from app.constraints import compile_bounds
from app.guardrails import GuardrailViolation, validate_directives
from app.optimizer import solve
from app.providers.sleepyai_provider import _extract_json_object
from app.replay_validator import replay
from app.schemas import ScenarioRequest
from tests._helpers import random_case
from tests.test_optimizer import _response_dict

json_scalars = st.none() | st.booleans() | st.integers() | st.floats(allow_nan=True, allow_infinity=True) | st.text()
json_values = st.recursive(
    json_scalars,
    lambda children: st.lists(children, max_size=8) | st.dictionaries(st.text(max_size=20), children, max_size=8),
    max_leaves=30,
)


@given(json_values)
@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
def test_arbitrary_json_never_crashes_request_validation(value):
    try:
        ScenarioRequest.model_validate(value)
    except ValidationError:
        pass


@given(json_values, st.integers(min_value=1, max_value=3), st.floats(min_value=0, max_value=1e6))
@settings(max_examples=300)
def test_arbitrary_json_never_escapes_guardrail_contract(raw, note_count, capacity):
    try:
        clean = validate_directives(raw, note_count, capacity)
    except GuardrailViolation:
        return
    assert [entry["note_index"] for entry in clean] == list(range(note_count))
    assert all(
        entry["directive_type"]
        in {
            "solar_reduction",
            "minimum_battery_reserve",
            "no_charge_window",
            "no_discharge_window",
            "max_grid_window",
            "no_op",
        }
        for entry in clean
    )


@given(st.lists(st.integers(min_value=0, max_value=23), min_size=1, max_size=100))
def test_valid_duplicate_hours_are_normalized_without_changing_membership(hours):
    raw = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "no_charge_window",
            "structured_adjustment": {"hours": hours},
            "explanation": "synthetic",
        }
    ]
    clean = validate_directives(raw, 1, 100)
    assert clean[0]["structured_adjustment"]["hours"] == sorted(set(hours))


@given(
    st.dictionaries(
        st.text(min_size=1, max_size=10),
        json_scalars.filter(lambda value: not isinstance(value, float) or math.isfinite(value)),
        max_size=10,
    )
)
def test_strict_model_json_round_trips_single_objects(value):
    encoded = json.dumps(value, allow_nan=False)
    assert _extract_json_object(encoded) == value


@given(st.integers(min_value=0, max_value=2**32 - 1))
@settings(max_examples=100, deadline=None)
def test_seeded_optimizer_cases_always_replay(seed):
    request, directives = random_case(random.Random(seed), seed % 10_000 + 1)
    parsed = ScenarioRequest.model_validate(request)
    hours = parsed.hours_by_index()
    result = solve(hours, parsed.battery, compile_bounds(hours, parsed.battery, directives))
    replay(request, directives, _response_dict(parsed, directives, result))
