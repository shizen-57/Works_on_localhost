from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from app.schemas import BatteryConfig, ScenarioRequest


def _valid_request(public_cases):
    return copy.deepcopy(public_cases[0]["input"])


def test_valid_request_parses(public_cases):
    req = ScenarioRequest.model_validate(_valid_request(public_cases))
    assert req.scenario_id == "SAMPLE-01"
    assert [h.hour for h in req.hours_by_index()] == list(range(24))


@pytest.mark.parametrize("n", [23, 25])
def test_wrong_hour_count_rejected(public_cases, n):
    req = _valid_request(public_cases)
    req["hours"] = req["hours"][:n] if n < 24 else req["hours"] + [req["hours"][0]]
    with pytest.raises(ValidationError):
        ScenarioRequest.model_validate(req)


def test_duplicate_hour_value_rejected(public_cases):
    req = _valid_request(public_cases)
    req["hours"][1]["hour"] = req["hours"][0]["hour"]
    with pytest.raises(ValidationError):
        ScenarioRequest.model_validate(req)


@pytest.mark.parametrize("notes", [[], ["a", "b", "c", "d"]])
def test_note_count_out_of_range_rejected(public_cases, notes):
    req = _valid_request(public_cases)
    req["operator_notes"] = notes
    with pytest.raises(ValidationError):
        ScenarioRequest.model_validate(req)


def test_whitespace_only_note_rejected(public_cases):
    req = _valid_request(public_cases)
    req["operator_notes"] = ["   "]
    with pytest.raises(ValidationError):
        ScenarioRequest.model_validate(req)


def test_unknown_top_level_field_rejected(public_cases):
    req = _valid_request(public_cases)
    req["extra_field"] = "nope"
    with pytest.raises(ValidationError):
        ScenarioRequest.model_validate(req)


def test_unknown_nested_field_rejected(public_cases):
    req = _valid_request(public_cases)
    req["battery"]["extra"] = 1
    with pytest.raises(ValidationError):
        ScenarioRequest.model_validate(req)
    req = _valid_request(public_cases)
    req["hours"][0]["extra"] = 1
    with pytest.raises(ValidationError):
        ScenarioRequest.model_validate(req)


@pytest.mark.parametrize("field", ["demand_kwh", "solar_kwh", "tariff_bdt_per_kwh"])
def test_bool_rejected_for_hour_numeric_field(public_cases, field):
    req = _valid_request(public_cases)
    req["hours"][0][field] = True
    with pytest.raises(ValidationError):
        ScenarioRequest.model_validate(req)


def test_numeric_string_rejected(public_cases):
    req = _valid_request(public_cases)
    req["hours"][0]["demand_kwh"] = "90"
    with pytest.raises(ValidationError):
        ScenarioRequest.model_validate(req)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_rejected(public_cases, value):
    req = _valid_request(public_cases)
    req["hours"][0]["demand_kwh"] = value
    with pytest.raises(ValidationError):
        ScenarioRequest.model_validate(req)


def test_negative_tariff_is_permitted(public_cases):
    """Documented divergence from a stricter reading of the spec: tariffs
    may be negative (Problem Statement Sec. 07 only says "number")."""
    req = _valid_request(public_cases)
    req["hours"][0]["tariff_bdt_per_kwh"] = -5.0
    ScenarioRequest.model_validate(req)  # must not raise


def test_battery_starting_below_reserve_is_permitted():
    """Documented divergence: initial_energy_kwh < minimum_energy_kwh is
    not rejected at the schema layer -- Sec. 9.2 only constrains E_after,
    and hour 0 could charge up into compliance. Let the LP decide."""
    BatteryConfig(
        capacity_kwh=100, initial_energy_kwh=10, minimum_energy_kwh=50,
        max_charge_kwh_per_hour=50, max_discharge_kwh_per_hour=50,
    )  # must not raise
