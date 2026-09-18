from __future__ import annotations

import copy

import pytest

from app.constraints import compile_bounds
from app.replay_validator import ReplayViolation, replay
from app.schemas import ScenarioRequest


@pytest.fixture()
def good_response_and_request(public_cases):
    case = public_cases[4]  # SAMPLE-05, has an active max_grid_window directive
    req = ScenarioRequest.model_validate(case["input"])
    hours = req.hours_by_index()
    directives = case["expected_output"]["directive_interpretation"]
    from app.optimizer import solve

    bounds = compile_bounds(hours, req.battery, directives)
    result = solve(hours, req.battery, bounds)
    response = dict(
        scenario_id=req.scenario_id,
        hourly_plan=[
            dict(
                hour=p.hour,
                grid_kwh=p.grid_kwh,
                solar_used_kwh=p.solar_used_kwh,
                battery_action=p.battery_action,
                battery_kwh=p.battery_kwh,
                battery_energy_after_kwh=p.battery_energy_after_kwh,
            )
            for p in result.hourly_plan
        ],
        total_grid_kwh=result.total_grid_kwh,
        total_cost_bdt=result.total_cost_bdt,
        peak_grid_kwh=result.peak_grid_kwh,
    )
    return case["input"], directives, response


def test_valid_response_passes(good_response_and_request):
    request, directives, response = good_response_and_request
    replay(request, directives, response)  # must not raise


@pytest.mark.parametrize(
    "field,value",
    [
        ("grid_kwh", -1),
        ("solar_used_kwh", 1e6),
        ("battery_energy_after_kwh", 1e6),
        ("battery_kwh", float("nan")),
    ],
)
def test_hourly_plan_field_mutations_rejected(good_response_and_request, field, value):
    request, directives, response = good_response_and_request
    bad = copy.deepcopy(response)
    bad["hourly_plan"][0][field] = value
    with pytest.raises(ReplayViolation):
        replay(request, directives, bad)


def test_invalid_battery_action_rejected(good_response_and_request):
    request, directives, response = good_response_and_request
    bad = copy.deepcopy(response)
    bad["hourly_plan"][0]["battery_action"] = "export"
    with pytest.raises(ReplayViolation):
        replay(request, directives, bad)


@pytest.mark.parametrize("field", ["total_grid_kwh", "total_cost_bdt", "peak_grid_kwh"])
def test_wrong_totals_rejected(good_response_and_request, field):
    request, directives, response = good_response_and_request
    bad = copy.deepcopy(response)
    bad[field] += 1
    with pytest.raises(ReplayViolation):
        replay(request, directives, bad)


def test_missing_hour_entry_rejected(good_response_and_request):
    request, directives, response = good_response_and_request
    bad = copy.deepcopy(response)
    bad["hourly_plan"].pop()
    with pytest.raises(ReplayViolation):
        replay(request, directives, bad)


def test_duplicated_hour_index_rejected(good_response_and_request):
    request, directives, response = good_response_and_request
    bad = copy.deepcopy(response)
    bad["hourly_plan"][1]["hour"] = 0
    with pytest.raises(ReplayViolation):
        replay(request, directives, bad)


def test_wrong_scenario_id_rejected(good_response_and_request):
    request, directives, response = good_response_and_request
    bad = copy.deepcopy(response)
    bad["scenario_id"] = "wrong"
    with pytest.raises(ReplayViolation):
        replay(request, directives, bad)


def test_directive_ignored_in_schedule_rejected(good_response_and_request):
    """A schedule that satisfies energy balance/battery bounds but ignores
    the true active directive (e.g. exceeds a max_grid_window cap) must be
    rejected -- correct extraction without correct application is a fail
    (Problem Statement Sec. 11.2)."""
    request, directives, response = good_response_and_request
    grid_cap_directive = next(d for d in directives if d["directive_type"] == "max_grid_window")
    capped_hour = grid_cap_directive["structured_adjustment"]["hours"][0]
    bad = copy.deepcopy(response)
    entry = bad["hourly_plan"][capped_hour]
    over_cap = grid_cap_directive["structured_adjustment"]["max_grid_kwh"] + 10
    delta = over_cap - entry["grid_kwh"]
    entry["grid_kwh"] = over_cap
    entry["solar_used_kwh"] = max(0.0, entry["solar_used_kwh"] - delta)
    with pytest.raises(ReplayViolation):
        replay(request, directives, bad)
