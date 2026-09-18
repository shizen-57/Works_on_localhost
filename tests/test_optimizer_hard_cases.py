from __future__ import annotations

import pytest

from app.constraints import compile_bounds
from app.optimizer import solve
from app.replay_validator import replay
from app.schemas import ScenarioRequest
from tests._helpers import make_directive
from tests.test_optimizer import _response_dict


def _request(*, demand=10.0, solar=0.0, tariff=10.0, battery=None) -> dict:
    return {
        "scenario_id": "HARD",
        "operator_notes": ["synthetic"],
        "hours": [
            {"hour": hour, "demand_kwh": demand, "solar_kwh": solar, "tariff_bdt_per_kwh": tariff} for hour in range(24)
        ],
        "battery": battery
        or {
            "capacity_kwh": 100.0,
            "initial_energy_kwh": 50.0,
            "minimum_energy_kwh": 0.0,
            "max_charge_kwh_per_hour": 20.0,
            "max_discharge_kwh_per_hour": 20.0,
        },
    }


def _solve_and_replay(request: dict, directives: list[dict]):
    request["operator_notes"] = ["synthetic"] * len(directives)
    parsed = ScenarioRequest.model_validate(request)
    hours = parsed.hours_by_index()
    result = solve(hours, parsed.battery, compile_bounds(hours, parsed.battery, directives))
    replay(request, directives, _response_dict(parsed, directives, result))
    return result


def test_overlapping_directives_choose_most_restrictive_bounds():
    request = _request(solar=100)
    directives = [
        make_directive("solar_reduction", [10], 0, factor=0.8),
        make_directive("solar_reduction", [10], 1, factor=0.2),
        make_directive("minimum_battery_reserve", [10], 2, minimum_energy_kwh=40),
    ]
    request["operator_notes"] = ["x", "y", "z"]
    parsed = ScenarioRequest.model_validate(request)
    bounds = compile_bounds(parsed.hours_by_index(), parsed.battery, directives)
    assert bounds.effective_solar[10] == 20
    assert bounds.reserve[10] == 40


def test_tight_grid_cap_forces_precharge_and_discharge():
    request = _request(demand=50, tariff=10)
    request["hours"][0]["tariff_bdt_per_kwh"] = 1
    request["battery"]["initial_energy_kwh"] = 0
    request["battery"]["minimum_energy_kwh"] = 0
    directives = [make_directive("max_grid_window", [1], 0, max_grid_kwh=30)]
    result = _solve_and_replay(request, directives)
    assert result.hourly_plan[0].battery_action == "charge"
    assert result.hourly_plan[1].battery_action == "discharge"


def test_negative_tariff_and_end_of_day_neutrality():
    request = _request(demand=10, tariff=0)
    request["hours"][0]["tariff_bdt_per_kwh"] = -10
    result = _solve_and_replay(request, [make_directive("no_op", [], 0)])
    assert result.total_cost_bdt < 0
    assert result.hourly_plan[0].grid_kwh > request["hours"][0]["demand_kwh"]
    assert result.hourly_plan[-1].battery_energy_after_kwh == pytest.approx(50)


def test_tiny_decimal_energy_is_not_rounded_away():
    request = _request(
        demand=0.000_004,
        tariff=30,
        battery={
            "capacity_kwh": 0.0,
            "initial_energy_kwh": 0.0,
            "minimum_energy_kwh": 0.0,
            "max_charge_kwh_per_hour": 0.0,
            "max_discharge_kwh_per_hour": 0.0,
        },
    )
    result = _solve_and_replay(request, [make_directive("no_op", [], 0)])
    assert result.total_grid_kwh == pytest.approx(24 * 0.000_004)
    assert result.total_cost_bdt == pytest.approx(24 * 0.000_004 * 30)
