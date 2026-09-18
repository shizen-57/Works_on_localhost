from __future__ import annotations

import random

import pytest

from app.constraints import compile_bounds
from app.optimizer import InfeasibleScenario, solve
from app.replay_validator import replay
from app.schemas import BatteryConfig, ScenarioRequest
from tests._helpers import make_directive, oracle_cost, random_case

SEED = 20260918  # matches review/smoke_results_reference.json for cross-comparison


def _response_dict(req: ScenarioRequest, directives: list[dict], result) -> dict:
    return dict(
        scenario_id=req.scenario_id,
        directive_interpretation=directives,
        hourly_plan=[
            dict(hour=p.hour, grid_kwh=p.grid_kwh, solar_used_kwh=p.solar_used_kwh,
                 battery_action=p.battery_action, battery_kwh=p.battery_kwh,
                 battery_energy_after_kwh=p.battery_energy_after_kwh)
            for p in result.hourly_plan
        ],
        total_grid_kwh=result.total_grid_kwh, total_cost_bdt=result.total_cost_bdt,
        peak_grid_kwh=result.peak_grid_kwh,
    )


def test_all_public_cases_match_organizer_optimal_cost(public_cases):
    for case in public_cases:
        req = ScenarioRequest.model_validate(case["input"])
        hours = req.hours_by_index()
        directives = case["expected_output"]["directive_interpretation"]
        bounds = compile_bounds(hours, req.battery, directives)
        result = solve(hours, req.battery, bounds)

        gap = abs(result.total_cost_bdt - case["expected_output"]["total_cost_bdt"])
        assert gap < 0.01, f"{case['id']}: cost gap {gap}"

        response = _response_dict(req, directives, result)
        replay(case["input"], directives, response, tol=1e-6)  # our own produced plan


def test_organizer_reference_plans_replay_cleanly(public_cases):
    """The organizer's own expected_output plans, checked against our
    independent replay implementation at the documented 0.01 tolerance."""
    for case in public_cases:
        replay(case["input"], case["expected_output"]["directive_interpretation"],
               case["expected_output"], tol=0.01)


def test_seeded_random_feasible_cases_replay_cleanly():
    rng = random.Random(SEED)
    oracle_checks = 0
    for i in range(300):
        request, directives = random_case(rng, i)
        req = ScenarioRequest.model_validate(request)
        hours = req.hours_by_index()
        bounds = compile_bounds(hours, req.battery, directives)
        result = solve(hours, req.battery, bounds)  # must not raise -- feasible by construction

        response = _response_dict(req, directives, result)
        replay(request, directives, response)  # tol defaults to 1e-6

        # Battery action purity: idle <=> zero magnitude; never both
        # charge and discharge nonzero for the same hour (the netting
        # invariant -- see app/optimizer.py docstring).
        for p in result.hourly_plan:
            assert p.battery_action in ("charge", "discharge", "idle")
            if p.battery_action == "idle":
                assert p.battery_kwh == 0.0

        if i < 40:
            oracle = oracle_cost(hours, req.battery, directives)
            assert abs(oracle - result.total_cost_bdt) < 1e-4, f"case {i}: LP {result.total_cost_bdt} vs oracle {oracle}"
            oracle_checks += 1
    assert oracle_checks == 40


def test_infeasible_scenario_raises_not_crashes():
    """Zero-everything battery + zero solar + a zero grid cap makes the
    hour unschedulable: demand cannot be met. Must raise a typed
    exception, never crash or silently return an invalid plan."""
    req = ScenarioRequest.model_validate(dict(
        scenario_id="INFEASIBLE", operator_notes=["x"],
        battery=dict(capacity_kwh=0, initial_energy_kwh=0, minimum_energy_kwh=0,
                     max_charge_kwh_per_hour=0, max_discharge_kwh_per_hour=0),
        hours=[dict(hour=h, demand_kwh=50 if h == 0 else 0, solar_kwh=0, tariff_bdt_per_kwh=1)
               for h in range(24)],
    ))
    hours = req.hours_by_index()
    directives = [make_directive("max_grid_window", [0], 0, max_grid_kwh=0)]
    bounds = compile_bounds(hours, req.battery, directives)
    with pytest.raises(InfeasibleScenario):
        solve(hours, req.battery, bounds)


def test_two_decimal_rounding_would_break_tolerance():
    """Documents why app/optimizer.py never rounds energy fields: rounding
    24 small grid values to 2 decimals can erase real cost far past the
    judge's 0.01 BDT tolerance."""
    exact = 24 * 0.0049 * 30
    rounded = 24 * round(0.0049, 2) * 30
    assert abs(exact - rounded) > 0.01


def test_optimizer_source_never_rounds_energy_fields():
    """Static regression guard against a future edit reintroducing
    2-decimal rounding of grid/solar/battery fields."""
    import inspect

    from app import optimizer
    source = inspect.getsource(optimizer)
    assert "round(" not in source
