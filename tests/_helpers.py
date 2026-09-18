"""Shared test-only helpers: seeded random scenario generation and an
independent MILP oracle. Adapted from review/smoke_plan.py (the external
design audit) into production-fixture form -- same feasibility construction
(idle-battery witness), rewritten against app.schemas / app.constraints
instead of duplicating bound-compilation ad hoc.
"""

from __future__ import annotations

import random

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from app.constraints import compile_bounds
from app.schemas import BatteryConfig, HourEntry, ScenarioRequest


def make_directive(directive_type: str, hours: list[int], note_index: int, **kwargs) -> dict:
    if directive_type == "no_op":
        return dict(
            note_index=note_index,
            applies=False,
            directive_type="no_op",
            structured_adjustment=None,
            explanation="synthetic",
        )
    return dict(
        note_index=note_index,
        applies=True,
        directive_type=directive_type,
        structured_adjustment=dict(hours=sorted(hours), **kwargs),
        explanation="synthetic",
    )


def random_case(rng: random.Random, index: int) -> tuple[dict, list[dict]]:
    """Returns (request_dict, directives) guaranteed feasible via an
    idle-battery witness: grid=demand, solar_used=0, charge=discharge=0
    satisfies every generated directive by construction."""
    capacity = rng.uniform(0, 500) if index % 17 else 0
    reserve = rng.uniform(0, capacity)
    initial = rng.uniform(reserve, capacity)

    request = dict(
        scenario_id=f"RANDOM-{index}",
        operator_notes=["synthetic"],
        battery=dict(
            capacity_kwh=capacity,
            initial_energy_kwh=initial,
            minimum_energy_kwh=reserve,
            max_charge_kwh_per_hour=0.0 if index % 13 == 0 else rng.uniform(0, 100),
            max_discharge_kwh_per_hour=0.0 if index % 11 == 0 else rng.uniform(0, 100),
        ),
        hours=[
            dict(
                hour=h,
                demand_kwh=0.0 if index % 19 == 0 else rng.uniform(0, 300),
                solar_kwh=rng.uniform(0, 450),
                tariff_bdt_per_kwh=0.0 if index % 7 == 0 else rng.uniform(0, 50),
            )
            for h in range(24)
        ],
    )
    demand_by_hour = {h["hour"]: h["demand_kwh"] for h in request["hours"]}

    n_notes = rng.randint(1, 3)
    directives = []
    for n in range(n_notes):
        hs = sorted(rng.sample(range(24), rng.randint(1, 24)))
        dtype = rng.choice(
            [
                "solar_reduction",
                "minimum_battery_reserve",
                "no_charge_window",
                "no_discharge_window",
                "max_grid_window",
            ]
        )
        if dtype == "solar_reduction":
            directives.append(make_directive(dtype, hs, n, factor=rng.choice([0.0, 1.0, rng.random()])))
        elif dtype == "minimum_battery_reserve":
            # Idle witness keeps energy at `initial` all day, so any reserve
            # <= initial is satisfiable without the LP needing to charge.
            directives.append(make_directive(dtype, hs, n, minimum_energy_kwh=rng.uniform(reserve, initial)))
        elif dtype == "max_grid_window":
            cap = max(demand_by_hour[h] for h in hs)
            directives.append(make_directive(dtype, hs, n, max_grid_kwh=cap))
        else:
            directives.append(make_directive(dtype, hs, n))

    request["operator_notes"] = ["synthetic"] * n_notes
    rng.shuffle(request["hours"])  # exercise "never trust array position"
    return request, directives


def oracle_cost(hours: list[HourEntry], battery: BatteryConfig, directives: list[dict]) -> float:
    """Independent explicit-state MILP with binary charge/discharge
    exclusivity. Confirms the continuous LP + full-netting formulation in
    app.optimizer reaches the true optimum, not just a feasible netted
    point. Bound compilation is intentionally reused from app.constraints
    here (this checks optimality of the LP relaxation over shared bounds,
    not bound-compilation correctness -- that is covered independently by
    replaying all 10 public cases and by test_replay.py's mutation probes)."""
    bounds = compile_bounds(hours, battery, directives)
    n = 6 * 24  # per hour: grid, solar, charge, discharge, energy_after, charge_enabled(binary)
    obj = np.zeros(n)
    lower = np.zeros(n)
    upper = np.zeros(n)
    integrality = np.zeros(n)
    rows, lows, highs = [], [], []

    for h in range(24):
        j = 6 * h
        cap = bounds.grid_cap[h]
        obj[j] = hours[h].tariff_bdt_per_kwh
        lower[j + 4] = bounds.reserve[h]
        upper[j : j + 6] = [
            1e12 if math_isinf(cap) else cap,
            bounds.effective_solar[h],
            bounds.charge_limit[h],
            bounds.discharge_limit[h],
            battery.capacity_kwh,
            1,
        ]
        integrality[j + 5] = 1

        row = np.zeros(n)
        row[j : j + 4] = [1, 1, -1, 1]
        rows.append(row)
        lows.append(hours[h].demand_kwh)
        highs.append(hours[h].demand_kwh)

        row = np.zeros(n)
        row[j + 4] = 1
        row[j + 2] = -1
        row[j + 3] = 1
        if h:
            row[j - 2] = -1  # previous hour's energy_after
        rows.append(row)
        lows.append(0.0 if h else battery.initial_energy_kwh)
        highs.append(lows[-1])

        row = np.zeros(n)
        row[j + 2] = 1
        row[j + 5] = -bounds.charge_limit[h]
        rows.append(row)
        lows.append(-np.inf)
        highs.append(0.0)

        row = np.zeros(n)
        row[j + 3] = 1
        row[j + 5] = bounds.discharge_limit[h]
        rows.append(row)
        lows.append(-np.inf)
        highs.append(bounds.discharge_limit[h])

    lower[-2] = upper[-2] = battery.initial_energy_kwh

    result = milp(
        obj,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(np.array(rows), lows, highs),
        options={"time_limit": 10, "mip_rel_gap": 0},
    )
    if not result.success:
        raise AssertionError(f"oracle MILP failed: {result.message}")
    return float(result.fun)


def math_isinf(x: float) -> bool:
    return x == float("inf")


def to_scenario_request(request_dict: dict) -> ScenarioRequest:
    return ScenarioRequest.model_validate(request_dict)
