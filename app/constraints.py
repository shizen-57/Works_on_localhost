"""Compile validated directives into per-hour optimizer bounds.

Input directives here have ALREADY passed app/guardrails.py -- this module
never sees raw LLM output and does no semantic validation of its own. It
only implements the deterministic math in Problem Statement Sec. 5.3:

    solar_reduction          effective_solar[h] = base_solar[h] * factor
    minimum_battery_reserve  reserve[h] = max(base_reserve, directive reserve)
    no_charge_window         charge_limit[h] = 0
    no_discharge_window      discharge_limit[h] = 0
    max_grid_window          grid_cap[h] = directive cap

Overlap policy (not specified by the organizer -- documented assumption,
see README "Known limitations"): multiple minimum_battery_reserve directives
on the same hour take the max; multiple max_grid_window take the min;
multiple solar_reduction take the most restrictive (lowest resulting
effective solar), applied by successive narrowing exactly as here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.schemas import BatteryConfig, HourEntry


@dataclass(frozen=True)
class HourBounds:
    effective_solar: list[float]  # len 24
    reserve: list[float]  # len 24, active minimum_energy_kwh per hour
    grid_cap: list[float]  # len 24, math.inf where no cap applies
    charge_limit: list[float]  # len 24
    discharge_limit: list[float]  # len 24


def compile_bounds(
    hours: list[HourEntry],
    battery: BatteryConfig,
    directives: list[dict],
) -> HourBounds:
    """`hours` must already be sorted by `.hour` (see ScenarioRequest.hours_by_index)."""
    if [h.hour for h in hours] != list(range(24)):
        raise ValueError("hours must be sorted 0..23")

    effective_solar = [h.solar_kwh for h in hours]
    reserve = [battery.minimum_energy_kwh] * 24
    grid_cap = [math.inf] * 24
    charge_limit = [battery.max_charge_kwh_per_hour] * 24
    discharge_limit = [battery.max_discharge_kwh_per_hour] * 24

    for d in directives:
        dtype = d["directive_type"]
        if dtype == "no_op":
            continue
        adj = d["structured_adjustment"]
        for h in adj["hours"]:
            if dtype == "solar_reduction":
                effective_solar[h] = min(effective_solar[h], hours[h].solar_kwh * adj["factor"])
            elif dtype == "minimum_battery_reserve":
                reserve[h] = max(reserve[h], adj["minimum_energy_kwh"])
            elif dtype == "max_grid_window":
                grid_cap[h] = min(grid_cap[h], adj["max_grid_kwh"])
            elif dtype == "no_charge_window":
                charge_limit[h] = 0.0
            elif dtype == "no_discharge_window":
                discharge_limit[h] = 0.0
            else:  # pragma: no cover -- guardrails.py should never let this through
                raise ValueError(f"unsupported directive_type reached constraints.py: {dtype!r}")

    return HourBounds(
        effective_solar=effective_solar,
        reserve=reserve,
        grid_cap=grid_cap,
        charge_limit=charge_limit,
        discharge_limit=discharge_limit,
    )
