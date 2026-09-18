"""Independent final gate: replay the emitted response against every rule.

This is the production analogue of the judge's own independent replay
(Problem Statement Sec. 11: "The completed schedule is replayed after
optimization to verify every extracted directive was actually followed").

Deliberately does NOT import app.constraints or reuse its compiled bound
arrays. Bounds are re-derived here directly from the raw directive list,
in a separate code path, so a bug in constraints.py's bound-compilation
cannot also make its own downstream check pass -- the audit's plan_review
flagged sharing bound compilation between the optimizer and its checker as
a common-mode risk. The two implementations are intentionally similar in
shape (same spec, same math) but structurally independent.

Call this on the response AFTER a JSON round trip (json.loads(json.dumps(...))),
never on the optimizer's in-memory dataclasses -- serialization itself can
introduce the failures this is meant to catch (see errors.py for how
main.py wires this in).
"""
from __future__ import annotations

import math

DEFAULT_TOL = 1e-6  # stricter than the judge's documented 0.01 -- see plan.


class ReplayViolation(ValueError):
    """Raised on the first rule the emitted plan fails to satisfy."""


def _check(ok: bool, reason: str) -> None:
    if not ok:
        raise ReplayViolation(reason)


def replay(request: dict, directives: list[dict], response: dict, tol: float = DEFAULT_TOL) -> None:
    """`request` and `response` are plain JSON-shaped dicts (post round-trip).

    `directives` is the validated (guardrailed) directive_interpretation
    list -- the ground truth to check the plan against, which in
    production is the same list embedded in `response`, but callers may
    also pass an organizer reference interpretation when replaying
    reference/test fixtures.
    """
    _check(response.get("scenario_id") == request.get("scenario_id"), "scenario_id mismatch")

    plan = response.get("hourly_plan")
    _check(isinstance(plan, list) and len(plan) == 24, "hourly_plan must have 24 entries")
    _check([p.get("hour") for p in plan] == list(range(24)), "hourly_plan hours must be 0..23 in order")

    source_by_hour = {h["hour"]: h for h in request["hours"]}
    battery = request["battery"]

    energy = battery["initial_energy_kwh"]
    total_grid = 0.0
    total_cost = 0.0
    peak_grid = 0.0

    for p in plan:
        h = p["hour"]
        src = source_by_hour[h]

        # Re-derive this hour's active bounds directly from raw directives.
        solar_limit = src["solar_kwh"]
        reserve = battery["minimum_energy_kwh"]
        grid_cap = math.inf
        charge_limit = battery["max_charge_kwh_per_hour"]
        discharge_limit = battery["max_discharge_kwh_per_hour"]
        for d in directives:
            adj = d.get("structured_adjustment")
            if adj is None or h not in adj.get("hours", ()):
                continue
            dtype = d["directive_type"]
            if dtype == "solar_reduction":
                solar_limit = min(solar_limit, src["solar_kwh"] * adj["factor"])
            elif dtype == "minimum_battery_reserve":
                reserve = max(reserve, adj["minimum_energy_kwh"])
            elif dtype == "max_grid_window":
                grid_cap = min(grid_cap, adj["max_grid_kwh"])
            elif dtype == "no_charge_window":
                charge_limit = 0.0
            elif dtype == "no_discharge_window":
                discharge_limit = 0.0

        for key in ("grid_kwh", "solar_used_kwh", "battery_kwh", "battery_energy_after_kwh"):
            v = p.get(key)
            _check(isinstance(v, (int, float)) and not isinstance(v, bool), f"{key} not numeric")
            _check(math.isfinite(v), f"{key} not finite")
        grid = float(p["grid_kwh"])
        solar_used = float(p["solar_used_kwh"])
        action = p.get("battery_action")
        amount = float(p["battery_kwh"])

        _check(grid >= -tol, "grid_kwh negative")
        _check(solar_used >= -tol, "solar_used_kwh negative")
        _check(amount >= -tol, "battery_kwh negative")
        _check(action in ("charge", "discharge", "idle"), "battery_action invalid")
        _check(action != "idle" or amount <= tol, "idle action must have battery_kwh == 0")

        charge = amount if action == "charge" else 0.0
        discharge = amount if action == "discharge" else 0.0

        _check(charge <= charge_limit + tol, "charge exceeds rate limit or no_charge_window")
        _check(discharge <= discharge_limit + tol, "discharge exceeds rate limit or no_discharge_window")
        _check(solar_used <= solar_limit + tol, "solar_used_kwh exceeds effective solar")
        _check(grid <= grid_cap + tol, "grid_kwh exceeds max_grid_window cap")

        energy = energy + charge - discharge
        _check(abs(energy - p["battery_energy_after_kwh"]) <= tol, "battery_energy_after_kwh transition mismatch")
        _check(reserve - tol <= energy <= battery["capacity_kwh"] + tol, "battery energy out of reserve/capacity bounds")

        expected_balance = grid + solar_used + discharge - charge
        _check(abs(expected_balance - src["demand_kwh"]) <= tol, "energy balance violated")

        total_grid += grid
        total_cost += grid * src["tariff_bdt_per_kwh"]
        peak_grid = max(peak_grid, grid)

    _check(abs(energy - battery["initial_energy_kwh"]) <= tol, "end-of-day battery neutrality violated")

    for key, expected in (
        ("total_grid_kwh", total_grid),
        ("total_cost_bdt", total_cost),
        ("peak_grid_kwh", peak_grid),
    ):
        v = response.get(key)
        _check(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v), f"{key} not numeric/finite")
        _check(abs(v - expected) <= tol, f"{key} does not match value recalculated from hourly_plan")
