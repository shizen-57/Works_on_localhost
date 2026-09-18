#!/usr/bin/env python3
"""Run all 10 public sample cases against a live GET /health + POST
/optimize-energy service and report pass/fail per case.

This is the exact command the README tells organizers to run for the
"public-sample test command" requirement (Participant Guide Sec. 03/05).
HTTP 200 alone is not a pass: this script re-derives the same checks
app/replay_validator.py runs internally (energy balance, battery bounds,
directive application, recalculated totals) PLUS compares interpreted
directive_type/hours/values against this pack's expected semantics, and
exits non-zero on any failure.

Usage:
    python scripts/run_public_samples.py --base-url http://localhost:8000
    python scripts/run_public_samples.py --base-url https://your-deployment.example.com
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import httpx

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "public_sample_cases.json"
TOL = 0.01  # Problem Statement Sec. 11.5 documented tolerance


def _replay(request: dict, directives: list[dict], response: dict, tol: float) -> list[str]:
    """Standalone re-implementation (no app import) so this script can run
    against a remote deployment with no local checkout of app/ needed."""
    errors: list[str] = []

    def check(ok: bool, reason: str) -> None:
        if not ok:
            errors.append(reason)

    check(response.get("scenario_id") == request.get("scenario_id"), "scenario_id mismatch")
    plan = response.get("hourly_plan") or []
    check(len(plan) == 24 and [p.get("hour") for p in plan] == list(range(24)), "hourly_plan hours must be 0..23 in order")
    if errors:
        return errors

    source = {h["hour"]: h for h in request["hours"]}
    battery = request["battery"]
    energy = battery["initial_energy_kwh"]
    total_grid = total_cost = peak_grid = 0.0

    for p in plan:
        h = p["hour"]
        src = source[h]
        solar_limit, reserve, grid_cap = src["solar_kwh"], battery["minimum_energy_kwh"], math.inf
        charge_limit, discharge_limit = battery["max_charge_kwh_per_hour"], battery["max_discharge_kwh_per_hour"]
        for d in directives:
            adj = d.get("structured_adjustment")
            if adj is None or h not in adj.get("hours", ()):
                continue
            t = d["directive_type"]
            if t == "solar_reduction":
                solar_limit = min(solar_limit, src["solar_kwh"] * adj["factor"])
            elif t == "minimum_battery_reserve":
                reserve = max(reserve, adj["minimum_energy_kwh"])
            elif t == "max_grid_window":
                grid_cap = min(grid_cap, adj["max_grid_kwh"])
            elif t == "no_charge_window":
                charge_limit = 0.0
            elif t == "no_discharge_window":
                discharge_limit = 0.0

        grid, solar_used = float(p["grid_kwh"]), float(p["solar_used_kwh"])
        action, amount = p["battery_action"], float(p["battery_kwh"])
        charge = amount if action == "charge" else 0.0
        discharge = amount if action == "discharge" else 0.0

        check(charge <= charge_limit + tol, f"h{h}: charge exceeds limit/no_charge_window")
        check(discharge <= discharge_limit + tol, f"h{h}: discharge exceeds limit/no_discharge_window")
        check(solar_used <= solar_limit + tol, f"h{h}: solar_used exceeds effective solar")
        check(grid <= grid_cap + tol, f"h{h}: grid exceeds max_grid_window cap")

        energy += charge - discharge
        check(abs(energy - p["battery_energy_after_kwh"]) <= tol, f"h{h}: battery transition mismatch")
        check(reserve - tol <= energy <= battery["capacity_kwh"] + tol, f"h{h}: battery out of bounds")
        check(abs(grid + solar_used + discharge - src["demand_kwh"] - charge) <= tol, f"h{h}: energy balance violated")

        total_grid += grid
        total_cost += grid * src["tariff_bdt_per_kwh"]
        peak_grid = max(peak_grid, grid)

    check(abs(energy - battery["initial_energy_kwh"]) <= tol, "end-of-day battery neutrality violated")
    for key, expected in (("total_grid_kwh", total_grid), ("total_cost_bdt", total_cost), ("peak_grid_kwh", peak_grid)):
        check(abs(response.get(key, float("nan")) - expected) <= tol, f"{key} does not match hourly_plan")
    return errors


def _check_interpretation(case: dict, got: list[dict]) -> list[str]:
    errors = []
    expected = {d["note_index"]: d for d in case["expected_output"]["directive_interpretation"]}
    got_by_index = {d["note_index"]: d for d in got}
    if set(got_by_index) != set(expected):
        errors.append(f"note_index coverage mismatch: got {sorted(got_by_index)} expected {sorted(expected)}")
        return errors
    for idx, exp in expected.items():
        act = got_by_index[idx]
        if act.get("directive_type") != exp["directive_type"]:
            errors.append(f"note {idx}: directive_type {act.get('directive_type')!r} != expected {exp['directive_type']!r}")
            continue
        if exp["directive_type"] == "no_op":
            continue
        exp_adj, act_adj = exp["structured_adjustment"], act.get("structured_adjustment") or {}
        if act_adj.get("hours") != exp_adj["hours"]:
            errors.append(f"note {idx}: hours {act_adj.get('hours')} != expected {exp_adj['hours']}")
        for key, val in exp_adj.items():
            if key == "hours":
                continue
            act_val = act_adj.get(key)
            if act_val is None or abs(act_val - val) > TOL:
                errors.append(f"note {idx}: {key}={act_val} != expected {val}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--timeout", type=float, default=35.0)
    args = parser.parse_args()

    cases = json.loads(DATA_PATH.read_text())["cases"]
    client = httpx.Client(base_url=args.base_url, timeout=args.timeout)

    health = client.get("/health")
    if health.status_code != 200 or health.json().get("status") != "ok":
        print(f"FAIL /health: status={health.status_code} body={health.text}")
        return 1
    print("PASS /health")

    failures = 0
    for case in cases:
        r = client.post("/optimize-energy", json=case["input"])
        if r.status_code != 200:
            print(f"FAIL {case['id']}: HTTP {r.status_code}: {r.text}")
            failures += 1
            continue
        body = r.json()

        interp_errors = _check_interpretation(case, body.get("directive_interpretation", []))
        replay_errors = _replay(case["input"], body.get("directive_interpretation", []), body, TOL)
        cost_gap = abs(body.get("total_cost_bdt", float("nan")) - case["expected_output"]["total_cost_bdt"])

        errs = interp_errors + replay_errors
        if cost_gap > TOL:
            errs.append(f"cost gap {cost_gap:.4f} BDT exceeds tolerance vs organizer reference")

        if errs:
            print(f"FAIL {case['id']}:")
            for e in errs:
                print(f"    - {e}")
            failures += 1
        else:
            print(f"PASS {case['id']} (cost={body['total_cost_bdt']:.2f}, gap={cost_gap:.4f})")

    print(f"\n{len(cases) - failures}/{len(cases)} public cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
