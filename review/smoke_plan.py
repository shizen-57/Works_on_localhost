"""Offline design audit, not tests of a production API or real language model.

Run: python review/smoke_plan.py --report review/smoke_results.json
Requires NumPy and SciPy. Reads only the explicit public-case JSON path.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / 'participant_docs/BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json'
TOL = 1e-6


def limits(req, directives):
    hours = sorted(req['hours'], key=lambda x: x['hour'])
    b = req['battery']
    solar = np.array([h['solar_kwh'] for h in hours], dtype=float)
    reserve = np.full(24, b['minimum_energy_kwh'], dtype=float)
    caps = np.full(24, np.inf)
    charge = np.full(24, b['max_charge_kwh_per_hour'], dtype=float)
    discharge = np.full(24, b['max_discharge_kwh_per_hour'], dtype=float)
    for d in directives:
        t, a = d['directive_type'], d['structured_adjustment']
        if t == 'no_op':
            continue
        for h in a['hours']:
            if t == 'solar_reduction':
                solar[h] = min(solar[h], hours[h]['solar_kwh'] * a['factor'])
            elif t == 'minimum_battery_reserve':
                reserve[h] = max(reserve[h], a['minimum_energy_kwh'])
            elif t == 'max_grid_window':
                caps[h] = min(caps[h], a['max_grid_kwh'])
            elif t == 'no_charge_window':
                charge[h] = 0
            elif t == 'no_discharge_window':
                discharge[h] = 0
            else:
                raise ValueError(t)
    return hours, solar, reserve, caps, charge, discharge


def solve(req, directives):
    hours, solar, reserve, caps, charge, discharge = limits(req, directives)
    b = req['battery']
    objective = np.zeros(96)
    objective[0::4] = [h['tariff_bdt_per_kwh'] for h in hours]
    bounds = []
    eq, rhs, ub, urhs = [], [], [], []
    for h in range(24):
        bounds.extend([(0, caps[h]), (0, solar[h]), (0, charge[h]), (0, discharge[h])])
        row = np.zeros(96)
        row[4*h:4*h+4] = [1, 1, -1, 1]
        eq.append(row)
        rhs.append(hours[h]['demand_kwh'])
        state = np.zeros(96)
        state[2:4*h+3:4] = 1
        state[3:4*h+4:4] = -1
        ub.extend([state, -state])
        urhs.extend([b['capacity_kwh']-b['initial_energy_kwh'],
                     b['initial_energy_kwh']-reserve[h]])
    eq.append(state)
    rhs.append(0)
    result = linprog(objective, A_eq=eq, b_eq=rhs, A_ub=ub, b_ub=urhs,
                     bounds=bounds, method='highs')
    if not result.success:
        return None
    raw = result.x.reshape(24, 4)
    state = b['initial_energy_kwh']
    plan = []
    for h, (grid, used, ch, dis) in enumerate(raw):
        # Net ALL simultaneous charging/discharging, not just numerical dust.
        net = ch-dis
        state += net
        plan.append(dict(hour=h, grid_kwh=float(grid), solar_used_kwh=float(used),
                         battery_action='charge' if net > 0 else 'discharge' if net < 0 else 'idle',
                         battery_kwh=float(abs(net)), battery_energy_after_kwh=float(state)))
    return dict(scenario_id=req['scenario_id'], directive_interpretation=directives,
                hourly_plan=plan, total_grid_kwh=sum(p['grid_kwh'] for p in plan),
                total_cost_bdt=sum(p['grid_kwh']*hours[p['hour']]['tariff_bdt_per_kwh'] for p in plan),
                peak_grid_kwh=max(p['grid_kwh'] for p in plan),
                plan_summary='Offline design probe'), raw


def replay(req, directives, out, tol=TOL):
    """Independent scalar replay; deliberately does not call limits()."""
    def check(ok, reason):
        if not ok:
            raise ValueError(reason)
    check(out['scenario_id'] == req['scenario_id'], 'scenario_id')
    plan = out['hourly_plan']
    check(len(plan) == 24 and [p['hour'] for p in plan] == list(range(24)), 'hours')
    source = {h['hour']: h for h in req['hours']}
    b = req['battery']
    energy = b['initial_energy_kwh']
    total = cost = peak = 0.0
    for p in plan:
        h = p['hour']
        s = source[h]
        solar, reserve, cap = s['solar_kwh'], b['minimum_energy_kwh'], math.inf
        chlim, dislim = b['max_charge_kwh_per_hour'], b['max_discharge_kwh_per_hour']
        for d in directives:
            a = d['structured_adjustment']
            if a is None or h not in a['hours']:
                continue
            t = d['directive_type']
            if t == 'solar_reduction': solar = min(solar, s['solar_kwh']*a['factor'])
            elif t == 'minimum_battery_reserve': reserve = max(reserve, a['minimum_energy_kwh'])
            elif t == 'max_grid_window': cap = min(cap, a['max_grid_kwh'])
            elif t == 'no_charge_window': chlim = 0
            elif t == 'no_discharge_window': dislim = 0
        for k in ['grid_kwh', 'solar_used_kwh', 'battery_kwh', 'battery_energy_after_kwh']:
            check(type(p[k]) in (int, float) and math.isfinite(p[k]) and p[k] >= -tol, 'numeric '+k)
        action, amount = p['battery_action'], p['battery_kwh']
        check(action in ['charge', 'discharge', 'idle'], 'action')
        ch = amount if action == 'charge' else 0
        dis = amount if action == 'discharge' else 0
        check(action != 'idle' or amount == 0, 'idle')
        check(ch <= chlim+tol and dis <= dislim+tol, 'rate/window')
        energy += ch-dis
        check(abs(energy-p['battery_energy_after_kwh']) <= tol, 'transition')
        check(reserve-tol <= energy <= b['capacity_kwh']+tol, 'reserve/capacity')
        check(p['solar_used_kwh'] <= solar+tol, 'solar')
        check(p['grid_kwh'] <= cap+tol, 'grid cap')
        check(abs(p['grid_kwh']+p['solar_used_kwh']+dis-s['demand_kwh']-ch) <= tol, 'balance')
        total += p['grid_kwh']
        cost += p['grid_kwh']*s['tariff_bdt_per_kwh']
        peak = max(peak, p['grid_kwh'])
    check(abs(energy-b['initial_energy_kwh']) <= tol, 'neutrality')
    for key, value in [('total_grid_kwh', total), ('total_cost_bdt', cost), ('peak_grid_kwh', peak)]:
        check(math.isfinite(out[key]) and abs(out[key]-value) <= tol, key)


def oracle(req, directives):
    """Independent explicit-state MILP with binary charge/discharge exclusivity.

    Constraint compilation uses limits(); independent replay above checks that
    compilation against raw directives. No claim of independent language ground truth.
    """
    hours, solar, reserve, caps, charge, discharge = limits(req, directives)
    b = req['battery']
    # hour blocks: grid, solar, charge, discharge, energy_after, charge_enabled
    obj, lower, upper = np.zeros(144), np.zeros(144), np.zeros(144)
    rows, lows, highs = [], [], []
    integrality = np.zeros(144)
    for h in range(24):
        j = 6*h
        obj[j] = hours[h]['tariff_bdt_per_kwh']
        lower[j+4] = reserve[h]
        upper[j:j+6] = [caps[h], solar[h], charge[h], discharge[h], b['capacity_kwh'], 1]
        integrality[j+5] = 1
        r = np.zeros(144); r[j:j+4] = [1, 1, -1, 1]
        rows.append(r); lows.append(hours[h]['demand_kwh']); highs.append(lows[-1])
        r = np.zeros(144); r[j+4] = 1; r[j+2] = -1; r[j+3] = 1
        if h: r[j-2] = -1
        rows.append(r); lows.append(0 if h else b['initial_energy_kwh']); highs.append(lows[-1])
        r = np.zeros(144); r[j+2] = 1; r[j+5] = -charge[h]
        rows.append(r); lows.append(-np.inf); highs.append(0)
        r = np.zeros(144); r[j+3] = 1; r[j+5] = discharge[h]
        rows.append(r); lows.append(-np.inf); highs.append(discharge[h])
    lower[-2] = upper[-2] = b['initial_energy_kwh']
    result = milp(obj, integrality=integrality, bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(np.array(rows), lows, highs),
                  options={'time_limit': 10, 'mip_rel_gap': 0})
    if not result.success:
        raise AssertionError('Oracle failed: '+result.message)
    return float(result.fun)


def directive(t, hs, **kwargs):
    return dict(note_index=0, applies=True, directive_type=t,
                structured_adjustment=dict(hours=sorted(hs), **kwargs), explanation='synthetic')


def validate_directives(raw, note_count, capacity):
    """Prototype of the revised fail-closed guardrail policy, not production code."""
    shapes = {'solar_reduction': 'factor', 'minimum_battery_reserve': 'minimum_energy_kwh',
              'max_grid_window': 'max_grid_kwh', 'no_charge_window': None, 'no_discharge_window': None}
    if not isinstance(raw, list) or len(raw) != note_count:
        raise ValueError('coverage')
    seen, clean = set(), []
    for original in raw:
        d = copy.deepcopy(original)
        if not isinstance(d, dict) or set(d) != {'note_index', 'applies', 'directive_type',
                                                'structured_adjustment', 'explanation'}:
            raise ValueError('entry shape')
        idx = d['note_index']
        if type(idx) is not int or not 0 <= idx < note_count or idx in seen:
            raise ValueError('note mapping')
        seen.add(idx)
        t, a = d['directive_type'], d['structured_adjustment']
        if not isinstance(t, str) or (t != 'no_op' and t not in shapes):
            raise ValueError('directive type')
        if type(d['applies']) is not bool or not isinstance(d['explanation'], str) or not d['explanation'].strip():
            raise ValueError('field type')
        if t == 'no_op':
            if d['applies'] or a is not None: raise ValueError('no_op semantics')
        else:
            value_key = shapes[t]
            keys = {'hours'} | ({value_key} if value_key else set())
            if not d['applies'] or not isinstance(a, dict) or set(a) != keys:
                raise ValueError('adjustment shape')
            hs = a['hours']
            if not isinstance(hs, list) or not hs or any(type(h) is not int or not 0 <= h < 24 for h in hs):
                raise ValueError('hours')
            a['hours'] = sorted(set(hs))
            if value_key:
                v = a[value_key]
                if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
                    raise ValueError('numeric')
                if t == 'solar_reduction' and v > 1: raise ValueError('factor')
                if t == 'minimum_battery_reserve' and v > capacity: raise ValueError('reserve')
        clean.append(d)
    return sorted(clean, key=lambda d: d['note_index'])


def guardrail_probes(cases):
    valid = copy.deepcopy(cases[0]['expected_output']['directive_interpretation'])
    capacity = cases[0]['input']['battery']['capacity_kwh']
    assert validate_directives(list(reversed(valid)), 2, capacity) == valid
    duplicate_hours = copy.deepcopy(valid)
    duplicate_hours[0]['structured_adjustment']['hours'] = [13, 12, 13]
    assert validate_directives(duplicate_hours, 2, capacity) == valid
    invalid = [None, {}, [], valid[:1], valid+valid[:1]]
    for field, value in [('note_index', 1), ('note_index', True), ('note_index', -1),
                         ('note_index', '0'), ('directive_type', 'change_tariff'),
                         ('applies', False), ('applies', 'true'), ('explanation', '')]:
        bad = copy.deepcopy(valid); bad[0][field] = value; invalid.append(bad)
    for value in [-1, 1.01, True, '0.5', float('nan'), float('inf')]:
        bad = copy.deepcopy(valid); bad[0]['structured_adjustment']['factor'] = value; invalid.append(bad)
    for value in [[], [24], [-1], [True], [12.5], ['12'], None]:
        bad = copy.deepcopy(valid); bad[0]['structured_adjustment']['hours'] = value; invalid.append(bad)
    bad = copy.deepcopy(valid); bad[0]['structured_adjustment']['tariff'] = 0; invalid.append(bad)
    bad = copy.deepcopy(valid); del bad[0]['note_index']; invalid.append(bad)
    bad = copy.deepcopy(valid); bad[1]['structured_adjustment'] = {}; invalid.append(bad)
    invalid.extend([[directive('minimum_battery_reserve', [0], minimum_energy_kwh=capacity+1)],
                    [directive('max_grid_window', [0], max_grid_kwh=-1)]])
    for bad in invalid:
        try: validate_directives(bad, 1 if isinstance(bad, list) and len(bad) == 1 and
                                 bad[0].get('directive_type') in ['minimum_battery_reserve', 'max_grid_window'] else 2,
                                 capacity)
        except ValueError: pass
        else: raise AssertionError('Guardrail accepted malformed output')
    return len(invalid)


def random_case(rng, index):
    # Known feasible witness: idle battery, zero solar use, grid = demand.
    cap = rng.uniform(0, 500) if index % 17 else 0
    reserve = rng.uniform(0, cap)
    initial = rng.uniform(reserve, cap)
    req = dict(scenario_id=f'RANDOM-{index}', operator_notes=['synthetic'],
               battery=dict(capacity_kwh=cap, initial_energy_kwh=initial, minimum_energy_kwh=reserve,
                            max_charge_kwh_per_hour=0 if index % 13 == 0 else rng.uniform(0, 100),
                            max_discharge_kwh_per_hour=0 if index % 11 == 0 else rng.uniform(0, 100)),
               hours=[dict(hour=h, demand_kwh=0 if index % 19 == 0 else rng.uniform(0, 300),
                           solar_kwh=rng.uniform(0, 450),
                           tariff_bdt_per_kwh=0 if index % 7 == 0 else rng.uniform(0, 50)) for h in range(24)])
    ds = []
    for n in range(rng.randint(1, 3)):
        hs = sorted(rng.sample(range(24), rng.randint(1, 24)))
        t = rng.choice(['solar_reduction', 'minimum_battery_reserve', 'no_charge_window',
                        'no_discharge_window', 'max_grid_window'])
        kw = {}
        if t == 'solar_reduction': kw = dict(factor=rng.choice([0, 1, rng.random()]))
        elif t == 'minimum_battery_reserve': kw = dict(minimum_energy_kwh=rng.uniform(reserve, initial))
        elif t == 'max_grid_window': kw = dict(max_grid_kwh=max(req['hours'][h]['demand_kwh'] for h in hs))
        d = directive(t, hs, **kw); d['note_index'] = n; ds.append(d)
    req['operator_notes'] = ['synthetic']*len(ds)
    rng.shuffle(req['hours'])
    return req, ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    cases = json.loads(SAMPLES.read_text(encoding='utf-8-sig'))['cases']
    report = dict(scope='Offline design prototype; no production API, real LLM, Docker or network tested',
                  python=platform.python_version(), scipy=scipy.__version__, numpy=np.__version__,
                  seed=20260918, sample_sha256=hashlib.sha256(SAMPLES.read_bytes()).hexdigest(), public=[])
    timings, noop_failures = [], []
    for c in cases:
        req, expected = c['input'], c['expected_output']
        ds = expected['directive_interpretation']
        replay(req, ds, expected, tol=0.01)
        start = time.perf_counter(); out, raw = solve(req, ds); timings.append(time.perf_counter()-start)
        replay(req, ds, out)
        error = abs(out['total_cost_bdt']-expected['total_cost_bdt'])
        assert error < 0.01, (c['id'], error)
        assert abs(oracle(req, ds)-out['total_cost_bdt']) < 1e-5
        report['public'].append(dict(id=c['id'], cost=out['total_cost_bdt'], gap=error))
        # Probe the original plan's all-no_op fallback against true directives.
        unguarded, _ = solve(req, [])
        try: replay(req, ds, unguarded)
        except ValueError as exc: noop_failures.append(dict(id=c['id'], reason=str(exc)))
    rng = random.Random(report['seed'])
    simultaneous = oracle_checks = 0
    for i in range(300):
        req, ds = random_case(rng, i)
        start = time.perf_counter(); result = solve(req, ds); timings.append(time.perf_counter()-start)
        assert result is not None, i
        out, raw = result
        replay(req, ds, json.loads(json.dumps(out, allow_nan=False)))
        simultaneous += int(np.any(np.minimum(raw[:, 2], raw[:, 3]) > 1e-6))
        if i < 40:
            assert abs(oracle(req, ds)-out['total_cost_bdt']) < 1e-5, i
            oracle_checks += 1
    # Deliberately infeasible: positive demand, no battery, no solar, grid cap zero.
    req = copy.deepcopy(cases[0]['input'])
    req['battery'].update(capacity_kwh=0, initial_energy_kwh=0, minimum_energy_kwh=0,
                          max_charge_kwh_per_hour=0, max_discharge_kwh_per_hour=0)
    req['hours'][0]['solar_kwh'] = 0
    assert solve(req, [directive('max_grid_window', [0], max_grid_kwh=0)]) is None
    # A broad set of output corruptions must be caught by independent replay.
    req = cases[4]['input']; ds = cases[4]['expected_output']['directive_interpretation']
    good, _ = solve(req, ds)
    mutations = []
    for field, value in [('grid_kwh', -1), ('solar_used_kwh', 1e6), ('battery_energy_after_kwh', 1e6),
                         ('battery_kwh', float('nan')), ('battery_action', 'export')]:
        bad = copy.deepcopy(good); bad['hourly_plan'][0][field] = value; mutations.append(bad)
    for field in ['total_grid_kwh', 'total_cost_bdt', 'peak_grid_kwh']:
        bad = copy.deepcopy(good); bad[field] += 1; mutations.append(bad)
    bad = copy.deepcopy(good); bad['hourly_plan'].pop(); mutations.append(bad)
    bad = copy.deepcopy(good); bad['hourly_plan'][1]['hour'] = 0; mutations.append(bad)
    bad = copy.deepcopy(good); bad['scenario_id'] = 'wrong'; mutations.append(bad)
    for bad in mutations:
        try: replay(req, ds, bad)
        except ValueError: pass
        else: raise AssertionError('Replay accepted corrupted output')
    # Rounding each energy to 2 decimals can amplify to >0.01 BDT total error.
    exact_cost = 24 * 0.0049 * 30
    rounded_cost = 24 * round(0.0049, 2) * 30
    assert abs(exact_cost-rounded_cost) > 0.01
    # Reassigning indices by array position changes meaning when output is reordered.
    reordered = [dict(note_index=1, directive_type='no_charge_window'),
                 dict(note_index=0, directive_type='no_discharge_window')]
    assert sorted(reordered, key=lambda x:x['note_index'])[0] != reordered[0]
    report.update(random_feasible_cases=300, independent_milp_comparisons=10+oracle_checks,
                  malformed_directive_rejections=guardrail_probes(cases),
                  nontrivial_simultaneous_flow_cases=simultaneous,
                  rejected_output_mutations=len(mutations), infeasible_case_detected=True,
                  noop_schedule_violations=noop_failures, noop_semantic_failures=len(cases),
                  two_decimal_cost_error_witness_bdt=abs(exact_cost-rounded_cost),
                  lp_timing_ms=dict(first=timings[0]*1000, p95=float(np.percentile(timings,95))*1000,
                                    max=max(timings)*1000), status='PASS')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
