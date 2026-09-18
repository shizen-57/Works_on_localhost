from __future__ import annotations

import copy
import math

import pytest

from app.guardrails import GuardrailViolation, validate_directives


def _valid(public_cases):
    return copy.deepcopy(public_cases[0]["expected_output"]["directive_interpretation"])


def _capacity(public_cases):
    return public_cases[0]["input"]["battery"]["capacity_kwh"]


def test_all_public_cases_validate_cleanly(public_cases):
    for case in public_cases:
        directives = case["expected_output"]["directive_interpretation"]
        capacity = case["input"]["battery"]["capacity_kwh"]
        clean = validate_directives(directives, len(directives), capacity)
        assert clean == sorted(directives, key=lambda d: d["note_index"])


def test_reordered_entries_are_resorted_not_rejected(public_cases):
    valid = _valid(public_cases)
    reordered = list(reversed(valid))
    result = validate_directives(reordered, 2, _capacity(public_cases))
    assert result == valid


def test_duplicate_hours_are_deduped_and_sorted(public_cases):
    valid = _valid(public_cases)
    dup = copy.deepcopy(valid)
    dup[0]["structured_adjustment"]["hours"] = [13, 12, 13]
    result = validate_directives(dup, 2, _capacity(public_cases))
    assert result[0]["structured_adjustment"]["hours"] == [12, 13]


@pytest.mark.parametrize("bad", [None, {}, []])
def test_non_list_or_wrong_length_rejected(public_cases, bad):
    with pytest.raises(GuardrailViolation):
        validate_directives(bad, 2, _capacity(public_cases))


def test_missing_and_extra_entries_rejected(public_cases):
    valid = _valid(public_cases)
    with pytest.raises(GuardrailViolation):
        validate_directives(valid[:1], 2, _capacity(public_cases))
    with pytest.raises(GuardrailViolation):
        validate_directives(valid + valid[:1], 2, _capacity(public_cases))


@pytest.mark.parametrize("field,value", [
    ("note_index", 1), ("note_index", True), ("note_index", -1), ("note_index", "0"),
    ("directive_type", "change_tariff"), ("applies", False), ("applies", "true"),
    ("explanation", ""),
])
def test_field_level_violations_rejected(public_cases, field, value):
    valid = _valid(public_cases)
    bad = copy.deepcopy(valid)
    bad[0][field] = value
    with pytest.raises(GuardrailViolation):
        validate_directives(bad, 2, _capacity(public_cases))


@pytest.mark.parametrize("value", [-1, 1.01, True, "0.5", float("nan"), float("inf")])
def test_invalid_factor_rejected(public_cases, value):
    valid = _valid(public_cases)
    bad = copy.deepcopy(valid)
    bad[0]["structured_adjustment"]["factor"] = value
    with pytest.raises(GuardrailViolation):
        validate_directives(bad, 2, _capacity(public_cases))


@pytest.mark.parametrize("value", [[], [24], [-1], [True], [12.5], ["12"], None])
def test_invalid_hours_rejected(public_cases, value):
    valid = _valid(public_cases)
    bad = copy.deepcopy(valid)
    bad[0]["structured_adjustment"]["hours"] = value
    with pytest.raises(GuardrailViolation):
        validate_directives(bad, 2, _capacity(public_cases))


def test_extra_adjustment_key_rejected(public_cases):
    valid = _valid(public_cases)
    bad = copy.deepcopy(valid)
    bad[0]["structured_adjustment"]["tariff"] = 0
    with pytest.raises(GuardrailViolation):
        validate_directives(bad, 2, _capacity(public_cases))


def test_missing_note_index_key_rejected(public_cases):
    valid = _valid(public_cases)
    bad = copy.deepcopy(valid)
    del bad[0]["note_index"]
    with pytest.raises(GuardrailViolation):
        validate_directives(bad, 2, _capacity(public_cases))


def test_empty_adjustment_on_non_no_op_rejected(public_cases):
    valid = _valid(public_cases)
    bad = copy.deepcopy(valid)
    bad[1]["structured_adjustment"] = {}
    with pytest.raises(GuardrailViolation):
        validate_directives(bad, 2, _capacity(public_cases))


def test_reserve_exceeding_capacity_rejected(public_cases):
    capacity = _capacity(public_cases)
    bad = [dict(note_index=0, applies=True, directive_type="minimum_battery_reserve",
                 structured_adjustment=dict(hours=[0], minimum_energy_kwh=capacity + 1),
                 explanation="x")]
    with pytest.raises(GuardrailViolation):
        validate_directives(bad, 1, capacity)


def test_negative_grid_cap_rejected(public_cases):
    bad = [dict(note_index=0, applies=True, directive_type="max_grid_window",
                 structured_adjustment=dict(hours=[0], max_grid_kwh=-1),
                 explanation="x")]
    with pytest.raises(GuardrailViolation):
        validate_directives(bad, 1, _capacity(public_cases))


def test_no_op_with_applies_true_rejected(public_cases):
    valid = _valid(public_cases)
    no_op_indices = [i for i, d in enumerate(valid) if d["directive_type"] == "no_op"]
    assert no_op_indices, "fixture SAMPLE-01 must contain a no_op entry"
    bad = copy.deepcopy(valid)
    bad[no_op_indices[0]]["applies"] = True
    with pytest.raises(GuardrailViolation):
        validate_directives(bad, 2, _capacity(public_cases))


def test_unsupported_type_never_coerced_to_no_op(public_cases):
    """Regression guard for the corrected design (plan_review.md Sec. 2):
    an unsupported directive_type must be REJECTED outright, never
    silently turned into a no_op that would drop a real constraint."""
    bad = [dict(note_index=0, applies=True, directive_type="change_tariff",
                 structured_adjustment=dict(hours=[0]), explanation="x")]
    with pytest.raises(GuardrailViolation):
        validate_directives(bad, 1, _capacity(public_cases))


def test_note_index_never_repaired_by_array_position(public_cases):
    """Regression guard: two entries both claiming note_index=0 (with 1
    missing) must be rejected, not silently remapped by list position."""
    bad = [
        dict(note_index=0, applies=False, directive_type="no_op", structured_adjustment=None, explanation="a"),
        dict(note_index=0, applies=False, directive_type="no_op", structured_adjustment=None, explanation="b"),
    ]
    with pytest.raises(GuardrailViolation):
        validate_directives(bad, 2, _capacity(public_cases))
