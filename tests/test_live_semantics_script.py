from __future__ import annotations

from scripts.eval_live_semantics import (
    _expected_directive,
    _expected_directives,
    _interpretation_errors,
    _request_for,
)


def test_live_semantic_request_is_a_complete_feasible_shape():
    case = {
        "id": "SEM-X",
        "note": "Do not charge from 2 PM to 4 PM.",
        "expected": {
            "directive_type": "no_charge_window",
            "applies": True,
            "hours": [14, 15],
            "value_key": None,
            "value": None,
        },
    }
    battery = {
        "capacity_kwh": 200.0,
        "initial_energy_kwh": 100.0,
        "minimum_energy_kwh": 30.0,
        "max_charge_kwh_per_hour": 50.0,
        "max_discharge_kwh_per_hour": 50.0,
    }
    request = _request_for(case, battery)
    assert request["scenario_id"] == "LIVE-SEM-X"
    assert len(request["hours"]) == 24
    assert [hour["hour"] for hour in request["hours"]] == list(range(24))


def test_live_semantic_comparison_detects_wrong_numeric_value():
    case = {
        "id": "SEM-X",
        "note": "Solar is reduced by 80% from 1 PM to 3 PM.",
        "expected": {
            "directive_type": "solar_reduction",
            "applies": True,
            "hours": [13, 14],
            "value_key": "factor",
            "value": 0.2,
        },
    }
    directive = _expected_directive(case)
    response = {"directive_interpretation": [directive]}
    assert _interpretation_errors(case, response) == []

    response["directive_interpretation"][0]["structured_adjustment"]["factor"] = 0.8
    assert any("factor" in error for error in _interpretation_errors(case, response))


def test_mixed_note_bundle_preserves_note_index_mapping():
    case = {
        "id": "BUNDLE-X",
        "notes": ["Do not charge from 2 PM to 4 PM.", "The library closes next week."],
        "expected": [
            {
                "directive_type": "no_charge_window",
                "applies": True,
                "hours": [14, 15],
                "value_key": None,
                "value": None,
            },
            {"directive_type": "no_op", "applies": False, "hours": None, "value_key": None, "value": None},
        ],
    }
    directives = _expected_directives(case)
    assert [directive["note_index"] for directive in directives] == [0, 1]
    assert _interpretation_errors(case, {"directive_interpretation": directives}) == []
