from __future__ import annotations

from app.optimizer import HourPlan, SolveResult
from app.summary import build_summary


def _result() -> SolveResult:
    plan = [
        HourPlan(
            hour=hour,
            grid_kwh=float(hour),
            solar_used_kwh=0,
            battery_action="idle",
            battery_kwh=0,
            battery_energy_after_kwh=10,
        )
        for hour in range(24)
    ]
    return SolveResult(plan, sum(item.grid_kwh for item in plan), 123.45, 23)


def test_summary_without_directives():
    summary = build_summary(
        [
            {
                "note_index": 0,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "none",
            }
        ],
        _result(),
    )
    assert "No operator directives" in summary
    assert "hour 23" in summary


def test_summary_lists_unique_applied_directive_types():
    directives = [
        {"directive_type": "no_charge_window"},
        {"directive_type": "no_charge_window"},
        {"directive_type": "max_grid_window"},
    ]
    summary = build_summary(directives, _result())
    assert "Applied 3 operator directive(s)" in summary
    assert summary.count("no_charge_window") == 1
