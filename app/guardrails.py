"""Fail-closed validation of raw LLM output (Problem Statement Sec. 08).

LLM output is untrusted structured data until it passes here. This module
draws a hard line the earlier design got wrong: it repairs nothing that
could change *meaning*. It is safe to sort a list by an already-valid,
already-unique note_index, and safe to sort/dedupe an already-valid set of
hour integers -- neither changes what the directive means. Everything else
that's wrong is rejected outright:

  - An unsupported directive_type is REJECTED, never silently turned into
    no_op. Coercing "the LLM said something we don't recognize" into
    no_op manufactures a confident, wrong success: the schedule then
    silently ignores whatever the operator actually asked for.
  - note_index is trusted only as a value, never as array position. If the
    model reorders or duplicates entries, position-based repair would
    misattribute one note's directive to another note entirely.

See app/llm_interpreter.py for the bounded-retry policy this feeds into,
and review/plan_review.md Sec. "unsafe repair can produce believable wrong
success" for why this module is shaped this way.
"""
from __future__ import annotations

import math

from app.schemas import DIRECTIVE_TYPES

_ADJUSTMENT_VALUE_KEY = {
    "solar_reduction": "factor",
    "minimum_battery_reserve": "minimum_energy_kwh",
    "max_grid_window": "max_grid_kwh",
    "no_charge_window": None,
    "no_discharge_window": None,
}

_ENTRY_KEYS = {"note_index", "applies", "directive_type", "structured_adjustment", "explanation"}


class GuardrailViolation(ValueError):
    """Raised with a short machine-stable reason code as the message.

    The reason is safe to log and safe to feed back into a single bounded
    LLM correction attempt (app/llm_interpreter.py) -- it never contains
    raw model output or request data, just what rule failed.
    """


def _is_int(value: object) -> bool:
    return type(value) is int  # excludes bool, which is an int subclass


def _is_number(value: object) -> bool:
    return type(value) in (int, float)  # excludes bool


def validate_directives(raw: object, note_count: int, battery_capacity: float) -> list[dict]:
    """Validate and minimally normalize raw LLM output.

    Returns a list of clean directive dicts sorted by note_index, covering
    every index 0..note_count-1 exactly once. Raises GuardrailViolation on
    the first rule violated -- callers never receive a partially-repaired
    result.
    """
    if not isinstance(raw, list) or len(raw) != note_count:
        raise GuardrailViolation("coverage: expected one entry per operator note")

    seen_indices: set[int] = set()
    clean: list[dict] = []

    for entry in raw:
        if not isinstance(entry, dict) or set(entry) != _ENTRY_KEYS:
            raise GuardrailViolation("entry_shape: unexpected fields on a directive_interpretation entry")

        idx = entry["note_index"]
        if not _is_int(idx) or not (0 <= idx < note_count) or idx in seen_indices:
            raise GuardrailViolation("note_index: missing, duplicate, or out-of-range mapping")
        seen_indices.add(idx)

        dtype = entry["directive_type"]
        if not isinstance(dtype, str) or dtype not in DIRECTIVE_TYPES:
            raise GuardrailViolation(f"directive_type: unsupported value {dtype!r}")

        applies = entry["applies"]
        if not isinstance(applies, bool):
            raise GuardrailViolation("applies: must be a boolean")

        explanation = entry["explanation"]
        if not isinstance(explanation, str) or not explanation.strip():
            raise GuardrailViolation("explanation: must be a non-empty string")

        adjustment = entry["structured_adjustment"]

        if dtype == "no_op":
            if applies is not False or adjustment is not None:
                raise GuardrailViolation("no_op: applies must be false and structured_adjustment must be null")
            clean.append(dict(entry))
            continue

        if applies is not True:
            raise GuardrailViolation(f"{dtype}: applies must be true for a non-no_op directive")
        if not isinstance(adjustment, dict):
            raise GuardrailViolation(f"{dtype}: structured_adjustment must be an object")

        value_key = _ADJUSTMENT_VALUE_KEY[dtype]
        expected_keys = {"hours"} | ({value_key} if value_key else set())
        if set(adjustment) != expected_keys:
            raise GuardrailViolation(f"{dtype}: structured_adjustment has the wrong fields")

        hours = adjustment["hours"]
        if (
            not isinstance(hours, list) or not hours
            or any(not _is_int(h) or not (0 <= h < 24) for h in hours)
        ):
            raise GuardrailViolation(f"{dtype}: hours must be a non-empty list of integers 0..23")
        # Safe normalization only: sort + dedupe a set of already-valid hours.
        normalized_hours = sorted(set(hours))

        normalized_adjustment: dict = {"hours": normalized_hours}

        if value_key:
            value = adjustment[value_key]
            if not _is_number(value) or not math.isfinite(value) or value < 0:
                raise GuardrailViolation(f"{dtype}: {value_key} must be a finite non-negative number")
            if dtype == "solar_reduction" and value > 1:
                raise GuardrailViolation("solar_reduction: factor must be between 0 and 1 inclusive")
            if dtype == "minimum_battery_reserve" and value > battery_capacity:
                raise GuardrailViolation("minimum_battery_reserve: minimum_energy_kwh exceeds battery capacity")
            normalized_adjustment[value_key] = value

        clean.append(dict(
            note_index=idx,
            applies=True,
            directive_type=dtype,
            structured_adjustment=normalized_adjustment,
            explanation=explanation,
        ))

    if seen_indices != set(range(note_count)):
        raise GuardrailViolation("coverage: note_index values do not cover every operator note exactly once")

    # Safe normalization: sort entries by their already-validated, already-unique
    # note_index. This never changes which directive maps to which note.
    return sorted(clean, key=lambda d: d["note_index"])
