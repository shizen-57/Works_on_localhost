"""Deterministic plan_summary text -- no LLM call.

Problem Statement Sec. 10.1 lists plan_summary as a required field, but
using an LLM only to write it does not satisfy the mandatory LLM
requirement (Sec. 02, Participant Guide Sec. 04 -- "AI used only for
plan_summary ... does not satisfy the LLM requirement"). Keeping this
template-based and separate from app.llm_interpreter removes any ambiguity
about which call does the graded interpretation work, and avoids spending
request-deadline budget on a second model call for cosmetic text.
"""

from __future__ import annotations

from app.optimizer import SolveResult


def build_summary(directives: list[dict], result: SolveResult) -> str:
    applied = [d for d in directives if d["directive_type"] != "no_op"]
    peak_hour = max(result.hourly_plan, key=lambda p: p.grid_kwh).hour

    if applied:
        types = ", ".join(dict.fromkeys(d["directive_type"] for d in applied))
        directive_clause = f"Applied {len(applied)} operator directive(s) ({types})."
    else:
        directive_clause = "No operator directives affected this schedule."

    return (
        f"{directive_clause} Total grid cost {result.total_cost_bdt:.2f} BDT "
        f"({result.total_grid_kwh:.2f} kWh) over 24 hours, peak grid draw "
        f"{result.peak_grid_kwh:.2f} kWh at hour {peak_hour}."
    )
