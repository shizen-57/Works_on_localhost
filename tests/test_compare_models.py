from __future__ import annotations

from scripts.compare_models import HIGH_RISK_IDS, _rank_key


def _result(model: str, *, eligible: bool, accuracy: float, failures: int, p95: float, cost: float | None) -> dict:
    return {
        "model": model,
        "eligible": eligible,
        "exact_accuracy": accuracy,
        "call_failures": failures,
        "latency_s": {"p95": p95},
        "usage": {"estimated_cost": cost},
    }


def test_high_risk_subset_is_frozen_at_24_unique_cases():
    assert len(HIGH_RISK_IDS) == 24


def test_cost_breaks_tie_only_after_all_mandatory_gates_pass():
    fast_but_expensive = _result("fast", eligible=True, accuracy=1, failures=0, p95=1, cost=2)
    slower_but_cheaper = _result("cheap", eligible=True, accuracy=1, failures=0, p95=4.9, cost=1)
    inaccurate = _result("wrong", eligible=False, accuracy=0.99, failures=0, p95=0.1, cost=0)
    ranked = sorted([fast_but_expensive, slower_but_cheaper, inaccurate], key=_rank_key)
    assert [item["model"] for item in ranked] == ["cheap", "fast", "wrong"]
