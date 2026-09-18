from __future__ import annotations

import json
from pathlib import Path

import pytest

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "public_sample_cases.json"


@pytest.fixture(scope="session")
def public_cases() -> list[dict]:
    return json.loads(DATA_PATH.read_text())["cases"]


@pytest.fixture(autouse=True)
def _placeholder_env(monkeypatch):
    """Default env for every test: valid, fail-closed-compliant placeholder
    config. Individual tests override specific vars as needed."""
    monkeypatch.setenv("LLM_PROVIDER", "placeholder")
    monkeypatch.setenv("ALLOW_STUB_INTERPRETER", "true")
    monkeypatch.setenv("REQUEST_DEADLINE_S", "25")
    yield
