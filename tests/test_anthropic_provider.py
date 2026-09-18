from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.llm_client import LLMClientError, LLMOutputError
from app.providers.anthropic_provider import AnthropicLLMClient
from app.schemas import BatteryConfig


class FakeAnthropicClient:
    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.messages = self
        self.kwargs = None

    def with_options(self, **kwargs):
        return self

    async def parse(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def _battery() -> BatteryConfig:
    return BatteryConfig(
        capacity_kwh=100.0,
        initial_energy_kwh=50.0,
        minimum_energy_kwh=10.0,
        max_charge_kwh_per_hour=20.0,
        max_discharge_kwh_per_hour=20.0,
    )


def _client(fake: FakeAnthropicClient) -> AnthropicLLMClient:
    client = AnthropicLLMClient.__new__(AnthropicLLMClient)
    client._client = fake
    client._model = "chosen-model"
    client._timeout_s = 1.0
    return client


@pytest.mark.asyncio
async def test_anthropic_success_uses_selected_model():
    directive = {
        "note_index": 0,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": "irrelevant",
    }
    parsed = SimpleNamespace(directives=[SimpleNamespace(model_dump=lambda: directive)])
    fake = FakeAnthropicClient(SimpleNamespace(stop_reason="end_turn", parsed_output=parsed))
    result = await _client(fake).interpret(["note"], _battery())
    assert result == [directive]
    assert fake.kwargs["model"] == "chosen-model"


@pytest.mark.asyncio
async def test_anthropic_refusal_is_controlled():
    fake = FakeAnthropicClient(SimpleNamespace(stop_reason="refusal", parsed_output=None))
    with pytest.raises(LLMOutputError, match="refused"):
        await _client(fake).interpret(["note"], _battery())


@pytest.mark.asyncio
async def test_anthropic_missing_parsed_output_is_controlled():
    fake = FakeAnthropicClient(SimpleNamespace(stop_reason="end_turn", parsed_output=None))
    with pytest.raises(LLMOutputError, match="no_parsed_output"):
        await _client(fake).interpret(["note"], _battery())


@pytest.mark.asyncio
async def test_anthropic_unexpected_error_is_sanitized():
    fake = FakeAnthropicClient(error=RuntimeError("sdk exploded"))
    with pytest.raises(LLMClientError, match="unexpected_provider_error"):
        await _client(fake).interpret(["note"], _battery())
