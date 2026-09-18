from __future__ import annotations

import json

import httpx
import pytest

from app.llm_client import LLMOutputError, LLMTransientError
from app.providers.sleepyai_catalog import discover_models
from app.providers.sleepyai_provider import SleepyAILLMClient, _extract_json_object
from app.schemas import BatteryConfig


@pytest.fixture()
def battery() -> BatteryConfig:
    return BatteryConfig(
        capacity_kwh=100.0,
        initial_energy_kwh=50.0,
        minimum_energy_kwh=10.0,
        max_charge_kwh_per_hour=20.0,
        max_discharge_kwh_per_hour=20.0,
    )


def _directive_content() -> str:
    return json.dumps(
        {
            "directives": [
                {
                    "note_index": 0,
                    "applies": False,
                    "directive_type": "no_op",
                    "structured_adjustment": None,
                    "explanation": "irrelevant",
                }
            ]
        }
    )


@pytest.mark.asyncio
async def test_openai_transport_sends_selected_model_and_auth(battery):
    observed = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        observed["path"] = request.url.path
        observed["auth"] = request.headers.get("authorization")
        observed["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": _directive_content()}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
        )

    client = SleepyAILLMClient(
        "secret-test-key",
        "chosen-model",
        "https://example.test/api/v1",
        transport=httpx.MockTransport(handler),
    )
    try:
        directives, usage = await client.interpret_with_usage(["hello"], battery)
    finally:
        await client.aclose()
    assert directives[0]["directive_type"] == "no_op"
    assert usage.total_tokens == 120
    assert observed["path"] == "/api/v1/chat/completions"
    assert observed["auth"] == "Bearer secret-test-key"
    assert observed["body"]["model"] == "chosen-model"
    assert observed["body"]["stream"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 502, 503])
async def test_transient_status_mapping(status, battery):
    transport = httpx.MockTransport(lambda request: httpx.Response(status))
    client = SleepyAILLMClient("key", "model", "https://example.test", transport=transport)
    try:
        with pytest.raises(LLMTransientError):
            await client.interpret(["note"], battery)
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 404, 500])
async def test_nontransient_status_mapping(status, battery):
    transport = httpx.MockTransport(lambda request: httpx.Response(status))
    client = SleepyAILLMClient("key", "model", "https://example.test", transport=transport)
    try:
        with pytest.raises(LLMOutputError):
            await client.interpret(["note"], battery)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_network_error_is_transient(battery):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    client = SleepyAILLMClient("key", "model", "https://example.test", transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(LLMTransientError):
            await client.interpret(["note"], battery)
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    "text",
    [
        '{"directives": [], "directives": []}',
        '{"directives": [], "x": NaN}',
        'prefix {"directives": []}',
        '{"directives": []} suffix',
        '[{"directives": []}]',
    ],
)
def test_strict_model_json_rejects_ambiguous_or_nonfinite_output(text):
    with pytest.raises(LLMOutputError):
        _extract_json_object(text)


def test_single_json_markdown_fence_is_tolerated():
    assert _extract_json_object('```json\n{"directives": []}\n```') == {"directives": []}


@pytest.mark.asyncio
async def test_model_refusal_is_rejected(battery):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={"choices": [{"finish_reason": "content_filter", "message": {"content": ""}}]}
        )
    )
    client = SleepyAILLMClient("key", "model", "https://example.test", transport=transport)
    try:
        with pytest.raises(LLMOutputError, match="refused"):
            await client.interpret(["note"], battery)
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": None}]},
        {"choices": [{"message": {"content": ""}}]},
        {"choices": [{"message": {"content": "{}"}}]},
        {"choices": [{"message": {"content": '{"directives":[1]}'}}]},
    ],
)
async def test_malformed_provider_payloads_are_controlled(payload, battery):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    client = SleepyAILLMClient("key", "model", "https://example.test", transport=transport)
    try:
        with pytest.raises(LLMOutputError):
            await client.interpret(["note"], battery)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_duplicate_keys_in_outer_provider_response_are_rejected(battery):
    raw = b'{"choices":[],"choices":[]}'
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=raw))
    client = SleepyAILLMClient("key", "model", "https://example.test", transport=transport)
    try:
        with pytest.raises(LLMOutputError, match="strict JSON"):
            await client.interpret(["note"], battery)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_catalog_is_sanitized_and_sorted():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "z-model", "inputPrice": 1, "private": "drop-me"},
                    {"id": "a-model", "contextWindow": 1000},
                    {"bad": "missing-id"},
                ]
            },
        )

    models = await discover_models("secret", "https://example.test/api/v1", transport=httpx.MockTransport(handler))
    assert [model["id"] for model in models] == ["a-model", "z-model"]
    assert "private" not in models[1]


@pytest.mark.asyncio
async def test_catalog_accepts_top_level_list_and_rejects_bad_shape():
    list_transport = httpx.MockTransport(lambda request: httpx.Response(200, json=[{"id": "model"}]))
    assert await discover_models("key", "https://example.test", transport=list_transport) == [{"id": "model"}]

    bad_transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"data": {}}))
    with pytest.raises(ValueError, match="model list"):
        await discover_models("key", "https://example.test", transport=bad_transport)
