"""SleepyAI provider using its documented OpenAI-compatible API.

The provider is deliberately transport-only. Prompt construction lives in
``app.llm_interpreter`` and semantic validation lives in ``app.guardrails``.
Model output is parsed as strict JSON: duplicate object keys and non-finite
numbers are rejected before guardrails see the data.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from app.llm_client import LLMOutputError, LLMTransientError
from app.llm_interpreter import SYSTEM_PROMPT, build_user_content
from app.schemas import BatteryConfig

_JSON_ENVELOPE_INSTRUCTION = """

## Output format (STRICT)

Respond with ONLY a single JSON object of this exact shape, and nothing else:

{"directives": [{"note_index": 0, "applies": true, "directive_type": "...", \
"structured_adjustment": {...} or null, "explanation": "..."}, ...]}
"""

_CODE_FENCE_RE = re.compile(r"\A```(?:json)?\s*(.*?)\s*```\Z", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class SleepyAIUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"non-finite numeric literal: {token}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse one JSON object, optionally wrapped in one Markdown fence."""
    stripped = text.strip()
    fenced = _CODE_FENCE_RE.fullmatch(stripped)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        parsed = json.loads(
            stripped,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise LLMOutputError(f"model output is not strict JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMOutputError("model output must be one JSON object")
    return parsed


def _usage_from_response(payload: dict[str, Any]) -> SleepyAIUsage:
    raw = payload.get("usage")
    if not isinstance(raw, dict):
        return SleepyAIUsage()

    def token_count(name: str) -> int:
        value = raw.get(name, 0)
        return value if type(value) is int and value >= 0 else 0

    prompt = token_count("prompt_tokens")
    completion = token_count("completion_tokens")
    total = token_count("total_tokens") or prompt + completion
    return SleepyAIUsage(prompt, completion, total)


class SleepyAILLMClient:
    """LLM client for ``POST /chat/completions`` on SleepyAI."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout_s: float = 15.0,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        usage_observer: Callable[[SleepyAIUsage], None] | None = None,
        request_observer: Callable[[], None] | None = None,
    ) -> None:
        self._model = model
        self._usage_observer = usage_observer
        self._request_observer = request_observer
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def interpret(
        self,
        notes: list[str],
        battery: BatteryConfig,
        correction_feedback: str | None = None,
    ) -> list[dict]:
        directives, _ = await self.interpret_with_usage(notes, battery, correction_feedback=correction_feedback)
        return directives

    async def interpret_with_usage(
        self,
        notes: list[str],
        battery: BatteryConfig,
        correction_feedback: str | None = None,
    ) -> tuple[list[dict], SleepyAIUsage]:
        user_content = build_user_content(notes, battery)
        if correction_feedback:
            user_content += (
                "\n\nYour previous response was rejected by validation for this "
                f"reason: {correction_feedback}\n"
                "Correct only that problem and resubmit a complete, valid response "
                "covering every note_index."
            )

        try:
            if self._request_observer is not None:
                self._request_observer()
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": self._model,
                    "stream": False,
                    "max_tokens": 2048,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT + _JSON_ENVELOPE_INSTRUCTION},
                        {"role": "user", "content": user_content},
                    ],
                },
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise LLMTransientError(f"provider_transport_error: {type(exc).__name__}") from exc

        if response.status_code in {429, 502, 503}:
            raise LLMTransientError(f"provider_transient_status: {response.status_code}")
        if response.status_code in {401, 403}:
            raise LLMOutputError(f"provider_access_denied: {response.status_code}")
        if response.is_error:
            raise LLMOutputError(f"provider_rejected_request: {response.status_code}")

        try:
            payload = response.json(
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMOutputError(f"provider response is not strict JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise LLMOutputError("provider response must be a JSON object")

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LLMOutputError("provider response has no completion choice")
        choice = choices[0]
        if choice.get("finish_reason") in {"content_filter", "refusal"}:
            raise LLMOutputError("model_refused")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise LLMOutputError("provider response choice has no message")
        if message.get("refusal"):
            raise LLMOutputError("model_refused")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMOutputError("provider response has no text content")

        envelope = _extract_json_object(content)
        directives = envelope.get("directives")
        if not isinstance(directives, list) or any(not isinstance(item, dict) for item in directives):
            raise LLMOutputError("JSON envelope must contain a directives list of objects")
        usage = _usage_from_response(payload)
        if self._usage_observer is not None:
            self._usage_observer(usage)
        return directives, usage
