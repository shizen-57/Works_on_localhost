"""Reseller/aggregator backend: SleepyAI (sleepyai.org), an Anthropic-
Messages-API-compatible proxy in front of many third-party models
(DeepSeek, GLM, Grok, Gemini, GPT, plus several free-tier models).

Unlike app.providers.anthropic_provider, this does NOT use
client.messages.parse() / output_config.format: confirmed live that the
proxy accepts the parameter without error but does not actually enforce it
for at least some free models -- the underlying model just replies in
plain prose and the SDK's schema validation then fails. So this adapter
asks for a JSON envelope by instruction instead, and parses the response
text defensively (stripping markdown code fences, extracting the first
top-level JSON object). This is not a weaker safety story than the
structured-output path: app.guardrails treats ALL LLM output as untrusted
regardless of which provider produced it, so a proxy that can't guarantee
schema-valid JSON is exactly the case that module exists for.

No `temperature` is passed for the same reason as app.providers.
anthropic_provider: the installed SDK's `messages.create` no longer accepts
it as a keyword for current-generation models (TypeError, confirmed live
against this proxy before any network call was even made).
"""
from __future__ import annotations

import json
import re
from typing import Any

import anthropic

from app.llm_client import LLMClientError, LLMOutputError, LLMTransientError
from app.llm_interpreter import SYSTEM_PROMPT, build_user_content
from app.schemas import BatteryConfig

_JSON_ENVELOPE_INSTRUCTION = """

## Output format (STRICT)

Respond with ONLY a single JSON object of this exact shape, and nothing \
else -- no markdown code fences, no commentary before or after it:

{"directives": [{"note_index": 0, "applies": true, "directive_type": "...", \
"structured_adjustment": {...} or null, "explanation": "..."}, ...]}
"""

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = _CODE_FENCE_RE.sub("", text).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Fallback: take the substring between the first '{' and the last '}'.
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMOutputError(f"no JSON object found in model output: {text[:200]!r}")
    try:
        return json.loads(stripped[start:end + 1])
    except json.JSONDecodeError as exc:
        raise LLMOutputError(f"model output is not valid JSON: {exc}") from exc


class SleepyAILLMClient:
    """Implements app.llm_client.LLMClient against SleepyAI's
    Anthropic-compatible endpoint via the Anthropic SDK's base_url override."""

    def __init__(self, api_key: str, model: str, base_url: str, timeout_s: float = 15.0) -> None:
        self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url, max_retries=0)
        self._model = model
        self._timeout_s = timeout_s

    async def interpret(
        self,
        notes: list[str],
        battery: BatteryConfig,
        correction_feedback: str | None = None,
    ) -> list[dict]:
        # The proxy's SDK usage here is synchronous (anthropic.Anthropic,
        # not AsyncAnthropic) -- run it off the event loop so a slow free
        # model can't block other concurrent requests.
        import asyncio

        return await asyncio.to_thread(self._interpret_sync, notes, battery, correction_feedback)

    def _interpret_sync(
        self, notes: list[str], battery: BatteryConfig, correction_feedback: str | None
    ) -> list[dict]:
        user_content = build_user_content(notes, battery)
        if correction_feedback:
            user_content += (
                "\n\nYour previous response was rejected by validation for this "
                f"reason: {correction_feedback}\n"
                "Correct only that problem and resubmit a complete, valid response "
                "covering every note_index."
            )

        try:
            response = self._client.with_options(timeout=self._timeout_s).messages.create(
                model=self._model,
                max_tokens=2048,
                system=SYSTEM_PROMPT + _JSON_ENVELOPE_INSTRUCTION,
                messages=[{"role": "user", "content": user_content}],
            )
        except anthropic.RateLimitError as exc:
            raise LLMTransientError(f"rate_limited: {exc}") from exc
        except anthropic.APITimeoutError as exc:
            raise LLMTransientError(f"timeout: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMTransientError(f"connection_error: {exc}") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise LLMTransientError(f"provider_5xx: {exc.status_code}") from exc
            raise LLMOutputError(f"provider_rejected_request: {exc.status_code}") from exc
        except Exception as exc:  # noqa: BLE001 -- last-resort safe failure
            raise LLMClientError(f"unexpected_provider_error: {exc}") from exc

        if response.stop_reason == "refusal":
            raise LLMOutputError("model_refused")

        text_blocks = [b.text for b in response.content if b.type == "text"]
        if not text_blocks:
            raise LLMOutputError(f"no text content in response: stop_reason={response.stop_reason}")

        envelope = _extract_json_object("".join(text_blocks))
        directives = envelope.get("directives")
        if not isinstance(directives, list):
            raise LLMOutputError(f"JSON envelope missing a 'directives' list: {envelope!r}")
        return directives
