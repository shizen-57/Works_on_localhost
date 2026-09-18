"""Real interpretation backend: Anthropic Messages API structured output.

Owns authentication, transport, and provider-specific response parsing
only. Prompt content lives in app.llm_interpreter (SYSTEM_PROMPT,
build_user_content) so it is defined in one place and shared with whatever
provider is active.

Uses client.messages.parse(..., output_format=<pydantic model>) so the API
itself enforces schema-valid JSON -- app.guardrails then does the semantic
validation (allowed enum values, hour ranges, applies semantics, etc.) that
a JSON schema alone can't express.

No `temperature` is passed: sampling parameters (temperature/top_p/top_k)
have been removed from the Messages API request shape for current-generation
models -- the installed SDK's `messages.create`/`.parse` no longer even
accept the keyword (confirmed: passing it raises TypeError before any
network call). Determinism instead comes from the structured-output schema
plus app.guardrails' bounded-retry policy, not from temperature=0.

UNTESTED AGAINST A LIVE KEY as of this commit -- no provider key was
available during development (see plan's "Blocking open item"). The request
shape and exception mapping follow the Anthropic Python SDK 1.6.0
docs/skill reference exactly; before submission this must be run against
real scenarios per the plan's "Real model (cannot be skipped)" test gate.
"""

from __future__ import annotations

from typing import Any

import anthropic
from pydantic import BaseModel

from app.llm_client import LLMClientError, LLMOutputError, LLMTransientError
from app.llm_interpreter import SYSTEM_PROMPT, build_user_content
from app.schemas import BatteryConfig


# Loose on purpose: structured_adjustment's exact required keys differ by
# directive_type (Problem Statement Sec. 04), so we accept any object here
# and let app.guardrails enforce the per-type shape. Tightening this further
# would mean re-deriving the same conditional schema guardrails already owns.
class _RawDirectiveOut(BaseModel):
    note_index: int
    applies: bool
    directive_type: str
    structured_adjustment: dict[str, Any] | None
    explanation: str


class _RawInterpretationOut(BaseModel):
    directives: list[_RawDirectiveOut]


class AnthropicLLMClient:
    """Implements app.llm_client.LLMClient using the Anthropic Messages API."""

    def __init__(self, api_key: str, model: str, timeout_s: float = 6.0) -> None:
        # max_retries=0: app.llm_interpreter owns the bounded-retry policy
        # (one correction attempt); the SDK's own hidden retry loop would
        # otherwise silently consume request-deadline budget outside that
        # accounting (see plan Sec. "Reliability, deadlines, and latency").
        self._client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=0)
        self._model = model
        self._timeout_s = timeout_s

    async def interpret(
        self,
        notes: list[str],
        battery: BatteryConfig,
        correction_feedback: str | None = None,
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
            response = await self._client.with_options(timeout=self._timeout_s).messages.parse(
                model=self._model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
                output_format=_RawInterpretationOut,
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
        if response.parsed_output is None:
            raise LLMOutputError(f"no_parsed_output: stop_reason={response.stop_reason}")

        return [d.model_dump() for d in response.parsed_output.directives]
