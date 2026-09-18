"""LLM backend abstraction.

`LLMClient` is the only interface app/llm_interpreter.py depends on. Swapping
providers is a new class in app/providers/ + one line in app/main.py's
factory -- nothing in the interpretation prompt, guardrails, or the rest of
the pipeline changes.

PlaceholderLLMClient exists ONLY for local development before a real
provider key is available. It is not a fallback and must never run in a
submitted deployment: it returns all-no_op, which the design audit measured
violating a true directive in 9 of the 10 public sample cases. See
app/config.py -- production config refuses to start with this client
selected unless ALLOW_STUB_INTERPRETER is explicitly set.
"""
from __future__ import annotations

from typing import Protocol

from app.schemas import BatteryConfig


class LLMClientError(RuntimeError):
    """Base class for interpretation-backend failures."""


class LLMTransientError(LLMClientError):
    """Retryable: rate limit, timeout, connection error, 5xx."""


class LLMOutputError(LLMClientError):
    """Non-retryable in the same way: the call succeeded but the model
    declined, or produced output structured-output validation couldn't
    even parse into the expected shape at all (guardrails.py handles
    everything that *does* parse but is semantically wrong)."""


class LLMClient(Protocol):
    async def interpret(
        self,
        notes: list[str],
        battery: BatteryConfig,
        correction_feedback: str | None = None,
    ) -> list[dict]:
        """Return one raw directive dict per note, in any order, matching
        the shape app.guardrails.validate_directives expects as input
        (before validation/normalization).

        `correction_feedback`, when given, is a short guardrail rejection
        reason from a previous attempt -- used for the single bounded
        correction retry in app/llm_interpreter.py. It contains no request
        or note content, only a rule name, so it is always safe to include.

        Raises LLMTransientError or LLMOutputError on failure; never
        returns a partially-formed result.
        """
        ...


class PlaceholderLLMClient:
    """Dev-only stub. Returns all-no_op, clearly flagged as such.

    Exists so the rest of the pipeline (guardrails -> constraints ->
    optimizer -> replay -> API) is runnable and testable end to end before
    a real provider key is wired in. NOT a valid competition configuration
    -- see module docstring and app/config.py.
    """

    async def interpret(
        self,
        notes: list[str],
        battery: BatteryConfig,
        correction_feedback: str | None = None,
    ) -> list[dict]:
        return [
            {
                "note_index": i,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": (
                    "PLACEHOLDER INTERPRETER: no real LLM is configured "
                    "(LLM_PROVIDER=placeholder). This response is dev-only "
                    "and not a valid competition submission."
                ),
            }
            for i in range(len(notes))
        ]
