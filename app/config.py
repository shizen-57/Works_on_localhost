"""Service configuration.

Deliberately fails closed: a production process with no real interpretation
provider configured must refuse to become ready, not silently serve an
all-no_op fallback. The offline audit measured that fallback violating a
true directive in 9 of the 10 public sample cases.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Keep this list limited to providers that build_llm_client actually wires up.
# Advertising a provider here that fails later in the factory turns a clear
# configuration mistake into a less useful startup error.
VALID_PROVIDERS = {"anthropic", "sleepyai", "placeholder"}
VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}


class ConfigError(RuntimeError):
    """Raised when the process configuration cannot start the service."""


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_model: str
    llm_api_key: str | None
    llm_base_url: str | None
    port: int
    log_level: str
    request_deadline_s: float
    allow_stub_interpreter: bool

    @property
    def using_stub_interpreter(self) -> bool:
        return self.llm_provider == "placeholder"


def load_settings(env: dict[str, str] | None = None) -> Settings:
    """Parse and validate environment variables into Settings.

    Raises ConfigError on anything that would make the service unable to
    meet the mandatory-LLM requirement or the documented contract. This is
    called once at startup (see app/main.py) so a bad deploy fails fast
    with a clear message instead of passing /health and failing every
    request, or worse, silently degrading to no_op.
    """
    e = env if env is not None else os.environ

    provider = e.get("LLM_PROVIDER", "").strip().lower()
    if provider not in VALID_PROVIDERS:
        raise ConfigError(f"LLM_PROVIDER must be one of {sorted(VALID_PROVIDERS)}, got {provider!r}")

    allow_stub = e.get("ALLOW_STUB_INTERPRETER", "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    if provider == "placeholder" and not allow_stub:
        raise ConfigError(
            "LLM_PROVIDER=placeholder requires ALLOW_STUB_INTERPRETER=true. "
            "The placeholder returns all-no_op and is NOT a valid competition "
            "configuration -- it exists only for local development before a "
            "real provider key is available. Never set this in the submitted "
            "deployment."
        )

    model = e.get("LLM_MODEL", "").strip()
    api_key = e.get("LLM_API_KEY", "").strip() or None
    base_url = e.get("LLM_BASE_URL", "").strip() or None

    if provider != "placeholder":
        if not model:
            raise ConfigError(f"LLM_MODEL is required when LLM_PROVIDER={provider!r}")
        if not api_key:
            raise ConfigError(f"LLM_API_KEY is required when LLM_PROVIDER={provider!r}")

    if provider == "sleepyai" and not base_url:
        raise ConfigError(
            "LLM_BASE_URL is required when LLM_PROVIDER=sleepyai "
            "(use the documented OpenAI-compatible root "
            "https://www.sleepyai.org/api/v1)."
        )

    try:
        port = int(e.get("PORT", "8000"))
    except ValueError as exc:
        raise ConfigError(f"PORT must be an integer, got {e.get('PORT')!r}") from exc
    if not (1 <= port <= 65535):
        raise ConfigError(f"PORT out of range: {port}")

    try:
        deadline = float(e.get("REQUEST_DEADLINE_S", "25"))
    except ValueError as exc:
        raise ConfigError(f"REQUEST_DEADLINE_S must be a number, got {e.get('REQUEST_DEADLINE_S')!r}") from exc
    if not (0 < deadline <= 30):
        raise ConfigError(
            f"REQUEST_DEADLINE_S must be in (0, 30] to respect the judge's 30s per-request cutoff, got {deadline}"
        )

    log_level = e.get("LOG_LEVEL", "INFO").strip().upper()
    if log_level not in VALID_LOG_LEVELS:
        raise ConfigError(f"LOG_LEVEL must be one of {sorted(VALID_LOG_LEVELS)}, got {log_level!r}")

    return Settings(
        llm_provider=provider,
        llm_model=model,
        llm_api_key=api_key,
        llm_base_url=base_url,
        port=port,
        log_level=log_level,
        request_deadline_s=deadline,
        allow_stub_interpreter=allow_stub,
    )
