"""Read-only discovery of models exposed by the SleepyAI account."""

from __future__ import annotations

from typing import Any

import httpx


def _safe_model(raw: object) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    model_id = raw.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        return None
    allowed = (
        "id",
        "name",
        "provider",
        "description",
        "contextWindow",
        "context_window",
        "inputPrice",
        "outputPrice",
        "cacheReadPrice",
        "capabilities",
        "active",
        "available",
        "status",
    )
    return {key: raw[key] for key in allowed if key in raw}


async def discover_models(
    api_key: str,
    base_url: str = "https://www.sleepyai.org/api/v1",
    *,
    timeout_s: float = 15.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout_s,
        transport=transport,
    ) as client:
        response = await client.get("/models")
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        raw_models = payload.get("data", payload.get("models"))
    else:
        raw_models = payload
    if not isinstance(raw_models, list):
        raise ValueError("model catalog response does not contain a model list")
    models = [safe for raw in raw_models if (safe := _safe_model(raw)) is not None]
    return sorted(models, key=lambda item: item["id"])
