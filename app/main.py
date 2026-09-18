"""FastAPI app: GET /health, POST /optimize-energy (Problem Statement Sec. 06).

Orchestrates: strict parse -> LLM interpretation (bounded-retry) ->
guardrails -> constraints -> LP -> serialize -> JSON round trip -> replay
-> respond. Every stage's failure mode maps to a controlled status code;
nothing here can crash the process on bad input or a bad model response
(Problem Statement Sec. 08, "SAFE FAILURE").
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from app.config import ConfigError, Settings, load_settings
from app.constraints import compile_bounds
from app.errors import json_error, log_and_error, logger
from app.llm_client import LLMClient, PlaceholderLLMClient
from app.llm_interpreter import InterpretationFailed, interpret_notes
from app.optimizer import InfeasibleScenario, SolveResult, solve
from app.replay_validator import ReplayViolation, replay
from app.schemas import (
    BatteryConfig,
    DirectiveInterpretation,
    HourEntry,
    HourlyPlanEntry,
    OptimizeResponse,
    ScenarioRequest,
)
from app.summary import build_summary

logging.basicConfig(level=logging.INFO)
MAX_REQUEST_BODY_BYTES = 256 * 1024


class RequestBodyTooLarge(ValueError):
    """Raised before JSON parsing when the request exceeds the service limit."""


async def _read_limited_body(request: Request) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            parsed_length = int(content_length)
        except ValueError as exc:
            raise ValueError("invalid content-length header") from exc
        if parsed_length < 0:
            raise ValueError("invalid content-length header")
        if parsed_length > MAX_REQUEST_BODY_BYTES:
            raise RequestBodyTooLarge

    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_REQUEST_BODY_BYTES:
            raise RequestBodyTooLarge
    return bytes(body)


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "placeholder":
        return PlaceholderLLMClient()
    if settings.llm_provider == "anthropic":
        from app.providers.anthropic_provider import AnthropicLLMClient

        return AnthropicLLMClient(
            api_key=settings.llm_api_key,  # type: ignore[arg-type]  # config.py guarantees non-None here
            model=settings.llm_model,
        )
    if settings.llm_provider == "sleepyai":
        from app.providers.sleepyai_provider import SleepyAILLMClient

        return SleepyAILLMClient(
            api_key=settings.llm_api_key,  # type: ignore[arg-type]
            model=settings.llm_model,
            base_url=settings.llm_base_url,  # type: ignore[arg-type]  # config.py guarantees non-None here
        )
    # Defensive fallback in case config and this factory ever drift apart.
    raise ConfigError(f"no LLMClient implementation is wired up for LLM_PROVIDER={settings.llm_provider!r}")


def _reject_nonfinite_literal(token: str) -> float:
    raise ValueError(f"non-finite numeric literal in JSON: {token!r}")


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _warmup_solver() -> None:
    """One trivial solve at startup so the first real request doesn't pay
    HiGHS/scipy import or JIT cost, and so a broken solver install fails
    fast at boot rather than on the first judge request."""
    battery = BatteryConfig(
        capacity_kwh=10,
        initial_energy_kwh=5,
        minimum_energy_kwh=0,
        max_charge_kwh_per_hour=5,
        max_discharge_kwh_per_hour=5,
    )
    hours = [HourEntry(hour=h, demand_kwh=1, solar_kwh=0, tariff_bdt_per_kwh=1) for h in range(24)]
    bounds = compile_bounds(hours, battery, [])
    solve(hours, battery, bounds)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        settings = load_settings()
    except ConfigError:
        logger.exception("config_invalid: service will not start")
        raise
    app.state.settings = settings
    logging.getLogger().setLevel(settings.log_level)
    app.state.llm_client = build_llm_client(settings)
    await asyncio.to_thread(_warmup_solver)
    logger.info(
        "gridwise ready: provider=%s model=%s deadline_s=%s",
        settings.llm_provider,
        settings.llm_model or "-",
        settings.request_deadline_s,
    )
    try:
        yield
    finally:
        close = getattr(app.state.llm_client, "aclose", None)
        if close is not None:
            await close()


app = FastAPI(title="GridWise LLM", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


async def _run_pipeline(req: ScenarioRequest, llm_client: LLMClient) -> tuple[list[dict], SolveResult]:
    hours = req.hours_by_index()
    directives = await interpret_notes(llm_client, req.operator_notes, req.battery)
    bounds = compile_bounds(hours, req.battery, directives)
    result = await asyncio.to_thread(solve, hours, req.battery, bounds)
    return directives, result


@app.post("/optimize-energy", response_model=OptimizeResponse)
async def optimize_energy(request: Request) -> Response:
    settings: Settings = request.app.state.settings
    llm_client: LLMClient = request.app.state.llm_client

    try:
        raw_body = await _read_limited_body(request)
    except RequestBodyTooLarge:
        return json_error(
            400,
            "request_too_large",
            f"request body must not exceed {MAX_REQUEST_BODY_BYTES} bytes",
        )
    except ValueError as exc:
        return json_error(400, "invalid_request", str(exc))
    try:
        parsed = json.loads(
            raw_body,
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_nonfinite_literal,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        return json_error(400, "malformed_json", str(exc))

    try:
        req = ScenarioRequest.model_validate(parsed)
    except ValidationError as exc:
        return json_error(400, "invalid_request", exc.errors(include_url=False).__repr__())

    start = time.monotonic()
    try:
        directives, result = await asyncio.wait_for(_run_pipeline(req, llm_client), timeout=settings.request_deadline_s)
    except TimeoutError:
        return log_and_error(
            500,
            "internal_deadline_exceeded",
            "the request could not be completed within the service's internal time budget",
        )
    except InterpretationFailed as exc:
        return log_and_error(500, "interpretation_failed", "operator note interpretation failed", exc=exc)
    except InfeasibleScenario as exc:
        return log_and_error(500, "optimization_infeasible", "no valid schedule could be produced", exc=exc)
    except Exception as exc:  # noqa: BLE001 -- last-resort safe failure, Sec. 08
        return log_and_error(500, "internal_error", "an internal error occurred", exc=exc)

    elapsed = time.monotonic() - start
    if elapsed > settings.request_deadline_s:
        logger.warning("pipeline completed but exceeded deadline: %.2fs", elapsed)

    response_model = OptimizeResponse(
        scenario_id=req.scenario_id,
        directive_interpretation=[DirectiveInterpretation.model_validate(item) for item in directives],
        hourly_plan=[
            HourlyPlanEntry(
                hour=p.hour,
                grid_kwh=p.grid_kwh,
                solar_used_kwh=p.solar_used_kwh,
                battery_action=p.battery_action,
                battery_kwh=p.battery_kwh,
                battery_energy_after_kwh=p.battery_energy_after_kwh,
            )
            for p in result.hourly_plan
        ],
        total_grid_kwh=result.total_grid_kwh,
        total_cost_bdt=result.total_cost_bdt,
        peak_grid_kwh=result.peak_grid_kwh,
        plan_summary=build_summary(directives, result),
    )

    # Replay the response AFTER a JSON round trip, not the in-memory
    # dataclasses/pydantic objects -- serialization itself is part of what
    # this check is meant to catch (see app/replay_validator.py docstring).
    response_json = json.loads(json.dumps(response_model.model_dump(mode="json")))
    request_json = req.model_dump(mode="json")

    try:
        replay(request_json, directives, response_json, tol=1e-6)
    except ReplayViolation as exc:
        # This is our own bug, never the caller's fault: the pipeline
        # produced a plan that fails the same check the judge will run
        # independently. Fail loudly server-side, safely to the client.
        return log_and_error(
            500,
            "internal_replay_check_failed",
            "the service could not verify its own generated schedule",
            exc=exc,
        )

    return JSONResponse(status_code=200, content=response_json)
