# GridWise LLM

An HTTP service for the BUP CSE Fest 2026 "Smart Campus Energy Optimization
Challenge" preliminary round. It reads a 24-hour campus energy scenario plus
1-3 natural-language operator notes, interprets the notes with a real
language model into strict structured directives, validates them
deterministically, and returns a valid, cost-minimizing 24-hour grid/solar/
battery schedule.

## Architecture

```
                 ┌──────────────────┐
 operator_notes  │  LLM interpreter  │  untrusted structured JSON
 + battery  ───▶ │ (app/llm_        │ ───────────────┐
   context       │  interpreter.py) │                 ▼
                 └──────────────────┘        ┌──────────────────┐
                                              │    guardrails     │  fail-closed
                                              │ (app/guardrails.py)│  validation
                                              └──────────────────┘
                                                        │ validated directives
                                                        ▼
                                              ┌──────────────────┐
                                              │    constraints     │  directives ->
                                              │(app/constraints.py)│  per-hour bounds
                                              └──────────────────┘
                                                        │
                                                        ▼
                                              ┌──────────────────┐
                                              │   LP optimizer     │  HiGHS,
                                              │ (app/optimizer.py) │  full netting
                                              └──────────────────┘
                                                        │ hourly_plan
                                                        ▼
                                              ┌──────────────────┐
                                              │ replay validator    │  independent
                                              │(app/replay_         │  final check
                                              │  validator.py)      │
                                              └──────────────────┘
                                                        │
                                                        ▼
                                              POST /optimize-energy response
```

The LLM is only ever a *proposer*. Its output is untrusted structured data
until `app/guardrails.py` validates it -- nothing downstream ever sees raw
model output. `app/replay_validator.py` is a second, independent pass
(deliberately not sharing bound-compilation code with `app/constraints.py`)
that re-checks the final serialized response against every rule before it
is returned, the same way the judge's own independent replay works.

- **LLM role**: interprets each operator note into one of six directive
  types (`solar_reduction`, `minimum_battery_reserve`, `no_charge_window`,
  `no_discharge_window`, `max_grid_window`, `no_op`). One batched call per
  request covers all 1-3 notes together (keeps latency low, gives the model
  the battery object as context for percentage-of-capacity notes like "keep
  50% in reserve"). See `app/llm_interpreter.py` for the versioned prompt.
- **Guardrails**: fail closed. An unsupported `directive_type` is rejected,
  never silently turned into `no_op` -- that would manufacture a confident
  but wrong schedule that silently drops a real constraint. `note_index` is
  trusted only as a value, never as array position. The only "repair" ever
  applied is sorting an already-valid, already-unique set of entries/hours
  -- nothing that could change meaning. See `app/guardrails.py`.
- **Optimizer**: a continuous LP (`scipy.optimize.linprog`, HiGHS), 4
  variables/hour (grid, solar_used, charge, discharge). This exact
  formulation reproduces the organizer's optimal cost on all 10 public
  sample cases with zero gap, and was independently cross-checked against a
  separate binary-exclusivity MILP on hundreds of generated cases (see
  `tests/test_optimizer.py`). Simultaneous charge+discharge in the raw LP
  solution (a real LP degeneracy, not noise) is always fully netted into a
  single action -- see the docstring in `app/optimizer.py` for the proof
  sketch.

## Environment variables

| Variable | Required | Meaning |
| --- | --- | --- |
| `LLM_PROVIDER` | yes | `anthropic` \| `sleepyai` \| `placeholder`. `placeholder` is dev-only (see below) and refuses to start unless `ALLOW_STUB_INTERPRETER=true` is also set. |
| `LLM_MODEL` | if provider != placeholder | Model identifier for the chosen provider. |
| `LLM_API_KEY` | if provider != placeholder | API key for the chosen provider. Read at runtime only -- never baked into the image or committed. |
| `LLM_BASE_URL` | for `sleepyai` | OpenAI-compatible API root. Use `https://www.sleepyai.org/api/v1`; the adapter calls `/chat/completions`. |
| `PORT` | no (default 8000) | Port the service binds to on `0.0.0.0`. |
| `REQUEST_DEADLINE_S` | no (default 25) | Absolute per-request deadline; must be in (0, 30] to respect the judge's 30s cutoff. |
| `ALLOW_STUB_INTERPRETER` | no (default false) | Dev escape hatch only. Must be `false`/unset for any real deployment. |
| `LOG_LEVEL` | no (default INFO) | Standard logging level. |

Copy `.env.example` to `.env` and fill in real values locally; `.env` is
git-ignored.

### About the placeholder provider

`LLM_PROVIDER=placeholder` returns an all-`no_op` response and exists ONLY
so the rest of the pipeline (schema, guardrails, optimizer, replay, API
contract) is runnable and testable before a provider key is available. **It
is not a valid competition configuration.** An offline design audit
measured this fallback violating a true operator directive in 9 of the 10
public sample cases -- shipping it would fail the mandatory
LLM-in-the-interpretation-path requirement. `app/config.py` refuses to
start the service with this provider unless `ALLOW_STUB_INTERPRETER=true`
is explicitly set, specifically to make this impossible to do by accident.

## Local quickstart (clean environment)

```bash
git clone <this-repo> && cd <this-repo>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env for Anthropic, or use these SleepyAI settings with your own
# model identifier and key:
# LLM_PROVIDER=sleepyai
# LLM_MODEL=<model-id>
# LLM_API_KEY=<secret>
# LLM_BASE_URL=https://www.sleepyai.org/api/v1
set -a; source .env; set +a

uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
```

In another terminal:

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl -s -X POST http://localhost:8000/optimize-energy \
  -H "content-type: application/json" \
  -d @<(python3 -c "import json; print(json.dumps(json.load(open('data/public_sample_cases.json'))['cases'][0]['input']))")

# Runs all 10 public samples against a live instance and checks
# interpretation, directive application, replay validity, and cost --
# HTTP 200 alone is not treated as a pass:
python3 scripts/run_public_samples.py --base-url http://localhost:8000
```

Expected result: `10/10 public cases passed` once a real provider is
configured (with `LLM_PROVIDER=placeholder`, expect every case to fail on
interpretation -- that is the intended, documented behavior described
above, not a bug in the script).

### Running tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The current suite has 165 tests. It covers strict schema/configuration and
HTTP parsing, duplicate keys, non-finite values, request-size and text limits,
both provider transports and error classes, adversarial guardrails, replay
mutation rejection, deadlines/cancellation, health responsiveness under load,
and Hypothesis-generated JSON/directive/optimizer inputs. The optimizer is
also exercised on all 10 public cases, 2,000 deterministic random feasible
scenarios, and 100 independent MILP cost cross-checks.

The permanent CI gate runs Python 3.12 and 3.14, branch coverage >=90%, Ruff,
Mypy, Bandit, dependency audit, secret scanning, and a clean Docker build/smoke
test that verifies the process is non-root, reaches `/health`, and serves a
safe no-op request. Credentialed provider and live-load tests remain manual so
CI cannot spend model quota.

### SleepyAI model discovery and selection (cannot be skipped before submission)

```bash
read -s LLM_API_KEY && export LLM_API_KEY
python3 scripts/discover_models.py --report eval_output/accessible-models.json
python3 scripts/compare_models.py
```

Discovery performs authenticated, read-only `GET /models`. Comparison uses the
same official OpenAI-compatible `/chat/completions` transport, retry budget,
and production guardrails as the service. It evaluates the frozen 72 single
notes plus 12 mixed-note bundles once, repeats the frozen 24-case high-risk set
three times, and reports exact/per-field/per-type accuracy, call and retry
rates, latency, tokens, and estimated cost. The timestamped JSON report selects
only a model with 100% accuracy, zero failures, p95 <=5 seconds, and every call
under 25 seconds; lowest known cost wins after those gates. Pin the exact winner
as `LLM_MODEL` in Railway. Do not silently switch it during judging.

To test the complete deployed pipeline without placing provider credentials on
the test machine, run the same labeled notes through the public API. This also
replays each returned schedule against fixture ground truth, not merely against
the service's own reported interpretation:

```bash
python3 scripts/eval_live_semantics.py \
  --base-url https://your-deployment.example.com --concurrency 2
```

### Reliability soak (before submission, against the deployed URL)

```bash
python3 scripts/soak_api.py --base-url https://your-deployment.example.com \
  --count 100 --concurrency 1 4
```

The default soak runs 100 requests at concurrency 1, another 100 at concurrency
4, and a final 20-request concurrency-8 burst. It independently replays every
response, probes `/health` throughout the load, and enforces zero failures,
p95 <=5 seconds, every request under 30 seconds, and health <=1 second.

### Latest live verification

On 2026-09-18, the currently deployed Railway revision passed:

- `/health` readiness;
- all 10 public sample cases, with zero cost gap on every case;
- a 40-request bounded soak (20 requests each at concurrency 1 and 4), with
  zero failures/timeouts and worst measured p95 4.90s.

The expanded 84-case corpus then scored 81/84 on its first pass (p50 2.15s,
p95 2.47s). The three failed cases all passed on immediate rerun after one
fixture-feasibility correction, exposing nondeterministic interpretation in
two time-window cases. Therefore the current live model is **not yet accepted**.
Run model comparison, deploy the selected exact model/base URL, then rerun the
complete live corpus and default 220-request soak before submission.

## Docker

```bash
docker build -t gridwise .
docker run --rm -p 8000:8000 \
  -e LLM_PROVIDER=sleepyai -e LLM_MODEL='<selected-exact-id>' \
  -e LLM_API_KEY='<secret>' -e LLM_BASE_URL=https://www.sleepyai.org/api/v1 \
  gridwise

curl http://localhost:8000/health
```

The image is multi-stage (`python:3.12-slim`), runs as a non-root user,
binds `0.0.0.0:$PORT`, has a healthcheck against `/health`, and never bakes
in secrets -- `LLM_API_KEY` etc. are read from the environment at container
start only. The exact application command and healthcheck have been exercised
locally, but the Docker daemon is unavailable in this environment, so a full
local `docker build`/`docker run` was not possible. GitHub Actions now performs
that build/runtime smoke; a pullable registry image with an exact tag/digest
still must be published and verified before submission.

## Model/provider and solver

- Interpretation: configurable via `LLM_PROVIDER`. The native Anthropic adapter
  uses structured output via `client.messages.parse`; the SleepyAI adapter uses
  the documented OpenAI-compatible `POST /chat/completions` endpoint with
  Bearer authentication, non-streaming responses, a 2,048-token output cap,
  and strict `choices[0].message.content` JSON extraction. Both feed the same
  deterministic guardrails. See `app/providers/`.
- Optimizer/solver: `scipy.optimize.linprog` with the `highs` method.
- Web framework: FastAPI + uvicorn. Request/response validation: Pydantic v2
  in strict mode.

## Known limitations and documented assumptions

- **Overlapping directives** (not fully specified by the Problem
  Statement): multiple `minimum_battery_reserve` on the same hour take the
  **max**; multiple `max_grid_window` take the **min**; multiple
  `solar_reduction` take the most restrictive (lowest resulting effective
  solar). See `app/constraints.py`.
- **Cross-midnight time windows** are not fully specified by the challenge.
  The prompt instructs the model to include only the portion of such a
  window that falls within hours 0-23 of the current scenario and to note
  the ambiguity in its explanation.
- **Battery initial-state consistency is validated early.** Because end-of-day
  neutrality makes the final state equal `initial_energy_kwh`, the initial
  value must already be between the base minimum and capacity; otherwise the
  final hour can never satisfy both rules.
- **Negative tariffs are accepted** -- the schema only requires the field
  to be a finite number, not non-negative.
- Requests above 256 KiB, notes above 4,000 characters, and scenario IDs above
  256 characters are rejected with controlled HTTP 400 responses.
- The exact SleepyAI model ID remains a release blocker until authenticated
  discovery/comparison is run and the winning ID is pinned in Railway.

## Dependencies and credits

FastAPI, Pydantic, uvicorn, httpx (web framework/validation/HTTP), NumPy and
SciPy/HiGHS (LP solver), and the Anthropic Python SDK (`anthropic`) for the
optional reference provider adapter. The development suite additionally uses
pytest, pytest-asyncio, Hypothesis, pytest-cov, Ruff, Mypy, Bandit, and
pip-audit. See `requirements.txt` / `requirements-dev.txt` for exact versions.
