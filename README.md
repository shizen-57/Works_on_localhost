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
| `LLM_PROVIDER` | yes | `anthropic` \| `openai` \| `placeholder`. `placeholder` is dev-only (see below) and refuses to start unless `ALLOW_STUB_INTERPRETER=true` is also set. |
| `LLM_MODEL` | if provider != placeholder | Model identifier for the chosen provider. |
| `LLM_API_KEY` | if provider != placeholder | API key for the chosen provider. Read at runtime only -- never baked into the image or committed. |
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
# edit .env: set LLM_PROVIDER=anthropic, LLM_MODEL=claude-opus-5, LLM_API_KEY=sk-...
export $(grep -v '^#' .env | xargs)

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

83 tests: schema strictness, the full guardrail adversarial set (unsupported
types, malformed hours/factors, positional-index attacks), the optimizer
against all 10 public cases plus 300 seeded random feasible scenarios (40
cross-checked against an independent MILP oracle), replay mutation-rejection
probes, the full HTTP API (including a fixture-driven fake LLM client that
returns organizer-correct answers, to exercise the complete contract without
a live key), and deadline enforcement under a hanging provider call.

### Real-model semantic evaluation (cannot be skipped before submission)

```bash
LLM_PROVIDER=anthropic LLM_MODEL=claude-opus-5 LLM_API_KEY=sk-... \
  python3 scripts/eval_interpreter.py --repeats 3
```

Runs the labeled cases in `tests/fixtures/semantic_cases.json` (a **starter
set** -- see that file's `_meta.purpose`; expand toward the plan's >=60
reviewed cases before treating a run of this as the final gate) directly
against the configured `LLMClient`, 3x each to expose nondeterminism, and
reports exact-note / per-field / per-directive-type accuracy and latency.

### Reliability soak (before submission, against the deployed URL)

```bash
python3 scripts/soak_api.py --base-url https://your-deployment.example.com \
  --count 100 --concurrency 1 4
```

## Docker

```bash
docker build -t gridwise .
docker run --rm -p 8000:8000 \
  -e LLM_PROVIDER=anthropic -e LLM_MODEL=claude-opus-5 -e LLM_API_KEY=sk-... \
  gridwise

curl http://localhost:8000/health
```

The image is multi-stage (`python:3.12-slim`), runs as a non-root user,
binds `0.0.0.0:$PORT`, has a healthcheck against `/health`, and never bakes
in secrets -- `LLM_API_KEY` etc. are read from the environment at container
start only. **Note:** the container build/run was validated by running the
exact application under the Dockerfile's shell-form CMD (`sh -c 'uvicorn ...
--port $PORT'`) and its healthcheck one-liner directly in the development
environment -- both work correctly. A full `docker build` was not
executable in the development sandbox (anonymous Docker Hub pulls were
rate-limited there). **Run `docker build` and `docker run` once for real
before submitting**, per the Participant Guide's Docker fallback
requirement.

## Model/provider and solver

- Interpretation: configurable via `LLM_PROVIDER` (Anthropic Messages API
  structured output via `client.messages.parse`, or a provider of your
  choice by implementing `app.llm_client.LLMClient` -- see
  `app/providers/anthropic_provider.py` for the reference implementation).
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
- **`initial_energy_kwh < minimum_energy_kwh` is accepted at the schema
  layer** (not auto-rejected) -- Problem Statement Sec. 9.2 only constrains
  `E_after`, so hour 0 charging up into compliance is a legitimate feasible
  scenario; the LP decides feasibility, not the schema.
- **Negative tariffs are accepted** -- the schema only requires the field
  to be a finite number, not non-negative.
- The Anthropic provider adapter (`app/providers/anthropic_provider.py`) was
  written against the documented SDK request/response shapes but has not
  been exercised against a live key in this environment -- run
  `scripts/eval_interpreter.py` against a real key before submitting.

## Dependencies and credits

FastAPI, Pydantic, uvicorn, httpx (web framework/validation/HTTP), NumPy and
SciPy/HiGHS (LP solver), the Anthropic Python SDK (`anthropic`) for the
reference LLM provider adapter, and pytest/pytest-asyncio for the test
suite. See `requirements.txt` / `requirements-dev.txt` for exact pinned
versions.
