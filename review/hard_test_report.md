# GridWise hard-test report

Generated: 2026-09-18 (Asia/Dhaka)

## Release verdict

The local implementation passes its current correctness, security, static-analysis,
dependency, and coverage gates. The currently deployed Railway revision is not yet
release-accepted: it passed every public optimization case, but the first expanded
semantic run was 81/84 and two time-window cases changed answer on immediate rerun.
The exact SleepyAI model therefore remains unselected and the final 220-request live
soak is intentionally deferred until an authenticated comparison winner is pinned.

## Provider and model status

- Provider target: SleepyAI OpenAI-compatible API.
- Base URL: `https://www.sleepyai.org/api/v1`.
- Production route: `POST /chat/completions`, Bearer authentication,
  `stream=false`, `max_tokens=2048`.
- Model discovery: implemented as authenticated read-only `GET /models`.
- Accessible models: pending; the key was not available in the local process
  environment and was not copied into commands or reports.
- Selected exact model ID: pending.
- Model comparison report date: pending authenticated run.
- Selection rule: 100% exact semantic accuracy, then zero provider/format failures,
  then p95 <=5 seconds and every call <25 seconds, then lowest known token cost.

## Local verification

Environment: macOS, Python 3.14.0, pytest 9.1.1, Hypothesis 6.168.0,
SciPy 1.17.1/HiGHS. CI also defines a Python 3.12 and 3.14 matrix.

| Gate | Result |
| --- | --- |
| Pytest | 165 passed; one third-party Starlette/AnyIO deprecation warning |
| Branch coverage | 92.57% (required >=90%) |
| Ruff | Pass |
| Ruff formatting | Pass |
| Mypy | Pass for `app/` |
| Bandit | Pass for `app/` and `scripts/` |
| Dependency audit | No known vulnerabilities in runtime requirements |
| Repository credential-pattern scan | No credential-shaped strings found |
| Public optimizer corpus | 10/10; every recalculated cost gap <0.01 BDT |
| Generated optimizer cases | 2,000 deterministic feasible scenarios replayed |
| Independent oracle | First 100 generated cases matched independent MILP |
| Semantic corpus shape | 72 single-note cases + 12 mixed bundles; 24 frozen high-risk IDs |
| Docker | Dockerfile/CI smoke defined; local daemon unavailable |

Primary command:

```text
.venv/bin/python -m pytest --cov=app --cov-branch --cov-report=term-missing
```

Other gates run with Ruff 0.16.8, Mypy 2.3.1, Bandit 1.9.4, and
pip-audit 2.10.1. The deterministic optimizer seed is `20260918`; Hypothesis
stores reproducible failures in its normal example database.

## Adversarial coverage and fixes

The suite now covers arbitrary nested JSON, Unicode, duplicate JSON keys,
non-finite values, strict directive shapes, numeric boundaries, output replay
mutations, negative tariffs, zero-capacity storage, tiny decimals, overlapping
directives, forced scheduling, infeasible combinations, invalid UTF-8,
empty/scalar/array request bodies, body/text limits, provider refusals and malformed
responses, documented provider status classes, retries, exhausted deadlines,
concurrent requests, and `/health` responsiveness during a slow provider call.

Confirmed defect fixed: input validation previously allowed
`initial_energy_kwh < minimum_energy_kwh`. End-of-day neutrality makes such a
scenario necessarily infeasible at the final hour. It is now rejected as controlled
HTTP 400 input validation rather than reaching the solver.

Additional hardening includes a 256 KiB body limit, 4,000-character note limit,
256-character scenario ID limit, strict duplicate-key/non-finite rejection for
provider output, explicit provider error mapping, non-assert runtime invariants,
and closing asynchronous provider clients during shutdown.

## Live Railway results

Base URL: `https://worksonlocalhost-production.up.railway.app/`

- `/health`: pass.
- Official public cases: 10/10 pass; zero cost gap.
- Expanded semantic first pass: 81/84; p50 2.148s, p95 2.470s,
  max 3.206s.
- First-pass misses:
  - `SEM-034`: cross-midnight window included tomorrow's hours 0-1.
  - `SEM-051`: synthetic request was infeasible because the harness did not leave
    enough post-reserve demand to restore end-of-day battery neutrality. The harness
    was corrected; this was not scored as a model defect.
  - `SEM-061`: explicit “hours 15 through 17” returned only 15-16.
- Focused rerun after the harness correction: 3/3 pass; p95/max 4.23s. The two
  interpretation changes demonstrate nondeterminism, so they do not erase the
  first-pass failures.
- Earlier bounded soak: 40 requests total, zero failures/timeouts, worst p95 4.90s.
- Final 220-request soak: deferred until a model passes comparison and is deployed.

## Permanent CI

GitHub Actions now requires the Python 3.12/3.14 test and quality matrix, branch
coverage >=90%, Ruff, Mypy, Bandit, dependency audit, and gitleaks. A separate
Docker job builds the image, starts it with the explicitly enabled development
placeholder, verifies a non-root UID, waits for `/health`, and serves one safe
no-op scenario. Credentialed model comparison and live load testing remain manual.

## Remaining deployment blockers

1. Rotate the key that was exposed outside repository configuration.
2. Export the rotated key as `LLM_API_KEY` locally, run model discovery/comparison,
   and retain the generated credential-free JSON report.
3. Pin the winning exact ID as Railway `LLM_MODEL`; set `LLM_PROVIDER=sleepyai` and
   `LLM_BASE_URL=https://www.sleepyai.org/api/v1`; deploy this revision.
4. Require 84/84 on the complete live corpus plus three total runs of the 24
   high-risk cases.
5. Run the 100@c1, 100@c4, and 20@c8 live soak with continuous health checks.
6. Run the Docker CI job, publish a pullable exact image tag/digest, and verify it.
7. Provide the required public-after-deadline repository and <=3-minute video.
