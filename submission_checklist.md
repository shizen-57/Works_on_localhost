# Submission checklist

Mirrors the Participant Guide & Evaluation Rubric's "Final pre-submit
checklist" (Sec. 11). Status reflects what is true in this repository as of
the last commit -- items marked PENDING require action only you can take
(a provider key, a live deployment, a video recording).

- [x] `GET /health` reachable, returns `{"status":"ok"}` -- verified via
      the test suite, a local run, and the public Railway deployment on
      2026-09-18.
- [x] `POST /optimize-energy` accepts 1-3 `operator_notes` with the exact
      Problem Statement schema (strict validation, verified in
      `tests/test_schemas.py` and `tests/test_api.py`).
- [x] Every operator note produces exactly one `directive_interpretation`
      entry in `note_index` order; `no_op` uses `applies=false` + null
      adjustment; every other directive uses `applies=true` with the exact
      required `structured_adjustment` shape -- enforced by
      `app/guardrails.py`, fail-closed (verified adversarially in
      `tests/test_guardrails.py`).
- [x] LLM output is deterministically guardrailed before optimization;
      directive hours are unique integers 0-23 ascending; numeric values
      validated; invalid model output cannot silently invent a constraint
      (`app/guardrails.py` never coerces an unsupported type to `no_op`,
      never repairs `note_index` by array position).
- [x] `hourly_plan` obeys organizer-ground-truth directives plus energy
      balance, effective-solar, battery, rate-limit, grid-cap, and
      end-of-day rules -- verified against all 10 public cases exactly
      (`tests/test_optimizer.py`) and against 2,000 seeded random scenarios
      plus a 100-case independent MILP oracle cross-check.
- [x] `total_grid_kwh`, `total_cost_bdt`, `peak_grid_kwh` match values
      recalculated from `hourly_plan` -- enforced by
      `app/replay_validator.py`, run on every request before responding
      (never on stale/in-memory values, always post-JSON-round-trip).
- [x] README is self-contained: setup, required environment-variable
      names, model/provider, LLM role, guardrails, optimizer/solver, exact
      run command, `/health` test, `/optimize-energy` sample test, known
      limitations, no committed secrets.
- [x] The frozen semantic release corpus contains 72 reviewed single-note
      cases and 12 mixed-note bundles, including paraphrases, boundaries,
      percentage/fraction conversions, negation, distractors, and injection.
- [ ] **PENDING:** the current public deployment passed all 10 public samples,
      but its first expanded semantic pass was 81/84 (p50 2.15s, p95 2.47s).
      All three cases passed on immediate rerun after correcting one synthetic
      fixture's feasibility, demonstrating nondeterminism in two time-window
      interpretations. Discover/compare accessible models, pin the exact winner,
      deploy the new `/api/v1/chat/completions` adapter, then require 84/84.
- [ ] **PENDING (your action):** repository created after question reveal,
      kept private during the event, made public after the submission
      deadline; the submitted endpoint remains reachable throughout
      evaluation.
- [ ] **PENDING (your action):** Docker fallback image built, pushed to a
      registry with an exact tag/digest, and verified with a fresh
      `docker pull` + the documented `docker run` command reaching
      `/health` (the Dockerfile is written and the app was verified to run
      correctly under its exact CMD; a full `docker build` could not be
      executed in the development sandbox -- see README "Docker" section).
- [x] Live deployment is reachable externally and passed all 10 public cases
      with zero cost gap. A bounded 40-request soak (20 each at concurrency 1
      and 4) had zero failures/timeouts and worst p95 4.90s.
- [ ] **PENDING:** after the selected model is deployed, run the full release
      soak: 100 requests at concurrency 1, 100 at concurrency 4, and a final
      20-request concurrency-8 burst while continuously probing `/health`.
- [x] Permanent CI covers Python 3.12 and 3.14, branch coverage >=90%, Ruff,
      Mypy, Bandit, dependency audit, secret scan, and a non-root Docker smoke.
- [ ] **PENDING (your action):** <=3-minute architecture/solution video
      recorded and accessible to judges (tie-break only, but required for
      submission per Sec. 02).
