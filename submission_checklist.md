# Submission checklist

Mirrors the Participant Guide & Evaluation Rubric's "Final pre-submit
checklist" (Sec. 11). Status reflects what is true in this repository as of
the last commit -- items marked PENDING require action only you can take
(a provider key, a live deployment, a video recording).

- [x] `GET /health` reachable, returns `{"status":"ok"}` -- verified via
      the test suite and a local run.
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
      (`tests/test_optimizer.py`) and against 300 seeded random scenarios
      plus a 40-case independent MILP oracle cross-check.
- [x] `total_grid_kwh`, `total_cost_bdt`, `peak_grid_kwh` match values
      recalculated from `hourly_plan` -- enforced by
      `app/replay_validator.py`, run on every request before responding
      (never on stale/in-memory values, always post-JSON-round-trip).
- [x] README is self-contained: setup, required environment-variable
      names, model/provider, LLM role, guardrails, optimizer/solver, exact
      run command, `/health` test, `/optimize-energy` sample test, known
      limitations, no committed secrets.
- [ ] **PENDING (your action):** a real `LLM_PROVIDER`/`LLM_MODEL`/
      `LLM_API_KEY` configured and evaluated with
      `scripts/eval_interpreter.py` against `tests/fixtures/semantic_cases.json`
      (a starter set -- expand toward the plan's >=60 reviewed cases first).
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
- [ ] **PENDING (your action):** live deployment reachable from outside
      your development network; run `scripts/run_public_samples.py
      --base-url <public URL>` and `scripts/soak_api.py --base-url <public
      URL> --count 100 --concurrency 1 4` against it.
- [ ] **PENDING (your action):** <=3-minute architecture/solution video
      recorded and accessible to judges (tie-break only, but required for
      submission per Sec. 02).
