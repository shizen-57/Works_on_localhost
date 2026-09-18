# GridWise plan review and executable smoke evidence

Date: 2026-09-18. Scope: review/improve the implementation plan and test its assumptions.
The team-ID directory was excluded. No external model, account or endpoint was used.

## Verdict

The original plan is not a viable submission as written: its production placeholder
violates the mandatory model requirement, unsafe recovery can erase hard directives,
and release work is incomplete. The underlying LP survives the tests with full
charge/discharge netting. `../implementation.md` now specifies the corrected architecture
and explicit release gates. It is ready to guide implementation, not proof that a deployed
system is ready or that the team will win.

## Reproduce

From `E:\bup hackathon\bup_hackathon` with NumPy and SciPy available:

```powershell
python review/smoke_plan.py --report review/smoke_results.json
```

The script reads the explicit extracted public JSON, seeds generated cases with 20260918,
raises on any unexpected failure, and writes a machine-readable report. It is a standalone
design prototype, not production application code. Do not use `python -O`, which disables
its assertion checks. Runtime/package versions and the sample SHA-256 are in the report.

## Observed results

| Probe | Result |
| --- | --- |
| Public reference-plan replay | 10/10 passed |
| Proposed LP versus public optimal costs | 10/10, exactly zero reported cost gap |
| Seeded generated feasible scenarios | 300/300 passed independent replay after JSON round trip |
| Explicit-state, exclusive-action MILP comparison | 50/50 matched LP cost within 0.00001 BDT |
| Revised guardrail prototype, malformed outputs | 31/31 rejected |
| Safe hour normalization and reordering by valid note_index | Passed |
| Corrupted final schedules/totals | 11/11 rejected |
| Constructed impossible zero-grid/no-storage hour | Infeasibility detected |
| Original all-no_op approach, semantic correctness | Fails every public case |
| Original all-no_op approach, actual schedules against ground truth | 9/10 schedules violate a directive |
| Generated cases with simultaneous charge/discharge >1e-6 | 89/300; all replayed successfully after full netting |
| Two-decimal precision counterexample | 3.528 BDT cost difference, exceeding the 0.01 tolerance |

The final run measured local LP p95 at approximately 5.27ms. An earlier run measured
approximately 6.73ms and a 73.54ms first solve. These are small local experiments, not
an API benchmark or a guarantee across hosts. Model/network latency was not measured.

## Findings and changes

1. **Critical: production no_op placeholder.** Every public case has an active directive.
   Solving while ignoring those directives violated true solar/window/reserve constraints
   in nine cases in this run; SAMPLE-05 happened to satisfy its grid cap incidentally, but
   its interpretation would still be wrong. The revised plan requires a real provider,
   production configuration validation, and actual-model evaluation.

2. **High: unsafe repair can produce believable wrong success.** Unknown directive types
   must not be changed into no_op. Array position is not evidence of note identity when
   model output is reordered or duplicated. The revised policy sorts valid indices,
   rejects invalid mappings, and permits one bounded correction before controlled failure.
   Strict applies/null and numeric/shape validation replaces silent semantic repair.

3. **High: simultaneous battery flow is not merely noise.** LP degeneracy produced
   meaningful simultaneous flows in 89 generated scenarios. Netting the full difference
   is valid under lossless accounting, preserves cost/state/balance, and satisfies rate
   limits. The plan now explains that invariant; the MILP checks enforce exclusivity
   independently to compare optimal costs. The exact count is solver/version dependent.

4. **High: precision needs an explicit policy.** Twenty-four grid values of 0.0049 kWh
   at 30 BDT/kWh total 3.528 BDT. Rounding each grid value to two decimals erases that
   cost. Recomputing totals afterward avoids a totals mismatch but does not repair the
   changed schedule/energy balance. The plan preserves precision and replays serialized values.

5. **High: mocked tests do not establish semantic performance.** A fake interpreter can
   give perfect expected constraints regardless of real language understanding. The plan
   adds reviewed held-out labels, repeated real-provider runs, semantic/ground-truth replay
   in live sample checks, and reports of accuracy, failures and latency.

6. **High: nominal timeout is not an end-to-end deadline.** Queue wait, SDK retry loops,
   correction calls and synchronous solver work can exceed 30 seconds or stall health.
   The plan now budgets the whole request, caps attempts, bounds worker/concurrency use,
   and makes provider-aware soak/load tests a release gate. These policies are not yet
   implemented or exercised against a server.

7. **High: missing scored/required artifacts.** Added external verification, published and
   freshly pulled Docker reference, fresh-environment reproduction, accurate tool credits,
   repository lifecycle checks, and the required video. A local Dockerfile is not a
   pullable fallback image. Corrected the unverified existing-repository assumption.

8. **Medium: overlapping solar semantics remain unspecified.** Minimum reserves and grid
   caps naturally intersect; solar factors might be interpreted differently by a judge.
   Retained a documented provisional minimum-factor policy and flagged organizer clarification.
   Passing tests under this policy does not verify the hidden judge's policy.

## Limits of this evidence

- The randomized generator creates feasible problems using an idle-battery witness. It
  covers decimal data, surplus solar, zero rates/capacity/tariffs, shuffled input hours and
  directive combinations; it does not prove every tight multi-hour feasibility case.
  Public cap/reserve cases add some forced scheduling coverage. More targeted production
  tests are specified in the revised plan.
- The oracle has a separate state-based formulation and binary action exclusivity but
  shares bound compilation with the prototype LP. Scalar replay independently derives
  bounds from raw directives to reduce that common-mode risk. Both use SciPy/HiGHS, so
  this is not independent solver-vendor verification or a formal numerical proof.
- The guardrail function is a prototype of the revised policy. HTTP request validation,
  provider transport, queue/deadline handling, logging and security are not implemented here.
- No held-out model evaluation, paid API calls, Docker build/run, deployment, external
  reachability, 100-request soak, or video review occurred. Those gates remain pending.
- Performance and winning odds cannot be inferred from these local checks. The strongest
  competition improvement is to prioritize the 50 interpretation/application points and
  retain deployment/schema/documentation points while keeping verified optimal scheduling.
