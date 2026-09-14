---
task: MarineAppliedResearch/marp-inference-worker#10
repos: [marp-inference-worker, marp-api]
status: design
needs: []
---

## Goal

While a distributed GPU attempt is running, MARP shows what stage the worker is in and a
usable frame proportion. An operator can distinguish startup, model/video preparation,
inference, reduction, and result publication instead of seeing a motionless counter with no
total and guessing whether the job is slow or stuck.

## Requirements

- **R1** — A leased frame-range attempt has a non-null total before its first processed
  frame. The total is the half-open range length, `end_frame - start_frame`.
- **R2** — The worker's live progress snapshot includes a phase that describes the work it
  is performing, including preparation, model/video setup, seeking, inference, reduction,
  and publication.
- **R3** — Phase changes travel through the existing child-process event channel and
  heartbeat request. No inbound connection, new worker endpoint, or second reporting loop is
  added.
- **R4** — MARP_API accepts an optional phase from new workers, persists the latest value on
  `gpu_job_attempts`, and returns it in both job detail and worker-pool views.
- **R5** — Compatibility is additive: an older worker that omits phase continues to
  heartbeat, while a newer worker remains usable with a coordinator that ignores the extra
  field.
- **R6** — A refused heartbeat with the wrong worker or lease epoch writes no total, phase,
  elapsed time, state, or heartbeat timestamp.
- **R7** — Terminal attempts retain their last accepted progress snapshot so history can
  explain where work ended.
- **R8** — Existing frame counts, attempt states, control actions, event batching, result
  publication, and inference output are unchanged.
- **R9** — The change is verified across the real worker HTTP client, MARP_API, and a
  disposable PostgreSQL database. A test must observe the phase and total through the API,
  rather than only inspecting an in-memory worker object.
- **R10** — This issue supplies data for the ML dashboard but does not implement or redesign
  the dashboard.
- **R11** — The inaccurate `job_pressure.active_jobs = 0` placeholder is recorded as an
  adjacent defect and left outside this issue.

## Open assumptions

- [ ] **A1 · api contract/behavioural · blocking** — Which exact phase vocabulary should be
  published? The implementation needs stable strings that tests, stored rows, and the
  dashboard can share. The code's real stages support `starting`, `opening_video`,
  `loading_model`, `seeking`, `inferring`, `reducing`, and `publishing`. A proposed
  `finishing` phase would require one extra heartbeat after artifact upload just before the
  terminal result, and may be too brief to be useful. **Recommendation:** use the seven real
  stages above, omit `finishing`, and preserve the engine's actual order instead of
  rearranging work to match a label list.
- [ ] **A2 · database/schema · blocking** — Should MARP_API persist the worker's existing
  `elapsed_s` alongside phase? The worker already sends it and MARP_API currently discards
  it. Persisting it as `progress_elapsed_s` would let a dashboard show average throughput
  after reload; phase plus `last_heartbeat_at` alone says what is alive but not how long the
  attempt has spent getting there. **Recommendation:** add it in the same additive migration
  as `progress_phase`, because this issue's purpose is distinguishing slow from stuck and
  the data already crosses the wire.
- [ ] **A3 · api contract/audit · blocking** — Should every phase transition also be durable
  in `gpu_job_events`, or only the latest phase be kept on the attempt? **Recommendation:**
  emit one ordinary structured `log` event per transition, using the existing event kind and
  table. Seven small events per attempt preserve the history without adding an event kind or
  schema.
- [ ] **A4 · api contract · blocking** — Should MARP_API enforce the worker's phase
  vocabulary? A closed enum makes typographical errors fail but forces worker and
  coordinator releases to move together when training adds new phases.
  **Recommendation:** validate a non-empty string with a short length limit, store it as
  `varchar`, and let the dashboard display an unknown future phase as text.
- [x] **A5 · cross-repository · blocking** — settled by inspection 2026-09-13: this is one
  cross-repository change. The worker produces the phase; MARP_API owns the heartbeat
  contract, migration, persistence, and query shapes. Neither half alone satisfies #10.
- [x] **A6 · product/UI · blocking** — settled by issue #10 and MARP_API#104: dashboard
  rendering remains in the dashboard implementation. This issue ends when its API data is
  correct and observable.
- [x] **A7 · scope · blocking** — settled by repository rules: do not repair the unrelated
  `job_pressure` placeholder while touching progress. Report it and leave it for its own
  issue.

## Decisions

- **2026-09-13** — Keep the worker-pull architecture and existing heartbeat channel.
- **2026-09-13** — Initialize the total from the job's required half-open range; probing the
  media container is not needed to know the leased piece size.
- **2026-09-13** — Store only current progress on the attempt. Any transition history uses
  the existing append-only event stream.
- **2026-09-13** — The worker issue is the coordinating issue for both repositories; API
  commits and the pull request reference it in full.

## Plan

1. Settle A1–A4 and record the shared contract in both task branches.
2. Initialize worker progress from the leased range and carry phase through child events,
   parent state, local status, in-flight recovery, and heartbeat payloads.
3. Mark the actual tracking and publication boundaries without changing their order.
4. Add an additive MARP_API migration and model fields for the accepted live snapshot.
5. Validate and persist the optional heartbeat fields, expose them through job and worker
   reads, and regenerate the API and developer documentation.
6. Write the G3 cross-repository verification plan, including a real worker/API/database
   round trip, for human approval before running it.

## Acceptance criteria

- Before the first detection, a running attempt read from MARP_API has a total and a
  meaningful current phase.
- During a real piece, the phase advances through the stages the worker actually enters and
  the frame count advances during inference.
- After terminal reporting, the attempt retains its final accepted snapshot.
- Old heartbeat bodies remain valid and invalid lease holders remain unable to modify the
  attempt.
- The worker, API, and disposable database agree on the same phase and total end to end.

## Test plan

Written at G3 after A1–A4 are answered. The focused worker runner tests and MARP_API GPU
group will cover each repository; the end-to-end tier must run the real worker client
against the isolated API and its disposable PostgreSQL database.

## Status

- **Gate:** design. G1 is blocked on A1–A4.
- **Notes:** The worker already sends frame progress and `elapsed_s`, but initializes total
  to null and carries no phase. MARP_API accepts only `done`, `total`, and `unit`,
  discarding other progress keys. The existing event stream already stores structured
  metrics, while worker-pool and job-detail queries already expose the current attempt row.
  The API workspace is isolated because this change requires a migration and another
  checkout has unrelated work in progress.
