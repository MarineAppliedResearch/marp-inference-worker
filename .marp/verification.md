# Verification — MarineAppliedResearch/marp-inference-worker#10

## What each test proves

| Requirement | Test | Tier | Proves |
| --- | --- | --- | --- |
| R1, R2 | `test_tracking_engine_reports_phases_in_actual_work_order` | worker orchestration | A frame-range inference run reports `opening_video`, `loading_model`, `seeking`, `inferring`, `reducing`, and `publishing` in the order the engine performs those operations, with the half-open range total available before inference. |
| R2, R3 | `test_frame_reader_reports_seek_complete_at_the_real_boundary` | worker unit | The transition from `seeking` to `inferring` occurs after the actual capture seek and before the first frame read, without materializing or reordering frames. |
| R3 | `test_progress_phase_is_retained_and_each_transition_is_logged_once` | worker unit | Progress calls retain the current phase when only counters change and emit exactly one structured log event for each phase transition. |
| R1, R2, R3, R8 | `test_progress_reaches_the_coordinator_and_advances` and `test_a_finished_attempt_reports_the_progress_it_finished_on` | worker process/runner | A real child process sends a non-null total, phase, slot, elapsed time, and advancing count through the parent heartbeat; its final publication snapshot and structured phase events reach the coordinator client. |
| R4, R5 | `GPU heartbeat › extends the lease, records progress, and says continue` and `accepts a bounded worker-defined phase and rejects invalid progress metadata` | API HTTP + PostgreSQL | MARP_API accepts phase and elapsed seconds, accepts a future job-type phase without an enum release, bounds phase length, rejects invalid elapsed values, and persists the accepted fields. Existing heartbeat tests continue to send older partial progress shapes. |
| R3 | `GPU attempt events › accepts a batch, and treats a replay of it as duplicates rather than new events` | API HTTP + PostgreSQL | A structured phase-transition log is stored durably through the existing replay-safe event route. |
| R4 | `GPU pool view › shows the machine, its hardware, and the job it is running` | API HTTP + PostgreSQL | The worker-pool response returns the same latest phase and elapsed time stored on the live attempt. |
| R6 | `GPU heartbeat › refuses a worker that claims a stale lease epoch, and writes nothing` | API HTTP + PostgreSQL | A wrong lease cannot write done, phase, elapsed time, or the heartbeat timestamp. |
| R7 | `GPU attempt result › records a success as an artifact belonging to the job rather than to a training run` | API HTTP + PostgreSQL | Job detail retains the final `publishing` phase and elapsed time after the attempt becomes terminal. |
| R8 | Existing worker runner, tracking-pipeline, API orchestration, artifact, control-action, and result tests in the selected groups | process + HTTP + PostgreSQL | Counts, control actions, event batching, publication, and inference output continue to follow their prior contracts. |
| R9 | `scripts/verify_progress_reporting.py` | real cross-repository HTTP + process + PostgreSQL | The actual CoordinatorClient, JobRunner, child process, MARP_API routes, and disposable database agree on the live pre-frame total/phase and retain the terminal snapshot. |
| R10, R11 | Source-scope review against the issue diff | review | No dashboard code or `job_pressure` behavior changes under this issue. |

## Requirements with no test

None. R10 and R11 are scope constraints verified by the branch diffs rather than executable behavior.

## Edge cases

- **Pre-frame visibility:** the cross-repository job deliberately delays its first mock frame. The first accepted heartbeat must read `done = 0`, `total = end_frame - start_frame`, and a non-empty phase through job detail.
- **Lazy seeking:** a callback is observed between the capture seek and first read, so generator laziness cannot make `inferring` cover the seek.
- **Repeated progress in one phase:** it updates counters without producing duplicate durable transition events.
- **Future training vocabulary:** `training_validation` is accepted as a bounded free-form phase.
- **Old worker payload:** existing tests that omit phase and elapsed time continue to pass unchanged.
- **Bad metadata:** a 65-character phase and negative elapsed time are rejected as validation errors.
- **Stale lease:** all newly added snapshot fields remain null, along with the existing counter and timestamp.
- **Terminal history:** successful result reporting changes attempt state while retaining its last snapshot.

## Regression coverage

Worker focused checks:

```powershell
.\.venv312\Scripts\python -m pytest -p no:cacheprovider --basetemp .marp\local\pytest-10 tests/test_job_context.py tests/test_job_runner.py tests/test_tracking_pipeline.py
.\.venv312\Scripts\python -m ruff check src/marp_inference_worker/engines/base_engine.py src/marp_inference_worker/engines/mock_engine.py src/marp_inference_worker/engines/tracking_engine.py src/marp_inference_worker/jobs/child_main.py src/marp_inference_worker/jobs/context.py src/marp_inference_worker/jobs/job_process.py src/marp_inference_worker/media/frame_range_reader.py scripts/verify_progress_reporting.py tests/test_job_context.py tests/test_job_runner.py tests/test_tracking_pipeline.py
```

API migration and GPU subsystem checks run only against the harness-created disposable database:

```powershell
npx sequelize-cli db:migrate
npx sequelize-cli db:migrate:undo
npx sequelize-cli db:migrate
npm run test:gpu
npm run docs:api:build
```

The cross-repository check uses the API workspace selected by `marp agent list`. Start that workspace's API, create a disposable service token with `scripts/create-application-token.js` for `workers:enrol,jobs:execute,jobs:read,jobs:write`, set `MARP_COORDINATOR_URL` from the workspace's configured port and `MARP_WORKER_TOKEN` to the emitted token, then run:

```powershell
.\.venv312\Scripts\python scripts/verify_progress_reporting.py
```

The script submits a diagnostic mock job so it exercises the complete transport and persistence path without a GPU, Jellyfin, observation ingest, or production data.

## Known gaps

- The cross-repository run uses the real worker process and API but the `mock` engine, because the reporting path does not depend on a model or GPU. The tracking-engine orchestration test observes the real inference phase boundaries with lightweight detector/tracker collaborators.
- It does not measure how long a real model takes to load or a real media stream takes to seek. Those timings do not change the phase contract.
- The broad developer-docs generator currently reports unrelated pre-existing JSDoc parse errors. Verification regenerates and checks the OpenAPI artifact only; the unrelated source files stay untouched.
- Dashboard presentation and the inaccurate `job_pressure.active_jobs` placeholder remain assigned to later work by R10 and R11.
- Nothing connects to production PostgreSQL or the live Jellyfin service.

## Manual steps

None. Every requirement is observable through automated worker, HTTP, and disposable-database checks.

---

## Results

- **Worker focused suite — PASS.** `59 passed in 19.58s` across
  `test_job_context.py`, `test_job_runner.py`, and `test_tracking_pipeline.py`.
- **Boundary regression after lint cleanup — PASS.** The seek-boundary and real phase-order
  tests passed again: `2 passed, 16 deselected in 0.15s`.
- **Worker lint delta — PASS.** Ruff reports 13 findings in the changed-file set; the same
  files on `origin/develop` report 14. Every remaining finding is present on the base and
  this branch introduces none.
- **API migration — PASS.** Additive migration up, down, and up again all completed. Each
  integrity guard reported `gpu_job_attempts=0 | 5 foreign key(s) watched`, with no deleted,
  dereferenced, or orphaned rows.
- **API GPU subsystem — PASS.** `6 passed, 0 failed` suites and
  `109 passed, 0 failed, 0 skipped` assertions in 15.0 seconds.
- **Generated contract — PASS.** `npm run docs:api:build` completed and a second generation
  left `docs/openapi.generated.json` unchanged.
- **Cross-repository round trip — PASS.** A real CoordinatorClient, JobRunner, child
  process, MARP_API server, and disposable PostgreSQL database agreed on the live pre-frame
  snapshot and retained the terminal snapshot: `PASS: real worker client -> MARP_API ->
  PostgreSQL retained live and terminal progress for job 99`.
- **Scope review — PASS.** The implementation changes neither dashboard code nor
  `job_pressure` behavior.

Two first attempts to start pytest failed before collection with `PermissionError:
[WinError 5] Access is denied` while pytest created its base temporary directory under the
sandbox. The identical approved command passed after granting filesystem access; this was
an execution-environment failure, not a test failure.

The broad developer-doc generator also printed pre-existing JSDoc parse errors in unrelated
database and Mosaic source files. It exits successfully, but those diagnostics are not
treated as evidence for this change; the issue-specific OpenAPI generator passed and was
stable.
