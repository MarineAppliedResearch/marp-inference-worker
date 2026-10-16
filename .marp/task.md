---
task: MarineAppliedResearch/marp-inference-worker#67
repos: [marp-inference-worker, MARP_API]
status: design
needs: []
---

## Goal

A model is trained by giving the job to the worker pool, the same way inference is given to
it. Today training happens in `src/old_scripts/model_training_live.py`, on one person's
machine, against a dataset folder only that machine has, writing to the database directly.
Afterwards nobody can say which weights came from which data without asking the person who
ran it. When this is done, a training run is queued in MARP, leased by a worker, watched
while it runs, and ends in weights MARP has registered and the pool can run inference with.

## Requirements

- **R1** — A job spec can describe a training job. `video` and `range` are required today
  and a training job has neither; it has a dataset, starting weights and hyperparameters.
  An inference spec keeps validating exactly as it does now.
- **R2** — The worker fetches the dataset from a locator the job spec carries and verifies
  it against a hash before use, the same way a model artifact is fetched and verified. It
  does not know what is serving that locator and holds no credential for it.
- **R3** — Training runs behind the existing `BaseEngine` contract — `preflight()` then
  `run(ctx, spec)` — so the runner, the slots, the lease, the stop file and the pause file
  need no training-specific cases.
- **R4** — Progress and metrics are reported per epoch through `ctx.report_progress()` and
  `ctx.report_metrics()`, from Ultralytics' own epoch callbacks. The unit is epochs, not
  frames.
- **R5** — A training attempt stops and pauses at an epoch boundary, and a stop reports the
  epochs completed rather than losing them.
- **R6** — The trained weights are handed over with `ctx.publish_artifact()` and carry the
  class names they were trained on, because the tracking engine refuses weights whose
  vocabulary disagrees with what MARP registered for the model.
- **R7** — A dataset is cached once per worker and shared by every attempt that uses it.
  It is not copied into each attempt's workspace. See #65: per-attempt copies of shared
  bytes are what filled a volunteer's disk, and a dataset is far larger than a browser
  profile.
- **R8** — Nothing on the training path imports the Jellyfin client, extending the existing
  `test_nothing_on_the_job_path_imports_jellyfin` boundary.
- **R9** — One training job is one training run: a single `model.train()` over one dataset.
  There are no curriculum phases. See the decision below — the legacy script's
  crops/frames/mixed pipeline is an abandoned experiment and must not be carried forward
  from it.
- **R10** — A training job occupies the machine. While one runs, the worker offers no
  further slots, whatever `_live_slot_count()` would otherwise derive from throughput.

## Open assumptions

- [x] **A1 · architectural · blocking** — answered 2026-10-15: MARP serves the dataset. The
      job spec carries a locator the worker fetches and verifies, exactly as `video.url`
      already works. marp-api serves it itself for now; later the API writes datasets to a
      file server and serves them from there, and because the worker only ever sees a
      locator, that change is MARP's alone. The file server does not exist yet and nothing
      here waits for it.
- [x] **A2 · architectural · blocking** — answered 2026-10-15: one job, one training run.
      The question assumed phases and there are none.
- [x] **A3 · scientific · blocking** — answered 2026-10-15: not applicable. crops/frames/
      mixed was an attempt at curriculum learning, it did not work well, and it is not what
      we do. The order the legacy code and its own docstring disagree about is therefore
      moot rather than a thing to resolve.
- [ ] **A4 · behavioural · blocking** — If a machine dies three hours into a four-hour run,
      is that work lost? The lease itself is fine — the parent process heartbeats every ten
      seconds for any running job regardless of what the engine is doing, so a long job
      holds its lease as long as the worker is alive. This is only about a crash, a reboot,
      or a volunteer closing the lid. Ultralytics already writes `last.pt` after every
      epoch and can resume from it, and `ctx.resume_from` exists for exactly this and is
      hard-coded to `None` today. So: publish a checkpoint periodically so a retry can pick
      it up, or accept that an interrupted training run starts again?
- [ ] **A5 · cross-repository · blocking** — When does the worker hand the weights over:
      every time training finds a new best, or once at the end? Every new best costs an
      upload per improvement but means a long run is never wholly lost and MARP can show
      the best model so far; once at the end is one upload and nothing until it finishes.
      Related to A4 and may be answered by the same choice.
- [x] **A6 · performance/concurrency · non-blocking** — answered 2026-10-15: a training job
      takes what it can, and the machine is not expected to have slots left over. R10.
- [ ] **A7 · environment · non-blocking** — Where does the shared dataset cache live, and
      what evicts it? Model weights cache under the install's `models/cache`; a dataset is
      far larger and the same choice may not suit.

## Decisions

- **2026-10-15** — The spec is written against the existing `BaseEngine` contract rather
  than a second job path. `report_metrics` is already unschema'd for Ultralytics' varying
  metric names, and `resume_from` was documented for training from the start, so the
  contract was built expecting this.
- **2026-10-15** — **No curriculum phases.** The legacy pipeline trains in three phases —
  crops, then frames, then mixed — passing each phase's best weights to the next. That was
  an experiment in curriculum learning, it did not work well, and training is now a single
  ordinary run. This is written down because `model_training_live.py` still contains the
  whole pipeline and reads like the intended design: it even disagrees with itself about
  the order, with `runTrainingPipeline`'s docstring saying crops/frames/mixed and the line
  below it running `["crops", "mixed", "frames"]`. Anyone porting from that file will find
  the phases first. Promote to a decision record at G2, when edits outside `.marp/` are
  unblocked.
- **2026-10-15** — The dataset reaches the worker as a locator plus a hash, and the worker
  never learns what serves it. This is the same boundary A8 drew for video, and it is what
  lets the file server arrive later without the worker changing.

## Plan

Written once A4 and A5 are answered; both change what the engine publishes and when.

## Acceptance criteria

- A training job spec validates, and an inference job spec is unaffected.
- A training attempt reports epochs as they finish and is visible while it runs.
- A stop part-way through reports the epochs completed, not a failure.
- The weights land in MARP as an artifact that can be registered and then run.
- A second attempt on the same worker reuses the cached dataset rather than fetching it
  again.
- While a training job runs, the worker offers no further slots.

## Test plan

Filled in at G3, after the gate.

## Status

- **Gate:** design
- **Also on this branch:** the worker-identity fix, specified in
  `.marp/task-worker-identity.md` and already implemented. One branch carries the
  work stream rather than one branch per task.
- **Notes:** G1. A1, A2, A3 and A6 answered 2026-10-15. A4 and A5 are still with Isaac and
  are related — both are about when work is handed over. Nothing implemented. The
  coordinator half is `MarineAppliedResearch/MARP_API#265`; the tracking issue is
  `MarineAppliedResearch/MARP#36`.
