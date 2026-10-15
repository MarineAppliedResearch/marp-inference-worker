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

## Open assumptions

- [ ] **A1 · architectural · blocking** — How does the dataset reach the worker? The
      proposal is the A8 shape: MARP resolves the dataset and the job spec carries a
      locator plus a hash, exactly as `video.url` does, so the worker never learns where
      datasets live and the coming file server is a change on MARP's side only. The
      alternatives are the worker being given per-frame URLs and labels and assembling the
      YOLO tree itself, or dataset building becoming its own job kind. Which?
- [ ] **A2 · architectural · blocking** — One training job that runs all three phases, or
      one job per phase with the coordinator chaining them and passing each phase's weights
      to the next? Chaining falls out of the existing lease, retry and resume machinery;
      one long job does not.
- [ ] **A3 · scientific · blocking** — What is the phase order?
      `runTrainingPipeline`'s own docstring says *"1. crops 2. frames 3. mixed"* and the
      line below it runs `phases = ["crops", "mixed", "frames"]`. The two have disagreed
      inside one function for as long as the file has existed, so neither can be taken as
      the intended one.
- [ ] **A4 · performance/concurrency · blocking** — A training run is hours and the lease
      is built for minutes: heartbeat every 10s, and a quiet attempt is reclaimed. Does a
      training attempt hold its lease for the whole run, or publish a checkpoint and resume
      after a restart? `ctx.resume_from` exists for the second and is hard-coded to `None`
      today.
- [ ] **A5 · cross-repository · blocking** — Who turns finished weights into an `ml_models`
      row with its `model_species`, and when? The worker can only publish an artifact; the
      registration is MARP's (`MarineAppliedResearch/MARP_API#265`).
- [ ] **A6 · performance/concurrency · non-blocking** — Does a training job take the whole machine, rather
      than one of the slots `_live_slot_count()` derived? That function measures frames per
      second and a training job reports epochs, so it would read a healthy run as a stalled
      one.
- [ ] **A7 · environment · non-blocking** — Where does the shared dataset cache live, and what evicts it?
      Model weights cache under the install's `models/cache`; a dataset is far larger and
      the same choice may not suit.

## Decisions

- **2026-10-15** — The spec is written against the existing `BaseEngine` contract rather
  than a second job path. `report_metrics` is already unschema'd for Ultralytics' varying
  metric names, and `resume_from` was documented for training from the start, so the
  contract was built expecting this.

## Plan

Written after A1–A5 are answered; the shape of every step depends on them.

## Acceptance criteria

- A training job spec validates, and an inference job spec is unaffected.
- A training attempt reports epochs as they finish and is visible while it runs.
- A stop part-way through reports the epochs completed, not a failure.
- The weights land in MARP as an artifact that can be registered and then run.
- A second attempt on the same worker reuses the cached dataset rather than fetching it
  again.

## Test plan

Filled in at G3, after the gate.

## Status

- **Gate:** design
- **Notes:** G1. Blocking assumptions A1–A5 are with Isaac; nothing is implemented until
  they are answered. The coordinator half is `MarineAppliedResearch/MARP_API#265` and the
  cross-repository tracking issue is `MarineAppliedResearch/MARP#36`.
