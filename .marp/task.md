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
- **R11** — The worker publishes its best weights while the run is in progress: on every
  new best, never more than once every ten minutes, and unconditionally once at the end.
  A run that dies loses only the epochs since the last publish.
- **R12** — After unpacking, the worker rewrites the dataset YAML's `train` and `val`
  paths to point at where it actually put the files. A dataset built by the legacy scripts
  records absolute paths on the machine that built it, so shipped as-is it names
  directories the worker does not have.
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
- [x] **A4 · behavioural · blocking** — answered 2026-10-15: the worker publishes its
      best weights as it goes, so an interrupted run is never wholly lost. Recovery is
      "start again from the last published best" rather than a true optimizer-state
      resume; see the decision below for why that is enough for now. `ctx.resume_from`
      stays unused.
- [x] **A5 · cross-repository · blocking** — answered 2026-10-15: on every new best,
      rate-limited, plus an unconditional publish at the end. Checkpoints go up under the
      artifact role `checkpoint`; the final one goes up as `weights` and is what MARP
      registers as a model.
- [x] **A6 · performance/concurrency · non-blocking** — answered 2026-10-15: a training job
      takes what it can, and the machine is not expected to have slots left over. R10.
- [ ] **A7 · environment · non-blocking** — Where does the shared dataset cache live, and
      what evicts it? Sharper now that the first real dataset is measured: rockfish5 is
      4.7 GB unpacked and Star4 is 10.7 GB, so two or three datasets on one volunteer
      machine is tens of gigabytes that nothing currently removes. Same shape as #65,
      and R7 only says the cache is shared, not what bounds it.
- [ ] **A8 · security/permissions · blocking** — Where does the 4.9 GB dataset fetch
      actually happen, and how is it authorized? Found in G2, wiring the engine up. The
      parent runner fetches the model today (`runner.py` `_prepare_params`) and hands the
      child a local path, which keeps the service token out of the child entirely. A
      dataset cannot simply follow it: the fetch is minutes rather than milliseconds, and
      `_prepare_params` runs on the runner's own loop, so a long fetch stops the
      heartbeats for every running job on the machine and can lose this attempt's own
      lease before it starts. Three ways out, and they trade the same thing differently:
      (a) the child fetches, with the parent passing the resolved URL and bearer header in
      `envelope.json` — simple, cancellable, and it can report progress, but the worker's
      service token lands in a job workspace, and #65 established that workspaces are
      never deleted; (b) as (a), but `child_main` overwrites the envelope as soon as it
      has read it, bounding the exposure to seconds; (c) the parent fetches on a
      background thread and launches only when it completes — the token never leaves the
      parent, at the cost of a "preparing" state the runner does not have today.
      Recommendation: (b).


## Decisions

- **2026-10-15** — **Publish on every new best, not every epoch, and never faster than
  once every ten minutes.** Isaac asked which is better; this is the reasoning. The two
  reasons to send anything are *don't lose the run* and *let MARP see the model*, and only
  a new best serves the second — an epoch that did not improve produces weights nobody
  would ever register or run. Every epoch is also expensive in a way that matters here:
  the weights are around 130 MB, a run is often a couple of hundred epochs, and that is
  tens of gigabytes up a volunteer's home connection for files superseded minutes later.
  A new best fires often early and rarely later, which is the right shape — it is exactly
  when there is something new worth having. The rate limit exists because early epochs can
  improve on consecutive passes, and the end-of-run publish is unconditional so the final
  best is never the one the limiter skipped.
- **2026-10-15** — **Recovery is restarting from the last published best, not a true
  resume.** Ultralytics resumes from `last.pt`, which carries optimizer and scheduler
  state that `best.pt` does not, so a real resume means publishing a second and larger
  file on a timer for a case that has not happened yet. Restarting from the last best
  keeps everything the run has learned and loses only the epochs since — acceptable, and
  far less machinery. Revisit when somebody actually loses a long run.
- **2026-10-15** — **A checkpoint and the finished model are different artifact roles.**
  `publish_artifact(path, "checkpoint")` while running, `publish_artifact(path, "weights")`
  at the end. MARP registers the second as an `ml_models` row; the first is attached to the
  training run and is not a model anybody is asked to use.

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

## Decisions (continued)

- **2026-10-19** — **The first end-to-end test runs on `rockfish5`.** Its training portion
  is 4.7 GB: 9,879 train images, 1,966 eval, 11 rockfish and groundfish classes. `Star4` is
  the same shape at 10.7 GB and is the obvious second. Two directories are excluded from
  the archive: `for_human_review/`, which is not training data, and `crops/`, which exists
  only for the curriculum phases we are not doing.
- **2026-10-19** — **`frames-classnames.yaml` is the one to use.** rockfish5 carries three
  — `crops-`, `frames-` and `mixed-` — because it was built for the three-phase pipeline.
  `frames-` is train plus eval and nothing else, which is exactly one ordinary training
  run; `mixed-` adds the crops tree. Picking `frames-` is what makes R9 true of a dataset
  that was built before R9 was decided.
- **2026-10-19** — **The archive lives in `MARP_API/storage/` for now**, beside
  `gpu-artifacts/`, and that is explicitly temporary. The worker only ever sees a locator
  and a hash, so when the file server in `MarineAppliedResearch/MARP_API#120` takes over,
  nothing in the worker changes and nothing in this spec needs revisiting. That is the
  whole reason the dataset is addressed by locator rather than by path.

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
