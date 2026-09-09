---
task: MarineAppliedResearch/marp-inference-worker#3
repos: [marp-inference-worker]
status: design
needs: []
---

## Goal

A GPU machine anywhere — in the office or on someone's home connection — can be pointed at
MARP with a single pasted token and will start taking work. It asks MARP for a job, runs
it, reports how far along it is, stops when told to stop, and hands the result back. It
never waits to be contacted, because a machine behind a home router cannot be.

This task covers only the worker's side of that loop, against the coordinator route family
being built in parallel (`MarineAppliedResearch/MARP_API`, same issue). Milestone 1's
workload is **inference over a Jellyfin video**; training is Milestone 2 and is
deliberately out of scope here.

## Requirements

- **R1** — The worker holds no inbound listener for MARP. All coordinator traffic is
  outbound from the worker. Its existing FastAPI app binds to `127.0.0.1` only, and serves
  the machine's own operator: status, and pause/resume of taking new work.
- **R2** — A worker is configured by one service token and nothing else. Hardware, GPU
  count, VRAM, driver and disk are discovered, not configured.
- **R3** — The worker has a persistent identity that survives restart, and re-enrols
  itself if its record is unknown to the coordinator.
- **R4** — The worker long-polls for work, falling back to interval polling. When it is
  given a job it leases it, and holds that lease by heartbeat.
- **R5** — Cancel, pause and abandon arrive **in the heartbeat response**. The worker
  acts on them within one heartbeat interval and never needs to be reachable.
- **R6** — One child process per job, pinned to one GPU slot. Worker identity is
  host-level; the host owns disk, bandwidth and how many jobs run at once.
- **R7** — A job's engine is invoked through one contract: `run(ctx, spec)`, where `ctx`
  offers exactly `params`, `log`, `report_progress(done, total, unit)`,
  `report_metrics(step, phase, mapping)`, `checkpoint_dir` / `resume_from` /
  `publish_artifact`, and `should_stop()`. The engine learns nothing about MARP, tokens or
  Jellyfin.
- **R8** — Inference runs as a generator, one frame at a time. Nothing accumulates the
  whole video in memory.
- **R9** — **A video is divisible across workers.** A job may cover a frame range of a
  video, so a ten-minute video can be split into one-minute pieces and a ten-hour video
  into ten hourly pieces, run on as many workers as are free. The job spec therefore
  carries a range, not just a video.
- **R10** — The MARP tracking pipeline is detect → track → reduce, not detect alone. The
  engine contract must express a stateful multi-stage pass over a frame range: YOLO per
  frame, `BYTETracker` assigning track ids, and a finished track reduced to keyframes
  labelled start/middle/end, as `object_tracking_live.py` does today.
- **R10a** — A range boundary is a seam and is left as one. A track ends at the end of its
  range; the next range starts a new track if the animal is still there. Ranges do not
  overlap, tracks are not stitched, and track ids are never compared across ranges. Two
  observations for one animal crossing a seam is the accepted outcome.
- **R10b** — The keyframe reduction is a **named, versioned step**, and which version
  produced an observation is recorded with it. Milestone 1 ports
  `reduce_to_keyframes_v3_dirpad` (the line-814 definition) verbatim; the reduction is
  expected to change, and a stored observation must stay attributable to the rule that
  made it.
- **R11** — Results are written to a file, hashed, and handed to the coordinator by that
  hash. Per-frame detections are never sent inline.
- **R12** — `/status` reports what is actually true: identity, slots, current jobs,
  hardware, engine list, versions. The present hard-coded literal is deleted.
- **R13** — A model file fetched by the worker has its sha256 **verified** before use. The
  existing code computes a hash and never checks it.
- **R14** — A job that cannot run on this machine fails immediately and says why, rather
  than starting and dying. `device="auto"` never silently falls back to CPU.
- **R15** — The worker survives its own restart: a job in flight is reported as lost, not
  silently forgotten, and the worker resumes taking work.
- **R16** — Ultralytics' runtime behaviour is controlled by the worker, not the engine:
  `YOLO_OFFLINE`, `YOLO_AUTOINSTALL`, `YOLO_CONFIG_DIR` are set per job, each job gets its
  own workspace, and the version is pinned exactly.

## Open assumptions

- [x] **A1 · api contract · blocking** — answered 2026-09-09 in the MARP_API task's
  decisions. Five worker-facing routes under `/api/v2/gpu/…`: `workers/enrol`, `poll`
  (long-poll, returns 204 when idle), `attempts/:id/heartbeat` (progress up,
  `{action: continue|cancel|pause|abandon}` down), `attempts/:id/events` (batched, keyed by
  `seq`), `attempts/:id/result` (idempotent). Artifacts hand off in two steps —
  `artifacts/check` by sha256, answering `already_have` or an upload target — outside
  `bodyParser`. Every state-changing call carries `(attempt_id, worker_id, lease_epoch)`
  and a mismatch is answered `abandon`. The job spec carries `engine`, `model{name,sha256}`,
  `video{jellyfin_item_id, source_name}` — **the video clause is superseded by A8: it is now
  `video{url, source_name, jellyfin_item_id?}`** — `range{start_frame, end_frame}`, `params`, and
  `reduction{name, version}`; `range` is always present so nothing special-cases a whole
  video.
- [x] **A2 · environment · blocking** — answered 2026-09-09: a fresh GPU worker runs a
  **Python script that loads the engines**. **ByteTrack is required, not optional** — the
  live tracking pipeline imports `BYTETracker` from `yolox.tracker.byte_tracker`, so its
  compiler-needing dependencies (`lap`, `cython-bbox`) are part of the install and need
  prebuilt wheels or a documented build step on Windows. Only the 54 MB of demo assets are
  removable. The broken checked-in `.venv` still needs recreating and the conflicting
  version claims reconciling, but neither changes the shape of this task.
- [ ] **A3 · architectural · non-blocking** — engine environments. Ultralytics pins
  `torch>=1.8.0` with no ceiling and its `[export]` extras pin `numpy<2`, so one
  environment cannot serve every engine indefinitely. Research recommends a venv per engine
  environment, named by contents. Confirm, or accept a single pinned environment for
  Milestone 1 and defer the split.
- [ ] **A4 · behavioural · non-blocking** — how many jobs may a host run at once, and is
  that set per machine or dictated by the coordinator? Proposed: one job per GPU slot,
  worker-configured, reported in its capabilities.
- [ ] **A5 · behavioural · non-blocking** — what a paused worker does with a job already
  running. Proposed: pause stops it taking *new* work and leaves the running job alone;
  abandoning a running job is a separate action.
- [x] **A6 · scientific/data-meaning · blocking** — answered 2026-09-09 by tracing the
  code. **`reduce_to_keyframes_v3_dirpad` is authoritative**, and specifically the
  definition at line 814 rather than the one at 628: Python keeps the later definition and
  the two differ (`vel_window` 3 vs 2, `speed_gain` 0.25 vs 0.35). Likewise
  `_apply_directional_pad` at line 930 wins over 780. The apparent second call site at line
  296 is **dead code** — it sits inside a triple-quoted block opened at line 124, an older
  `process_video` commented out by wrapping it in quotes, which is why it appeared to be
  inside `parse_timecode()`. There is one live path, at line 468. Port the live pair
  verbatim rather than merging variants; and because reduction ideas are expected to change,
  the reduction belongs behind a named, versioned step rather than an inline function.
- [x] **A7 · scientific/data-meaning · blocking** — answered 2026-09-09: **a track ends at
  the end of its range.** The worker holding the next range starts a fresh track if the
  animal is still there. No overlap, no stitching, and track ids are never compared across
  ranges. Two observations for one animal crossing a seam is the accepted outcome.

- [x] **A8 · api contract · blocking** — answered 2026-09-09 by Isaac, and it **reverses part
  of A1**: **the coordinator resolves the video and the job spec carries a playable URL.**
  The worker is not to know what Jellyfin is. It does not search, does not score a filename
  match, and holds no media credential — it is handed a source it can open and opens it.
  This also settles the wider point: a worker can process **any** reachable source, not only
  a Jellyfin item, because a URL is all the contract carries.

  Consequences, all of which are work: `media/jellyfin_client.py` and
  `media/video_source_resolver.py` come off the job path entirely (they stay for the legacy
  dataset scripts that still import them); `JobRunner`'s `video_resolver` argument and
  `_resolve_video` are deleted rather than fixed; the spec's `video` field carries the URL as
  a required value instead of `jellyfin_item_id` being mandatory; and the three `JELLYFIN_*`
  variables stop being worker configuration.

  Recorded against the earlier decision it supersedes: *"The worker is handed a Jellyfin
  credential outright."* That is no longer the design.
- [x] **A9 · security/permissions · blocking** — **a Jellyfin stream URL carries its own
  credential**, so A8 cannot be implemented without deciding what is handed over. A
  direct-play URL embeds an api_key; putting it in the job spec stores a media credential in
  the database, returns it in `GET /gpu/jobs/:id` to anything holding `jobs:read`, and hands
  it to every machine that leases the job — **including a volunteer's, which is the stated end
  goal.** A leaked URL is read access to the media library for as long as the key lives.

  Three shapes, needing a choice: a **short-lived token per attempt**, minted at lease time
  and expiring with the lease, so a leaked URL dies quickly; the **coordinator proxying the
  bytes**, so no media credential ever leaves MARP and the worker sees only a MARP URL
  carrying its own worker token; or **accepting the exposure** for office machines and
  revisiting before any outside machine runs.

  Note this also forces *when* resolution happens: a URL with an expiring token cannot be
  resolved at submit time and left sitting in a queue, so it has to be resolved **at lease
  time**, which is a change to the poll handler rather than to job creation.

  **Answered 2026-09-09 by Isaac: accept the exposure for now and revisit later.** So the
  coordinator may hand over a URL with a long-lived media key embedded, and that key is
  readable by anything holding `jobs:read` and by every machine that leases a job.

  The condition to revisit on is not a date, it is an event: **this is only acceptable while
  every worker is a machine MARP controls.** The first machine outside the office — the
  stated end goal of the whole design — makes it a media-library leak. Whoever enrols that
  machine has to answer this first, and resolution moving to lease time is the change that
  buys the short-lived token, so the poll handler is where it lands when it does.


## Decisions

Carried in from the investigation (`marp-inference-worker#3`, comments of 9 Sep) and not
re-opened here:

- The worker dials out; nothing connects to it. Push is to be **unrepresentable** — no
  host, url or port field anywhere in the contract.
- There is no "volunteer" concept. A worker is a worker; who owns the machine is not part
  of the design. No trust tiers, no result validation, no scoped media credentials.
- ~~The worker is handed a Jellyfin credential outright.~~ **Superseded by A8, 2026-09-09.**
  The coordinator resolves the video and hands the worker a playable url; the worker holds
  no media credential and knows nothing about Jellyfin.
- Multi-GPU jobs are supported. Where an engine's own callbacks are unavailable — as with
  Ultralytics' multi-GPU re-exec — progress is read from the run's `results.csv`.
- A job runs on one worker start to finish. No cross-machine resume.
- Inference first. Training, datasets and checkpoint/resume are Milestone 2.
- **Tracking already reaches MARP and there is a working precedent to preserve.**
  `object_tracking_live.py` posts an observation per finished track to `/api/observation`
  with its reduced keyframes embedded: `session_id`, `comname`, `taxserial`, `count`, `tc`,
  `frame` (the sub-second index, `chosen_frame % frame_rate`), `video_source`,
  `videoLocation`, `mediaPosition`, `actualPosition`, `keyframes[]`. The `# TODO: send obs
  + keyframes to DB` comment above it is stale — the write is seventeen lines below it.
  A tracking job's output must be expressible in those terms.
- Ultralytics licensing is deferred; it does not block this work.

## Plan

1. Replace `engines/base_engine.py` with the `run(ctx, spec)` contract and the six `ctx`
   calls. Delete the `hasattr` reach-throughs at `models/model_manager.py:124,158`.
2. Add the job runner: lease, child process per job, heartbeat, cancellation, terminal
   report. This layer does not exist today in any form.
3. Rewrite `models/model_manager.py` for per-process ownership with an unload path, and fix
   `models/model_cache.py` to verify its hash and to fetch with a timeout and resume.
4. Replace `api/status_routes.py` with a real `describe()`, and bind the app to loopback.
5. Extend inference to `infer_stream(source, params) -> Iterator[FrameResult]`; single
   frame becomes `next(...)`.
6. Wire `UltralyticsEngine` onto the same six calls, with the per-job environment controls.
7. Delete `src/old_scripts/` once its progress-field lists are captured in the contract.
   Keep `ByteTrack` — it is required — but strip its demo assets and pin its build
   dependencies.
8. Port the detect → track → reduce pipeline from `legacy/object_tracking_live.py` behind
   the engine contract, carrying over the keyframe-reduction code as-is rather than
   reinterpreting it.

## Acceptance criteria

Milestone 1, worker side:

- A fresh GPU machine, given only a token, appears in MARP with its real hardware.
- It leases an inference job over a Jellyfin video, runs it, and reports progress that
  advances.
- Cancel from MARP stops it within one heartbeat.
- The result arrives as a hashed file and its hash verifies.
- Killing and restarting the worker mid-job leaves MARP able to re-queue the job, and the
  worker takes work again without re-configuration.
- The worker refuses a job it cannot run, with a reason, before starting it.

## Test plan

Written at G3, after the spec settles. Tiers available here are parse/unit (fast, the
working loop) and a real-GPU tier that cannot run in CI. Note the platform doctrine: a
defect is not fixed until a test at a tier that can *observe* it exists — a unit test
cannot see a GPU job being cancelled.

## Status

Design. Blocked on A1 and A2 by the harness's own gate: nothing outside `.marp/` is
implemented while a blocking assumption is unticked.
