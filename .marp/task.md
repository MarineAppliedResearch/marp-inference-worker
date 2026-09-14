---
task: MarineAppliedResearch/marp-inference-worker#11
repos: [marp-inference-worker, marp-video-player]
status: verifying
needs: []
---

## Goal

An inference job may open a local screensaver-style MARP window showing the exact frames
being processed, with the model's detections and tracks drawn as they happen.

## Requirements

- **R1** — A job with `params.watch: true` opens a local window by default. The worker may
  explicitly use `--screen off` to suppress it or `--screen fullscreen` to start fullscreen.
- **R2** — The local display uses marp-video-player and accepts frames already decoded by
  inference rather than decoding the source again.
- **R3** — Every produced frame is presented in order, without sampling, dropping, or
  source-rate pacing. The browser acknowledges the synchronous canvas draw rather than a
  visibility-dependent animation callback, so covered fullscreen windows keep consuming frames.
- **R4** — Each species has one stable box/label colour. The top label is centered over
  and sized to its box and contains only the species name. A centered bottom label shows
  confidence while retaining track id and persistence.
- **R5** — A quiet status area shows frame number, achieved rate, and live-track count. A
  read-only timeline shows progress through the assigned frame range without pause or scrub input.
  A translucent context strip shows the model name, MARP job number, and model class list.
- **R6** — The display and frame channel bind only to loopback and expose no credential or
  remote media URL.
- **R7** — Closing or failing the window records one warning and continues inference
  headless.
- **R8** — The job child owns and closes its display resources.
- **R9** — Watch mode changes no scientific output. With display disabled it performs no
  encoding, browser, or frame-channel work.
- **R10** — The job option is an additive boolean engine parameter and requires no API or
  database change.
- **R11** — Existing marp-video-player APIs and offline host behavior remain compatible.
- **R12** — Escape leaves fullscreen for the normal app window; the window close control
  closes the display and continues inference headless. Pause and scrubbing are unsupported.
- **R13** — Concurrent watched jobs open visibly distinct, independent windows; audio is
  muted, and an occluded window or a window returning from display sleep continues
  presenting while another is in front.
- **R14** — Installer, dependency distribution, self-update, activation, volunteer compute
  controls, and remote viewing are separate work.

## Open assumptions

- [x] **A1 · product/UI · blocking** — answered 2026-09-14: machine policy is
  `--screen off|window|fullscreen`, default window, and the job independently requests watch.
- [x] **A2 · performance · blocking** — answered 2026-09-14: draw every frame in order as
  fast as inference produces it; accepted display cost is measured rather than hidden.
- [x] **A3 · environment · blocking** — corrected 2026-09-14: issue #11 uses a browser and
  player build supplied by the development checkout. Packaging them is later work with an
  explicit small-download design.
- [x] **A4 · product/UI · blocking** — answered 2026-09-14: fullscreen, Escape, and close;
  no pause or scrubbing, and closing never stops inference.
- [x] **A5 · behavioural · blocking** — answered 2026-09-14: one independent window per
  concurrently watched job, with no audio.
- [x] **A6 · architectural · blocking** — settled 2026-09-14: the job child owns a local
  loopback channel and receives already-decoded frames inline.

## Decisions

- **2026-09-14** — Use marp-video-player's presentation surface.
- **2026-09-14** — Keep machine and job display gates independent.
- **2026-09-14** — Keep release packaging outside #11 after the first CUDA bundle measured
  3,058,143,651 bytes and demonstrated that bundling the full ML runtime is unacceptable.
- **2026-09-14** — Use species identity for annotation colour, split the label above and
  below its box, cascade concurrent windows, and let Chromium return Escape to windowed mode.

## Plan

1. Add the player's external live-frame surface without changing existing media playback.
2. Add worker launch and job gates.
3. Feed ordered annotated frames from the tracking loop to a child-owned loopback display.
4. Verify rendering, failure-to-headless behavior, scientific-output equivalence, and cost.

## Acceptance criteria

- A watched job shows every processed frame with readable species and track information.
- Either gate being off produces no display work.
- Closing or breaking the display leaves the job and its output intact.
- The same range produces identical observations with display on and off, with throughput
  recorded for both.

## Test plan

See `.marp/verification.md`.

## Status

- **Gate:** verifying.
- **Notes:** Watch implementation exists in both repositories. Packaging work was removed
  from this issue and retained only on local backup branches.
