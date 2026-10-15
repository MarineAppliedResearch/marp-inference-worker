---
task: MarineAppliedResearch/marp-inference-worker#57
repos: [marp-inference-worker]
status: design
needs: []
---

## Goal

A volunteer who installs the worker can see it is running, and can tell it to get off
their machine without uninstalling it. Today the worker enrols and starts taking jobs with
nothing on screen to say so, and the only controls are a command line. After this, there
is a tray icon whose picture says what the worker is doing, and a menu with three states
the volunteer chooses between: **Pause**, **Background**, **Foreground**.

## Requirements

- **R1** — A tray icon appears whenever the worker is running, and disappears when it
  stops. Hovering it says the worker's state in words.
- **R2** — The tray menu offers three states, one of which is always shown as the current
  one — Pause, Background, Foreground — plus the immediate release in R10, which is an
  action rather than a state.
- **R3** — **Background** is the normal state. The worker takes jobs and never puts a
  window in front of anybody. This is what a fresh install starts in.
- **R4** — **Foreground** shows the live view of the work in progress, the way
  `--screen window` already does, and switching to it while a job is running retargets
  that job rather than waiting for the next one.
- **R5** — **Pause** suspends the job in flight within one progress interval, so the GPU
  stops computing and the volunteer can use their machine. It also stops new work
  arriving, which `free_slots()` already does. The model stays loaded, which is what makes
  R6 free.
- **R6** — Leaving Pause resumes the suspended job where it stopped. The job is not
  restarted and no frames are reprocessed.
- **R7** — A paused job still paused after **five minutes** is stopped properly and
  handed back to the coordinator, so the VRAM and the attempt are both released rather
  than held indefinitely. Five minutes is a named constant, not a literal in a loop.
- **R10** — The volunteer does not have to wait those five minutes. A second menu item —
  **Give me my GPU now** — does the same release immediately: the job stops, the model is
  unloaded, the VRAM is freed and the attempt goes back to the coordinator. This is the
  one to press before starting a game; plain Pause is the one to press before a video
  call.
- **R8** — The volunteer's choice is durable. A worker restarted while paused comes back
  paused, not running.
- **R9** — A machine with no tray (headless, a service account, Linux without a status
  area) runs exactly as it does today. Absence of a tray is never a startup failure.

## Open assumptions

- [x] **A1 · architectural · blocking** — answered 2026-10-14: **yes, `pystray`**, rather
      than hand-rolling `Shell_NotifyIconW` through `ctypes`. Pure Python, no compiler,
      about thirty lines against a few hundred for a native notify-icon and its message
      pump, and it works on Linux too. It brings `Pillow` with it, which is what builds
      the icon bitmap.
- [x] **A2 · behavioural · blocking** — answered 2026-10-14: **five minutes**, and the
      volunteer gets a way to skip the wait. See R7 and R10.
- [ ] **A3 · product/UI · non-blocking** — Does the icon itself change with state (three
      pictures: working, paused, watching), or does only the tooltip and the menu's tick
      change? Proposed: the icon changes, because the point of R1 is that a glance tells
      you.
- [ ] **A4 · behavioural · non-blocking** — Is Foreground sticky for later jobs, or only
      for the one being watched? Proposed: sticky — it is a state, not a one-off action,
      and `config["screen"]` is already durable.
- [ ] **A5 · product/UI · non-blocking** — Does the menu also offer Quit? Proposed: no.
      Stopping the worker for good is the installer's job; a Quit that a volunteer presses
      by accident looks identical to the worker being broken.

## Decisions

- **2026-10-14** — Three volunteer-facing states map onto machinery that already exists
  rather than replacing it: Foreground/Background are the existing `screen` modes
  `window`/`off`, retargetable on a running worker through `/status/screen`. Pause is the
  existing `WorkerState.set_paused`, plus the new suspend described in R5.
- **2026-10-14** — Pause means *suspend now*, not *finish the job then stop*. Isaac:
  *"on pause the job it's running will be paused so the user can use their computer"*.
  The reason somebody presses it is that they want the GPU back, so draining is the wrong
  behaviour even though it is the gentler one.
- **2026-10-14** — **Pausing and releasing the GPU are two different things, and the menu
  offers both.** Suspending the frame loop stops the GPU computing but leaves the model
  resident in VRAM -- which is what makes resuming free, and is also useless to a
  volunteer who wants to play a game. So Pause suspends and keeps the weights, escalating
  to a release after five minutes (R7), and *Give me my GPU now* releases immediately
  (R10). Isaac, 2026-10-14: *"if the user needs their gpu right away they should have that
  option as well"*.

## Plan

1. `jobs/worker_state.py` — the paused flag already exists; add the wait primitive a
   running job blocks on, and the timestamp pause began, for R7's escalation.
2. `engines/tracking_engine.py` — at the existing progress-and-stop checkpoint
   (`ctx.should_stop()`, every `PROGRESS_INTERVAL` frames) also honour a pause: block
   there while paused. That checkpoint is already the one place the frame loop yields, so
   R5 and R6 need no new loop and no frame is reprocessed.
3. `jobs/runner.py` — release: `request_stop()` on the paused job, which is the path a
   coordinator cancel already takes and already unloads and hands the attempt back. Two
   callers -- the five-minute escalation (R7) and the menu item (R10). One code path, so
   the slow one cannot drift from the fast one.
4. `installation/tray.py` (new) — the icon, the menu and the tooltip. Reads state from
   the running worker's own `/status`; writes through `/status/pause` and
   `/status/screen`, which both exist. R10 is the one that needs a new endpoint, because
   nothing today lets a local caller release a running job -- the smallest thing that
   works is a `release` flag on the existing `/status/pause`, not a new route.
5. `installation/launcher.py` — start the tray alongside the worker, and swallow its
   absence (R9).
6. Durability (R8): the tray writes the volunteer's choice through
   `installation/operator_control.py`, which is already the durable local choice, and the
   launcher reads it at startup.

## Acceptance criteria

- Installing the worker on a Windows machine puts an icon in the tray with no extra step.
- Pressing Pause while a job is running: GPU utilisation drops to idle within one progress
  interval, and the coordinator still shows the attempt as held, not failed.
- Leaving Pause: the same job continues, and its progress numbers carry on from where they
  stopped rather than restarting.
- Staying paused past five minutes: the attempt is returned to the coordinator and the
  worker reports no fault.
- Pressing *Give me my GPU now*: VRAM drops to idle immediately -- not just utilisation --
  and the attempt is returned. Measured with `nvidia-smi`, because used memory is the
  number that decides whether a game starts.
- Foreground while a job runs: the live view appears for *that* job.
- Killing and restarting the worker while paused: it comes back paused.
- `pytest` green on a machine with no GPU and no tray.

## Test plan

Filled in at G3.

## Status

- **Gate:** implementing
- **Notes:** The machinery for Background and Foreground already exists and is wired
  (`--screen`, `/status/screen`, `watch/display.py`); this task surfaces it rather than
  building it. The genuinely new code is the tray and the mid-job suspend. A1 and A2 are
  the two that block.
