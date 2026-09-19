---
task: MarineAppliedResearch/marp-inference-worker#39
repos: [marp-inference-worker]
status: verifying
needs: []
---

## Goal

A volunteer's worker never signals a process it does not own. Today `close()` terminates the
process id it launched, and on Linux that id is routinely not the browser any more — so
ending a job can kill an unrelated program belonging to the volunteer.

## Requirements

- **R1** — `close()` signals the launched pid only when that pid is still this job's browser.
- **R2** — A pid that has vanished is treated as not ours, without raising. This is the
  common case, not an edge case: it happens on every job.
- **R3** — The browser is still closed. `_kill_by_profile()` remains the mechanism; the pid
  path is a backstop.
- **R4** — Windows behaviour is unchanged.

## Open assumptions

None. The traceback names the failing call, the mechanism is `/snap/bin/chromium` being a
symlink to `/usr/bin/snap`, and the fix is the one #28 already established on Windows —
match on what is uniquely ours rather than on a pid. Nothing here is a judgement call that a
different reasonable answer would change.

## Decisions

- **2026-09-18** — ownership is decided by reading `/proc/<pid>/cmdline` for this job's
  profile directory. An unreadable or absent `/proc` entry means "not ours", so the failure
  direction is leaving a window for `_kill_by_profile()` rather than signalling a stranger.

## Plan

1. Branch `39-close-trusts-a-recycled-pid` off `develop` at `7964572`. *(done)*
2. Add `_still_our_browser()` and guard the non-Windows terminate with it. *(done)*
3. Tests at the tier that can see it. *(done)*

## Acceptance criteria

- A pid whose command line does not name this job's profile is not signalled.
- A vanished pid returns False rather than raising.
- Windows still uses `taskkill /T /F` on the process tree.

## Test plan

`tests/test_watch_display.py`, two added tests: a pid whose command line names something
else is refused, one that names the profile is accepted, and a vanished pid returns False.
26 passed, up from 24. Both new tests proven red against the unfixed module first.

**What this does not prove:** that no window is left behind in production. That needs a real
browser on a real desktop over many jobs, and the evidence for it is operational rather than
a test — ~50 attempts on this machine with one Chromium at a time and no accumulation.

## Status

- **Gate:** verifying
- **Notes:** found by MARP-DESKTOP-DEV pulling the traceback off the coordinator, because
  **the worker writes no local record of a failed job**. 550 KB of worker log across the
  failure window contains no traceback and no error line. That gap is worth its own issue and
  is not fixed here — a volunteer whose every job fails has nothing to look at and nothing to
  send.
