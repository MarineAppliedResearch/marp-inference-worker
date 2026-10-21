---
task: MarineAppliedResearch/marp-inference-worker#65
repos: [marp-inference-worker]
status: implementing
needs: []
---

## Goal

A worker machine ran out of disk. Everything was under `<state>/jobs`, one directory per
attempt, never removed, and almost all of it was a Chromium profile rebuilt from scratch
for every job. At 86-101 attempts in twelve hours that is 14-24 GB a day against roughly
20 KB a day of actual job output.

When this is done a worker's disk use is flat: the browser profile is set up once per slot
and reused, it stops fetching things it cannot use, and a job workspace is removed once
MARP has the result.

## What was found

Measured against 31 real workspaces left in a checkout's `.marp/local/`:

| | size |
| --- | --- |
| `chromium-profile/` | 69 MB median, 123 MB max, ~700 files |
| everything else in the workspace | under 206 KB |

Two profiles made days apart were 81 MB of 122 MB byte-identical. They are the same bytes,
fetched once per job: `optimization_guide_model_store` 35.8 MB, Safe Browsing lists
15.8 MB, `OnDeviceHeadSuggestModel` 7.6 MB, `component_crx_cache` 7.6 MB.

**None of it is for anything this window does.** The watch window is a display surface for
marp-video-player: one local page on `127.0.0.1`, no address bar, no browsing, no user.
Every background service Chromium starts for a fresh profile is for a feature the window
cannot use.

## Requirements

- **R1** - The Chromium profile is per slot, not per attempt. It lives outside the job
  workspace, is created on first use and is reused by every job that runs in that slot.
- **R2** - A worker's windows are still only ever its own. `close()`, `_profile_in_use()`
  and `close_orphaned_windows()` keep identifying a window by the profile path on its
  command line; a per-slot path is still unique per window that can exist at once.
- **R3** - Orphan cleanup at start covers the profile root. This matters more than before:
  a ghost browser from a crashed worker holds a profile that is now about to be *reused*,
  and Chromium routes a second launch into the existing instance rather than starting one.
- **R4** - Chromium does no background networking and no component updating, and its disk
  cache is capped.
- **R5** - A job workspace is removed once the coordinator has accepted the result, and
  only then: a result held for retry is re-offered by hash at the next start, which needs
  the artifact still on disk.
- **R6** - Leftover workspaces from previous runs are pruned at worker start.
- **R7** - Nothing is removed outside the worker's own roots.

## Design

`<state>/chromium-profiles/slot-<n>` beside `<state>/jobs` and `<state>/datasets`, which is
where the machine-wide dataset cache already went for the same reason.

The runner is the only thing that knows the state directory, and `WatchDisplay` is built in
the child process by `tracking_engine`, so the root travels in `params` exactly as
`dataset_cache` does.

`ultralytics/` in the workspace is `YOLO_CONFIG_DIR` - a settings file, about 1 KB, per
attempt so two concurrent jobs do not fight over one settings file. Same reasoning as the
profile at the same wrong granularity, but it costs nothing and is left alone.

## Not in scope

- **One window for the worker rather than one per job.** The cleaner end state, and where
  the per-slot profile stops being needed at all. It is a redesign of `watch/display.py`
  and belongs with #60.
- **`installation/updater.py` keeping every installed version.** Named in the issue as
  secondary and separate.
- **Dataset cache and gpu-artifacts eviction.** Neither has an issue.

## Open assumptions

- [x] **A1 - architectural - blocking** - answered 2026-10-21: one profile per slot rather
      than one per worker. One per worker is what a display surface should need, and the
      only thing preventing it is that a worker runs several jobs at once - slots are
      derived from VRAM and CPU - and Chromium will not run two processes against one
      `--user-data-dir`. Per slot is "set up once" for every window that can exist, and it
      leaves the find-and-kill-by-profile-path mechanism working unchanged.
- [x] **A2 - behavioural - blocking** - answered 2026-10-21: a long-lived profile directory
      on a volunteer's machine is acceptable. It holds no history and no credentials and
      only ever loads `127.0.0.1`.
- [x] **A3 - destructive - blocking** - answered 2026-10-15 in the issue: a workspace is
      deleted after the result is accepted, and `jobs/` is pruned at start.

## Status

- **Branch:** `65-worker-storage`, the one branch for this work stream.
- **Notes:** G2. A1 and A2 answered 2026-10-21 in conversation, A3 by the issue itself.
