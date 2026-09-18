---
task: MarineAppliedResearch/marp-inference-worker#31
repos: [marp-inference-worker]
status: verifying
needs: []
---

## Goal

A volunteer running the worker on Linux sees the watch window — the model's annotations drawn
live over the video it is working on — exactly as a Windows volunteer does. Today they see
nothing: the browser search looks only for `chrome.exe` under Windows `Program Files`, so
`_find_chromium()` returns `None`, `start()` raises, and the window has never opened on any
Linux machine. The window is how a volunteer can tell the worker is doing something real,
and it is how a wrong model was noticed by eye the one time it happened.

## Requirements

- **R1** — On Linux the worker finds a Chromium-family browser and opens the watch window.
- **R2** — Discovery is **bundled first, system second**, matching Windows, so a packaged
  build and a development machine take the same path rather than two.
- **R3** — A browser found through a multi-call wrapper is launched by the name it was found
  under, not by the target of its symlink.
- **R4** — Window geometry comes from the real display on Linux, with the single-screen
  fallback retained for when it cannot be read.
- **R5** — A finished or killed job closes its own window on Linux, matching on the
  attempt's profile directory so it can never reach the volunteer's own browser.
- **R6** — Windows behaviour is unchanged throughout.

## Open assumptions

- [x] **A1 · environment · blocking** — answered 2026-09-18 by Isaac: this machine had no
  Chromium-family browser at all, so #31 could not be demonstrated. **Install a system
  Chromium**, explicitly as an interim measure. His words: *"I don't want a volunteer to have
  to install fucking things to get this working!!!!! but for now have it go ahead and install
  the damned chromium browser, but we have to make sure later that we package up everything
  the user needs."* So the shipped answer is the bundled-snapshot pattern that
  `packaging/chromium-windows-x64.lock.json` already implements for Windows, pointed at
  `Linux_x64/<snapshot>/chrome-linux.zip`. That is **not in this branch** — see *Not in
  scope*.

- [x] **A2 · product/UI · blocking** — answered 2026-09-18 by Isaac: the only display here is
  a 5120x2880 panel rotated left, so the desktop is 2880x5120 portrait. Is a landscape window
  on a portrait screen acceptable rather than a reason to hold? **Yes.** So R4 is about
  reading the real geometry, not about laying out differently for portrait.

## Decisions

- **2026-09-18** — browser discovery asks `PATH` via `shutil.which` before falling back to
  fixed paths. Linux has no `Program Files` equivalent and a volunteer's browser may be a
  distribution package, a vendor `.deb`, a snap or a flatpak. The fixed paths remain for a
  worker started from a service with a minimal environment, where `PATH` is not the whole
  answer.

- **2026-09-18** — `_find_chromium()` no longer resolves a candidate whose resolved basename
  differs from the name it was found under. `/snap/bin/chromium` is a symlink to
  `/usr/bin/snap`, which reads `argv[0]` to decide which snap to run: resolving it launches
  the snap tool with Chromium's arguments and nothing appears. This was found by running it,
  not by reading it — the first attempt returned `/usr/bin/snap` and opened no window.

- **2026-09-18** — window cleanup on Linux is `pkill -f <profile path>`. Matching the profile
  directory rather than the process name is what keeps it off the volunteer's own browser,
  and that matters more here than on Windows because the system Chromium the worker borrows
  may be the browser they are reading the instructions in.

- **2026-09-18** — raising the window and idle detection are left as no-ops on Linux, as
  advised. They are polish and they are not worth blocking the port on.

## Plan

1. Branch **from `32-linux-credential-store`, not `develop`** — stacked deliberately. Without
   #32 a Linux worker cannot read its credential at startup, so a `develop`-based branch
   could not be run at all on the machine this was developed and tested on. *(done)*
2. Linux browser discovery in `_find_chromium()`, bundled first. *(done)*
3. Stop resolving multi-call wrappers. *(done)*
4. `xrandr --listmonitors` branch in `_monitors()`, ahead of the existing fallback. *(done)*
5. Linux branch in `_kill_by_profile()`. *(done)*
6. Tests for each. *(done)*

## Acceptance criteria

- The watch window opens on Linux during a real job. **Met** — observed on attempt 5043,
  with `browser-render-proof.jpg` written and Chromium running as pid 20041.
- `_find_chromium()` returns a launchable path on a snap-based Ubuntu. **Met** — returns
  `/snap/bin/chromium`, not `/usr/bin/snap`.
- `_monitors()` reports the real desktop. **Met** — `[(0, 0, 2880, 5120)]` on the rotated 5K
  panel, where the fallback would have said 1920x1080.
- A finished job leaves no window. **Not directly observed** — see below.
- Windows unchanged. **Not executable here.**

## Test plan

`tests/test_watch_display.py`, six added tests, run as a file rather than as a suite:
browser found on `PATH`; a multi-call wrapper not resolved; `xrandr` parsed into rectangles;
the fallback when `xrandr` is absent; `pkill` invoked with the profile path; and nothing
invoked when there is no profile. 24 passed, up from 18.

**What these do not prove**, stated rather than left to look like coverage:
- **R5 is proven at the wrong tier.** The test asserts `pkill` is *called with* the profile
  path. It does not assert a window actually disappeared, because that needs a real window
  and a real desktop. The failure mode #29 describes could still exist here.
- **R6 is not proven at all.** Windows cannot be executed on this machine. Every change is a
  `sys.platform` branch that returns before the Windows code, but the Windows path has not
  been run.

## Not in scope

- **The bundled Chromium snapshot for Linux**, which is the actual answer to A1 and is the
  difference between this working for a developer and working for a volunteer. Needs a
  `chromium-linux-x64.lock.json` alongside the Windows one.
- **Snap confinement is untested and may matter.** The browser used here is the Chromium
  snap, which is AppArmor-confined. It rendered the page from the package directory under
  `$HOME` and it worked, but a confined snap is a poor stand-in for the bundled build, and
  it could pass here and fail for a volunteer, or the reverse.
- **Raising and idle detection**, left as no-ops by decision.
- **#29**, orphaned windows from a previous run, which is `marp-laptop`'s branch.

## Status

- **Gate:** verifying
- **Notes:** written **after** implementation rather than before it, which is not the order
  the harness asks for. Both assumptions had been answered by Isaac beforehand, so no
  blocking assumption was open while the code was written, but the spec did not exist at
  G1 and this note is here rather than left for someone to notice.

  Found and left alone: `test_slots_fill_the_monitors_before_they_are_subdivided` is defined
  twice in `tests/test_watch_display.py`, at lines 318 and 660. The second shadows the first,
  so one of them has never run. Pre-dates this branch.
