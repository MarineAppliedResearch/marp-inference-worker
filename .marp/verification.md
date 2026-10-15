---
task: MarineAppliedResearch/marp-inference-worker#57
status: verified-at-the-tiers-that-can-see-it
---

## What was run

`pytest` in this worktree's own `.venv312`, 2026-10-14:

```
304 passed, 4 skipped, 1 warning in 66.09s
```

`tests/test_volunteer_control.py` is the new file: 17 tests.

## Each requirement, and the test that proves it

| | Requirement | Test |
| --- | --- | --- |
| R1 | tray appears, tooltip says the state in words | `tooltip_for` covered; the icon itself is **not** covered -- see below |
| R2 | three states, current one marked | `test_the_tray_reads_the_worker_rather_than_its_own_memory` |
| R3 | Background is normal and takes work | `test_the_three_states_map_onto_actions_the_worker_already_has` |
| R4 | Foreground retargets the running job | same test: asserts the `/status/screen` call, which is the existing retarget path |
| R5 | pause suspends the job in flight | `test_pause_suspends_the_job_and_resuming_lets_it_carry_on`, `test_pause_reaches_the_job_in_flight`, `test_the_service_pass_actually_applies_the_pause` |
| R6 | resuming carries on, nothing reprocessed | `test_pause_suspends_...` (blocks on the real file and returns when it goes), `test_resuming_releases_every_job`, `test_resuming_clears_the_release_clock` |
| R7 | released after five minutes | `test_a_pause_held_too_long_hands_the_attempt_back` |
| R8 | the choice is durable across a restart | `test_pause_is_a_durable_choice` |
| R9 | no tray is not a failure | **not covered** -- see below |
| R10 | give me my GPU now | `test_give_me_my_gpu_now_releases_rather_than_suspends` |

## Proved red before green

Three tests were run against the pre-#57 behaviour, with the pause removed from
the context and the call removed from the service pass:

```
FAILED tests/test_volunteer_control.py::test_pause_suspends_the_job_and_resuming_lets_it_carry_on
FAILED tests/test_volunteer_control.py::test_a_paused_job_can_still_be_stopped
FAILED tests/test_volunteer_control.py::test_the_service_pass_actually_applies_the_pause
3 failed, 14 passed
```

`test_the_service_pass_actually_applies_the_pause` exists **because of** that run.
The first attempt at this file tested `_apply_pause_to_running_jobs` directly, and
those tests stayed green when the call to it was deleted -- they proved the method
worked and not that anything ran it. That is the tier trap in the testing doctrine,
caught here rather than in a bug report.

## Run on real hardware, 2026-10-14

Isaac drove the tray on his own machine against job 14370 and 14371, real
inference on an RTX 4080 SUPER through the installed worker. Measured
alongside with `nvidia-smi` and the worker's own `/status`.

| | What happened |
| --- | --- |
| R5 pause suspends the job | utilisation fell within one progress checkpoint; the job stayed held, not failed |
| R6 resume continues | 5,010 -> 19,470 frames on 14370; 4,620 -> 22,320 on 14371 after a release. No frame reprocessed, no restart |
| R10 release frees the card | **1558 MiB -> 634 MiB**, which is the model unloaded rather than merely idle |
| R10 work is not thrown away | the attempt recorded `yielded`, and the job carried `resume_from_frame = 4620`; the retry carried on from there |
| the window survives a pause | paused twice with the live view open; it held both times and closed only on release, which is correct |

The sample that caught the release, from the watcher running alongside:

```
16:46:13 action/frames:[running 3450] gpu:[51 %, 1558 MiB] chrome:37
16:46:34 action/frames:[stop    4590] gpu:[1 %,   634 MiB] chrome:26
16:47:14 action/frames:[running  420] gpu:[20 %, 1558 MiB] chrome:38
```

**Four defects were found by using it that the suite had passed over**, and each
now has a test at a tier that can see it:

1. pystray refuses a menu action whose signature is not exactly `(icon, item)`.
   The menu raised before the icon existed and the launcher's swallow hid it.
2. `pystray` reached `pyproject.toml` but not `packaging/requirements-*.in`,
   which is what the installer resolves from. The artifact built, installed and
   ran with no icon.
3. The tray sent `running` to resume; the endpoint accepts `resume`. Pause
   worked, un-pause was rejected with a 422 the tray swallowed.
4. Pausing closed the live window permanently -- the page gives up when its
   long poll goes unanswered, and a pause outlasts that.

Three of those passed a green suite because the tests asserted the calls the
tray makes -- they checked it against itself rather than against the thing it
talks to. That is the lesson from this task, and
`test_every_action_the_tray_sends_is_accepted_by_the_endpoint` is the shape of
the answer: read the other side's own declaration, never a copy of it.

## The five-minute release, run twice, 2026-10-15

Isaac paused a real job at 10:49 and left the worker alone. It fired on time
and freed the card -- **and failed the requirement**:

```
10:54:12  "paused for 303s; releasing 12753 back to the coordinator"  vram 1563 -> 640 MiB
```

| | first run | after the fix |
| --- | --- | --- |
| job state | `cancelled` -- dead, nobody picks it up | `queued` |
| `yields_made` | 0 | 1 |
| `resume_from_frame` | `None` -- twelve thousand frames lost | 1560 |
| attempt | `cancelled` | `yielded` |

The escalation called `request_stop()` where the volunteer's own *Give me my
GPU now* calls `request_yield()`. One flag apart, and that flag decides whether
the coordinator gets the attempt back with its resume point or the job dies
with the work in it. The comment justifying it said `request_stop()` was *"the
same path a coordinator cancel takes"* -- which is exactly why it was wrong.

**The test passed throughout**, because it asserted the job was stopped. It
was; it was simply the wrong kind of stop. The stand-in job now tells a yield
from a bare stop, and `npx`-style clock-moving still cannot see this: the
second run is what proves it, and it is the reason this requirement needed a
real five minutes rather than a faster test.

The invisibility of the lost range is tracked separately as
`MarineAppliedResearch/MARP_API#257` -- the worker bug is fixed, but nothing in
the coordinator would have said that range was still owed.

## What is NOT covered, and why

- **The tray icon itself.** No test drives a real notify-icon: that needs a
  desktop session and a window manager, which this tier does not have. What is
  tested is everything behind it -- the state reading, the tooltip text, the
  colours, and the exact HTTP calls each menu item makes. **Somebody has to look
  at the actual icon**, and that is in the acceptance list below.
- **R9, running without a tray.** The launcher swallows every tray failure, which
  by construction cannot be observed from inside the process that swallowed it.
  Verified by reading, not by test. A Linux machine with no status area is the
  real check.
- *(settled -- see the five-minute run below)*
- **A volunteer's first install**, as opposed to a reinstall over an existing
  one. Everything here ran on a machine that already had a worker.

## For the human to check on a real machine

1. Install and look at the tray. Is there an icon, and does hovering say something
   true?
2. Start a job. Pause it. Does `nvidia-smi` utilisation drop, and does the job
   survive?
3. Resume. Does it carry on from where it was rather than restarting?
4. Pause and walk away for five minutes. Does the attempt go back to the
   coordinator with no fault reported?
5. Press *Give me my GPU now* during a job. Does `nvidia-smi` **used memory** drop?
6. Switch to Foreground mid-job. Does the live view appear for that job?
7. Kill the worker while paused and restart it. Does it come back paused?
