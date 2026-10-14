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
- **The GPU actually going idle.** `pytest` runs with no GPU here. That the frame
  loop blocks is proved; that blocking frees the card follows from it but is not
  measured. `nvidia-smi` on a real run is the check, and used memory is the number
  that matters for *Give me my GPU now* -- utilisation dropping is what a suspend
  looks like either way.
- **The five minutes.** The release is tested with the clock moved on, not by
  waiting five minutes.

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
