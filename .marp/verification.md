# Verification — marp-inference-worker#20, the stop-outcome contract

Written after G2 rather than before it, and that is a deviation worth naming: this
branch grew from a one-word contract break into seven fixes because a second machine
joined the pool and kept finding things. Every result below was actually run; nothing
here is planned-and-assumed.

## What each test proves

| Requirement | Test | Tier | Proves |
| --- | --- | --- | --- |
| R1 | `test_an_operator_stop_reports_yielded_with_the_frame_it_reached` | unit | a stop reports `yielded`, the word MARP accepts |
| R2 | same test | unit | `completed_through_frame` is a real frame inside the range |
| R1/R2 | live, jobs 3105 / 3385-3396 | two machines | MARP accepts it, stores it, and resumes from it |
| R3 | `test_a_refused_result_survives_to_be_reported_at_the_next_start` | unit | a refused result is kept and re-sent as the outcome it was |
| R3 | live, attempts 2997 / 2998 | two machines | a restart reports its lost attempt and the job requeues |
| R4 | `test_a_childs_warning_reaches_the_operator_not_only_the_coordinator` | unit | a warning inside a job reaches `/status` |
| R4 | `test_a_coordinator_cancel_is_a_notice_not_an_error` | unit | an instruction obeyed is not recorded as a fault |
| R5 | the pair above plus `MARP_API#197`'s tests | unit, both repos | neither side can change the vocabulary silently |
| watch | `test_the_watch_page_ships_with_the_package` | unit | the page `start()` needs exists and polls |
| watch | `test_chromium_is_found_from_configuration_or_an_installed_browser` | unit | a volunteer's own Chrome is found |
| watch | `test_watch_display_is_decidable_by_either_side` | unit | the volunteer's mode alone opens a window |
| watch | live, jobs 3106 / 3392 | screenshot | real video, real boxes, maximised and in front |

**Tier note.** Every watch-mode test that existed before this branch passed while the
feature had never once run, because not one of them called `start()`. The page's absence
was observable only at a tier that touches the file, which is why the new test asserts the
file exists *and* contains `/next` and `acknowledged_frame` — a stub would pass a bare
existence check and leave a blank window.

## Requirements with no test

None outstanding. R2 needed no worker change — `runner.py` already sent the frame and
already honoured a moved `start_frame` — so it is proved live rather than by a new unit
test, and that is recorded above rather than left implied.

## Edge cases

- **A second refusal.** Dropped deliberately and named in `last_error`, rather than
  retried at every start forever.
- **A refused outcome is re-sent as itself**, never downgraded to `failed`. Rewriting a
  worker's own history to fit a coordinator's vocabulary would have destroyed the evidence
  that the vocabulary was wrong.
- **A configured browser that does not exist** is an error, not a silent fallback to a
  different one: somebody who set `MARP_CHROMIUM_PATH` meant that browser.
- **Fullscreen stays the volunteer's own choice.** A job asking to be watched on a machine
  with no mode set gets a window, never the whole screen.
- **Ordinary chatter must not overwrite a warning** in `last_notice`, or the one line
  worth reading is gone before anybody looks.

## Regression coverage

- `yielded` refused by MARP. Nothing in either repository mentioned the word; both suites
  passed against each other's assumption. Now asserted on both sides.
- `failure_reason` on a non-failure.
- A coordinator cancel recorded as an error, leaving `/status` reporting a fault for the
  rest of the session.
- The watch page missing from both repositories and from the installer's staging step.

## Known gaps

- **The watch window caps throughput.** Measured, not fixed: 55.6-75.0 f/s headless
  against 28.0-29.8 windowed on an RTX 4080 SUPER, and 15.7 against 10.0-13.4 on an RTX
  5060 Laptop. `_FrameChannel.publish()` blocks until the browser acknowledges each frame,
  inside the per-frame loop, so inference runs at the speed the display can draw. The
  ceiling is per-machine. **This is a defect and it is not fixed on this branch.**
- The A/B isolating it — same binary, `--screen window` against `--screen off` — is not
  yet run. The numbers above come from two machines on two branches.
- Backing off GPU use when the volunteer starts using their computer: asked for, not
  built, and meaningless until the display stops setting the pace.
- No GPU test tier runs in CI here.

## Manual steps

1. Activate against a coordinator: `worker_main --activate-code-file <file>` with
   `MARP_COORDINATOR_URL` set. Expect a credential at
   `<state dir>/worker-credential.dpapi` and a worker id.
2. Run with `--screen window` and no `MARP_CHROMIUM_PATH`. Expect a maximised window
   titled "MARP · watching inference", in front, drawing boxes with species, track id and
   confidence.
3. Stop a running job through `POST /status/control {"action":"stop"}`. Expect `yielded`
   upstream with `completed_through_frame`, the job requeued, and the next lease starting
   at that frame.

## Walkthrough videos

None. Not asked for, and nothing here would be evidenced by one.

---

## Results

Run on 2026-09-17, on `20-stop-reports-refused-outcome`, Python 3.12.10,
torch 2.11.0+cu128, RTX 5060 Laptop.

```
pytest tests/test_job_runner.py tests/test_status.py tests/test_job_context.py \
       tests/test_coordinator_contract.py tests/test_watch_display.py
86 passed, 2 warnings in 54.28s
```

Earlier passes over the same and adjacent files: 102, 88, 73, 72 passed, no failures.

**Every new test was proved red before it was made green**, by mutating the fix away:
the watch page deleted (`FileNotFoundError` at the assert), the old both-sides watch gate
restored, the in-flight record dropped on refusal, and the cancel written back into
`note_error`. Each failed exactly as the shipped state would have.

### Live, two machines on two networks

Coordinator on the desktop, reached over a public address; worker 898 on this laptop.

```
job 3106  marp_tracking  3000 frames  succeeded   window open, boxes drawn
job 3385  marp_tracking  4500 frames  succeeded   336s, 13.4 f/s, windowed
job 3392  marp_tracking  4500 frames  succeeded   448s, 10.0 f/s, windowed, 2 attempts
attempt 2998  dropped by a restart -> reported "Lost: the worker restarted while this
              job was running", job requeued, re-leased, completed
```

Twelve hand-made jobs and then 56 coordinator-split pieces across both machines. No job
was ever leased by two machines; no lease expired, including at a fifth of throughput;
both cancels stopped a working machine with partial progress recorded. Zero observations
reached the corpus — every job was submitted without `spec.session` deliberately.
