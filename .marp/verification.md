# Verification — marp-inference-worker#60, the watch window's timeline

## What each test proves

| Requirement | Test | Tier | Proves |
| --- | --- | --- | --- |
| R1 | `test_a_frame_packet_carries_the_video_length_and_the_job_start` | unit | the axis and the playhead's bounds reach the page at all |
| R1 | `test_the_watch_window_is_told_the_video_length_and_the_jobs_own_start` | unit (engine) | the engine chooses both numbers from the container and the spec |
| R1, R4 | `test_the_page_reads_the_fields_the_window_actually_publishes` | contract | the page reads the names the publisher really publishes |
| R2 | `test_a_resumed_lease_keeps_the_job_start_the_coordinator_sent` | contract | the coordinator's field survives validation **and the runner's re-dump** |
| R2 | `test_a_lease_without_a_job_start_is_a_job_that_starts_where_it_says` | contract | its absence is not a validation failure |
| R2 | `test_a_fresh_job_begins_where_this_attempt_does` | unit (engine) | a fresh job is drawn as starting where it starts |
| R2 | `test_a_job_start_after_this_attempt_is_ignored` | unit (engine) | a nonsensical resume point cannot invert the bar |
| R2 | MARP_API `tells the worker where the job itself began` | api | the lease actually carries it, from the coordinator's side |
| R2 | MARP_API `says nothing about a job start on a first lease` | api | a fresh lease is not made to look resumed |
| R1, R2, R3 | the browser render, below | render | the only tier that can see the page draw |
| R5 | — | — | no new per-frame work: the page reads the packet it already received and the `/status` poll it already made. Argued, not tested; see *Known gaps* |
| R6 | `test_the_tooltip_says_how_far_through_the_job_is` | unit | the tooltip, built from the real `_describe_jobs` and the real `current_progress` |
| R6 | `test_a_fresh_job_is_measured_from_its_own_start` | unit | the percentage without the coordinator's field |
| — | `test_every_element_the_page_reaches_for_exists_on_it` | contract | no `getElementById` in the page names an id the page does not have |

## Requirements with no test

**R3** has no automated test. Elapsed and the estimate are read from `/status` and
formatted in the page, and nothing in this suite executes the page — so the evidence for
R3 is the render below, which is a human looking at it.

**R5** is an argument rather than a test: the timeline adds two constants to a packet that
was already being sent and reads a `/status` poll that was already being made, so there is
nothing new on the engine's thread. A frame-rate measurement on real hardware would prove
it properly, and is listed as a gap.

## Edge cases

- **A resumed attempt**, which is the case the issue is about: the window must show the
  earlier attempt's frames as covered, not as frames this job skipped. Drawn as a dim
  band, a tick at the resume point, and the words `resumed at 12,000` — three signals,
  because shading alone was too quiet when looked at.
- **A container that reports no frame count.** Some streams do. The axis falls back to the
  job's own window, which is all the window can honestly draw.
- **A container reporting fewer frames than the job's range.** Not trusted either, same
  fallback: a video shorter than the work asked of it is a number to disbelieve.
- **A job start later than this attempt's start.** Impossible from a working coordinator;
  ignored rather than drawn inverted, because a page reading a field out of a job spec has
  no way to ask.
- **A paused job.** No rate, so the estimate is omitted rather than invented.
- **A frame number from the previous job arriving late.** The playhead never moves behind
  where this attempt began.

## Regression coverage

- `test_a_resumed_lease_keeps_the_job_start_the_coordinator_sent` is the trap this change
  would otherwise have walked into. The runner does not hand the child the mapping the
  coordinator sent: it rebuilds it with `envelope.spec.model_dump()`, so a field the spec
  model does not declare is dropped silently. Both repositories' suites pass without it —
  MARP's proves it sent the field, the worker's proves nothing about it, and the window
  never sees it. Proved red by removing the declaration from `FrameRange`.
- `test_the_page_reads_the_fields_the_window_actually_publishes` and
  `test_the_tooltip_says_how_far_through_the_job_is` both build their input from the real
  producer rather than a hand-written dict. That is the `jobs` / `active_jobs` defect from
  #57, where the tray read a key `/status` does not have and every test agreed with it.

## The render

No automated tier executes `live.html`, so it was run: the real page, the real display
server, and synthetic frames in place of a GPU
(`scratchpad/timeline_preview.py`, not committed). Two cases, both read back out of the
DOM rather than judged by eye:

```
resumed job, 54,000-frame video, job 0..30,000, attempt from 12,000
  tltrack  = [10, 1004]   the whole video
  tlwindow = [10,  557.8] 55.6% -- 30,000/54,000
  tlearlier= [10,  223.1] 22.2% -- 12,000/54,000
  tlfill   = [233.1, 11]  from the resume point to the playhead
  tlhead   = [243.1,  2]  23.2% -- frame 12,528
  legend: 0 | frame 12,592 - 42% of this job - resumed at 12,000
             - 16s elapsed - ~7m left | 54,000

fresh job, container reports no frame count, job 12,000..30,000
  tlwindow fills the track, tlearlier is empty, tlstart is hidden
  legend: 12,000 | frame 12,496 - 3% of this job - 12s elapsed - ~7m left | 30,000
```

## Known gaps

- **No GPU and no real video.** The render above used synthetic frames, so what is
  unproven is the same list as always: that the numbers are right while a real engine is
  producing them, and that the window's frame rate is unchanged (R5).
- **The estimate is not calibrated.** It is the observed rate times the frames left, and
  it will swing while a model loads or a job is paused. Nobody has watched one run to the
  end against the clock.
- **Four slots at once has not been looked at.** Each window draws its own job (A4), which
  is what the tiling already does, but four windows with four timelines is a layout nobody
  has seen yet.
