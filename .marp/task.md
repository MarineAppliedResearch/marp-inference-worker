---
task: MarineAppliedResearch/marp-inference-worker#60
repos: [marp-inference-worker]
status: design
needs: []
---

## Goal

Somebody watching a donated machine work can see where in the video this job is, how long
it has been going, and roughly how much longer it has left. Today the window shows the
frame being worked on and a 3-pixel bar at the very bottom, which says "some fraction of
something" and nothing more -- not which part of the video this job owns, not where the
run started, not whether it is ten minutes or three hours from finishing.

## Requirements

- **R1** -- The watch window carries a timeline showing the frame range this job is
  working and the playhead inside it. Frame numbers are stated, not only drawn.
- **R2** -- A resumed attempt reads as a resumed attempt rather than as a job that skipped
  its first frames: what an earlier attempt already covered is visibly distinct from what
  this one is doing.
- **R3** -- Elapsed time for this attempt, and an estimate of time remaining, both in the
  window. The estimate is from frames actually finished, not from wall clock alone.
- **R4** -- The timeline is still correct when the job's window is a slice of a longer
  video, and degrades to the window alone when the container will not report a length.
- **R5** -- Nothing on the timeline sets the pace of inference. It is drawn from data the
  window already receives and already polls; no new per-frame work on the engine's thread.
- **R6** -- (tooltip; see A3) The tray tooltip says how far through the running job is, so
  Background mode is not blind.

## Open assumptions

- [x] **A1 - product/UI - blocking** -- answered 2026-10-15: **(b)+(c)**. The bar is the
      whole video, the job's window is marked inside it, what an earlier attempt covered
      is dim and this attempt's part is bright. The question was:
      What does the full width of the timeline represent?
      (a) **this attempt's window only** -- `start_frame`..`end_frame` as the worker
      receives it, which for a resumed job is already the remaining part;
      (b) **the job's whole window**, with the part earlier attempts covered drawn dim and
      this attempt's part bright -- this is what the issue describes, and it needs A2;
      (c) **the whole video**, with the job's window marked inside it and the playhead
      within that -- the worker already knows the video's length from the container.
      (b) and (c) combine: the whole video, the job's window marked, this attempt's start
      ticked. Recommendation: **(b)+(c)**, because it is the only reading in which "where
      the job actually started" means anything, and the video-length context is free.

- [x] **A2 - cross-repository - blocking** -- answered 2026-10-15: **yes, add it**. The
      coordinator puts the job's own window start on the copy it hands over; the stored
      spec stays untouched. Second branch and second pull request in MARP_API, and the
      worker tolerates its absence so the two can merge in either order. The question was:
      Only if A1 includes (b). The worker cannot
      tell a resumed attempt from a fresh one: `applyResumePoint` in MARP_API hands over a
      *copy* of the spec with `range.start_frame` moved to the resume point, deliberately,
      and the original start is not in anything the worker receives. So (b) needs one
      extra field on the leased spec copy -- the job's own `start_frame` -- which is a
      change in MARP_API, a second branch and a second pull request. Is that worth it, or
      do we take (a)/(c) and keep this to one repository?

- [x] **A3 - product/UI - blocking** -- answered 2026-10-15: **yes, in the tooltip**.
      Does the tray tooltip carry a percentage as well,
      or does the timeline live only in the window? Recommendation: **yes, in the
      tooltip** -- in Background mode the tooltip is the only thing a volunteer can see,
      and `tooltip_for()` already reads `/status`, which already carries `progress`.

- [ ] **A4 - product/UI - non-blocking** -- One timeline per window, or one combined view
      of every slot? Recommendation: **one per window**. Each slot already gets its own
      watch window, tiled by `_tile(slot_index, slot_count)`, and each is a different
      video; a combined view would put four unrelated videos' ranges on one axis. The
      machine-wide number that is worth having -- frames per second across all slots -- is
      already in the top bar.

- [ ] **A5 - behavioural - non-blocking** -- Where does "time remaining" come from? The
      page already measures a 10-second rolling frame rate for the `fps` readout, and
      `/status` carries this attempt's `progress.done` and `elapsed_s`, whose ratio is the
      attempt's average rate. Recommendation: **the attempt average**, falling back to the
      rolling rate before there is enough of it -- an estimate that jumps around is an
      estimate nobody reads, and the average steadies as the job goes on.

## Decisions

- **2026-10-15** -- Display only. Every number already flows: `report_progress` fires every
  `_PROGRESS_EVERY_FRAMES` frames, `/status` reports each running job's `progress`, and the
  frame packet already carries `range_start` and `range_end`. Nothing new is measured.

## Plan

1. Settle A1-A3.
2. Carry the video's length to the window. `geometry.total_frames` is already read in
   `tracking_engine.py`; it reaches the page through the frame packet, beside the range.
3. Draw the timeline in `live.html`: the band, the window inside it, the attempt's start,
   the playhead, and the frame numbers under it.
4. Elapsed and remaining, from `/status` in `refreshFooter`, which already polls.
5. If A3 is yes: the percentage into `tooltip_for()` in `installation/tray.py`.
6. If A2 is yes: the job's own start on the leased spec copy, in MARP_API, on its own
   branch and its own pull request, with the worker tolerating its absence.

## Acceptance criteria

- With a job running, the window shows the job's range, the playhead inside it, and both
  frame numbers; the bar moves as the job runs.
- A job that resumes part-way through its window shows the earlier part as already covered
  rather than starting its bar part-filled with no explanation.
- Elapsed counts up and the remaining estimate falls; neither is blank once the job has
  run long enough to have a rate.
- A video whose container reports no frame count still draws a usable timeline.
- Frame rate with the window open is unchanged, slot for slot, from before this change.

## Test plan

`.marp/verification.md`, written at G3. Two things in it are worth reading before the
rest: R3 and R5 have no automated test and say so, and the render section is the only
tier that can see the page draw.

## Status

- **Gate:** verifying
- **Notes:** A1-A3 answered 2026-10-15. A4 stands as recommended. **A5 was implemented
  the other way round from its recommendation**, and the reason is a fact found after it
  was written: `progress.elapsed_s` counts from when the job *process* started, so the
  model load, the seek and every second of a pause are inside it. Dividing by it
  understates the rate and inflates the estimate, worst on exactly the paused job somebody
  is looking at the window to understand. So the rolling rate leads and the average is the
  fallback. The coordinator's half is MARP_API branch `lease-says-where-the-job-began`,
  which has no issue of its own -- one is needed there, or this can be referenced from
  #60.
