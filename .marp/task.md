---
task: MarineAppliedResearch/marp-inference-worker#49
repos: [marp-inference-worker]
status: design
needs: []
---

# A track's species comes from the evidence, not its first frame

Refs MarineAppliedResearch/marp-inference-worker#49

## Goal

A tracked animal is recorded as the species the model most believed it was across the whole
time it was tracked, not the species it was called on the first frame. That first frame is
usually the worst view: smallest, furthest and most occluded. At `conf 0.001` almost any
detection opens a track. The watch window stops flickering between species on one animal
and shows the species that will be recorded.

## What investigation found

- `tracking/track_accumulator.py#observe` writes `class_name` when a track opens and never
  again. The frames appended after carry geometry and confidence, but no class.
- `engines/tracking_engine.py#_observe_tracks` does know each frame's class: it matches the
  track box to the detection it overlaps most (IoU above `_CLASS_MATCH_IOU`) and takes that
  detection's class and confidence. It then throws the class away, except for the display.
- The display (`live_track_metadata`) shows the latest single frame with a matched
  detection. That is why it flickers.
- `reduction/keyframes.py` labels every keyframe with the track's one `class_name`, and
  `tracking/observations.py` takes the observation's `comname` from the first keyframe. So
  keyframes can never disagree (0 of 191 observations did, per the issue), and that should
  stay true.
- A frame the tracker predicted with no detection behind it has no class ("Unknown") and
  no confidence.

## Requirements

- **R1** — Each frame of a track keeps the class of the detection behind it, or none when
  the frame was predicted.
- **R2** — When a track ends, its species is decided from all of its frames by the rule
  settled in A1, not from its first frame.
- **R3** — Every keyframe and the observation carry that one species. Keyframes of one
  observation never disagree.
- **R4** — The watch window shows each track's species by the same rule, applied to the
  frames seen so far, so it does not flip on one frame's opinion and it converges on what
  will be recorded.
- **R5** — A frame with no detection behind it contributes nothing to the decision. A track
  none of whose frames had a detection is "Unknown", as today.
- **R6** — The keyframe reduction is not rewritten. It is handed the decided species in the
  same `class_name` field it reads today.

## Open assumptions

- [ ] **A1 · scientific · blocking** — **Which rule decides the species?** They give
  different answers on the same track:
  - **(a) Confidence-weighted vote.** Sum each class's detection confidence across the
    track, and the highest total wins. Recommended: it counts every view, and a few
    confident frames outweigh a long stretch of 0.01 noise. Of the three, it is the closest
    to "the one that's most likely".
  - **(b) Highest-confidence frame.** The single view the model was most sure of. Easiest to
    explain, but one lucky frame decides.
  - **(c) Majority vote.** One frame, one vote. At `conf 0.001` a long dull stretch of
    near-zero detections can outvote a few excellent ones.
- [ ] **A2 · scientific · blocking** — **What about a track that drifts onto a different
  animal?** The rule gives the whole track one species. That is the vote winner, not the
  first frame, so it is not first-frame-wins-forever, but half a track can still belong to
  something else. Options:
  - **(a) One species per track.** Recommended: drift is the tracker's error, and the vote
    already labels the track by what it mostly was.
  - **(b) Split the track into two observations** where the species persistently changes.
    That changes how many observations a run produces, and needs a rule for "persistently".
- [ ] **A3 · data-meaning · blocking** — **Which confidence does the observation report?**
  `confidence` is the detection score at the observation frame. After this change, that
  frame's detection can be a *different* class from the decided species. For example, it
  says 0.9 for rockfish on an observation recorded as an anemone. Options:
  - **(a) Unchanged:** the score at the observation frame, whatever class it was for.
  - **(b) The decided species' score at that frame,** or null when that frame's detection
    was another class. Recommended: the number then always means "how sure the model was
    that this is the recorded species, here".
  - **(c) An aggregate** for the decided species across the track, which changes what the
    column means.
- [ ] **A4 · behavioural · non-blocking** — Ties go to the class seen first, so the result
  is deterministic. Rare with real-valued confidences.
- [ ] **A5 · API contract · non-blocking** — The per-frame classes stay inside the worker.
  Nothing new is added to the result rows, so the coordinator's ingest is unchanged.
  Reporting the runner-up species for review could come later, as its own contract change.
- [ ] **A6 · product/UI · non-blocking** — The display uses the tally so far, not a sliding
  window. A window would follow drift faster, but it would flicker again and show something
  other than what gets recorded.

## Decisions

## Plan

1. `track_accumulator.observe`: store each frame's class, with none for a predicted frame
   (R1, R5).
2. One small function that decides a species from a track's frames by the A1 rule (R2, A4).
3. `_close`: set the track's `class_name` from it before handing the track to the
   reduction (R3, R6). Adjust the observation's confidence per A3.
4. `_observe_tracks`: the display asks the accumulator for the running decision instead of
   remembering the last frame (R4).
5. Tests, below.

## Acceptance criteria

- A track whose first frame says one species and whose later, more confident frames say
  another is recorded as the second, with every keyframe agreeing.
- The watch window's label for a track does not change on a single dissenting frame.
- `pytest` passes.

## Test plan

- Unit, `track_accumulator`: the rule on hand-built frames, including the first-frame case
  from the issue, predicted frames, an all-Unknown track and a tie (R1, R2, R5, A4).
- Pipeline, `test_tracking_pipeline.py` with mocked detections: an ended track's keyframes
  and observation carry the decided species, and its confidence follows A3 (R3, R6).
- Engine, `_observe_tracks` with fake tracks: the display label holds through one
  dissenting frame (R4).
- Once, not committed: one real job on this machine's GPU, with how many observations
  changed species against the same piece run on develop.

## Status

- **Gate:** design
- **Notes:** A1–A3 need Isaac. Nothing is implemented yet.
