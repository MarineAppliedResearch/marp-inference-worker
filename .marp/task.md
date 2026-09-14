---
task: MarineAppliedResearch/marp-inference-worker#9
repos: [marp-inference-worker]
status: verifying
needs: []
---

## Goal

Every keyframe produced by the tracking pipeline retains the model confidence measured on
that keyframe's own frame, so MARP can rank or filter representative images without falling
back to box area.

## Requirements

- **R1** — Every keyframe emitted by the selected reduction carries a `confidence` field.
- **R2** — A keyframe's confidence is the raw detection confidence recorded on that exact
  frame, carried through unchanged. It is not a track average, an extremum, a recomputed
  score, or a score borrowed from a neighbouring frame.
- **R3** — When the selected raw frame has no matched detection and therefore records
  `confidence: None`, the emitted keyframe keeps the field with a null value rather than
  inventing a score or omitting the field.
- **R4** — The confidence survives the complete reduction and observation-shaping path and
  remains JSON serializable for the existing MARP_API ingest consumer.
- **R5** — Keyframe selection, start/middle/end labels, frame numbers, and directional box
  padding remain unchanged.

## Open assumptions

- [x] **A1 · scientific/data-meaning · blocking** — answered 2026-09-13: yes, the confidence
  measured on the raw detection box deliberately accompanies the directionally padded
  keyframe box. The confidence describes the model result on the named frame; padding is
  geometry added afterward and must not cause the score to be recomputed. Directional
  padding may be replaced by a different detection representation in future work, but that
  is outside this additive repair.
- [x] **A2 · api contract · blocking** — answered 2026-09-13: keep `v3_dirpad` version `1`.
  Adding the missing `confidence` field repairs its output contract; frame selection and box
  geometry remain unchanged. A future change to padding, selection, or other reduction
  behavior should use a new reducer name or version so stored results remain attributable.

## Decisions

- **2026-09-13** — The MARP_API ingest path already accepts numeric keyframe confidence and
  stores null otherwise; issue #9 requires no API-side contract or schema change.
- **2026-09-13** — Predicted frames use the same established semantics as observation
  confidence: no matched detection means an explicit null, never a nearby detection's score.
- **2026-09-13** — Confidence continues to describe the raw model detection while the
  keyframe box remains directionally padded. The existing reducer stays at version `1` for
  this additive repair.

## Plan

1. Add the confidence from each selected raw frame to the keyframe emitted by
   `reduce_to_keyframes_v3_dirpad`, without changing selection or padding.
2. Add focused pipeline tests proving distinct per-frame scores remain attached to their own
   keyframes and that an undetected selected frame remains null.
3. Confirm the complete observation payload serializes those values in the shape consumed by
   MARP_API.
4. Write the G3 verification plan for human review before running it.

## Acceptance criteria

- Reduced keyframes contain the confidence from their own source frames.
- At least two selected frames with different scores prove the value is not copied from one
  track-level source.
- A selected predicted frame produces `"confidence": null` after JSON serialization.
- Existing keyframe selection, labels, frame numbers, and padded coordinates are unchanged.
- No MARP_API code or database migration is required.

## Test plan

Written in `.marp/verification.md`. The focused tests belong in
`tests/test_tracking_pipeline.py`, where the real tracker, accumulator, reducer, and
observation shaper can observe whether a score stayed attached to its frame. Awaiting human
review before anything is run.

## Status

- **Gate:** verifying
- **Notes:** Branch `9-keyframe-confidence` is based on current `origin/develop`. Local code
  inspection confirms `TrackAccumulator` already stores per-frame confidence and
  `reduce_to_keyframes_v3_dirpad` drops it only when constructing each output dictionary.
  A1 and A2 were answered by Isaac on 2026-09-13. The implementation and focused tests are
  written. The approved focused verification passed all three selected pipeline tests. Ruff
  found five pre-existing issues also present on `origin/develop`; none points to an issue #9
  addition. The G4 evidence awaits human review.
