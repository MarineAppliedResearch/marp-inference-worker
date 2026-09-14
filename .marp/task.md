---
task: MarineAppliedResearch/marp-inference-worker#9
repos: [marp-inference-worker]
status: design
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

- [ ] **A1 · scientific/data-meaning · blocking** — May the confidence measured on the raw
  detection box deliberately accompany the directionally padded keyframe box? Proposed:
  yes. The confidence describes the model result on the named frame; padding is presentation
  geometry added afterward and must not cause the score to be recomputed.
- [ ] **A2 · api contract · blocking** — Is adding the missing `confidence` field a repair
  to `v3_dirpad` version `1`, or must the reduction be registered under a new version?
  Proposed: keep version `1`, because frame selection and box geometry do not change and the
  existing API consumer already treats `confidence` as part of the keyframe shape.

## Decisions

- **2026-09-13** — The MARP_API ingest path already accepts numeric keyframe confidence and
  stores null otherwise; issue #9 requires no API-side contract or schema change.
- **2026-09-13** — Predicted frames use the same established semantics as observation
  confidence: no matched detection means an explicit null, never a nearby detection's score.

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

To be written at G3 after A1 and A2 are answered. The focused tests belong in
`tests/test_tracking_pipeline.py`, where the real tracker, accumulator, reducer, and
observation shaper can observe whether a score stayed attached to its frame.

## Status

- **Gate:** design
- **Notes:** Branch `9-keyframe-confidence` is based on current `origin/develop`. Local code
  inspection confirms `TrackAccumulator` already stores per-frame confidence and
  `reduce_to_keyframes_v3_dirpad` drops it only when constructing each output dictionary.
  Implementation is blocked on A1 and A2.
