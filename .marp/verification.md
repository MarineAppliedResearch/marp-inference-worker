# Verification — MarineAppliedResearch/marp-inference-worker#9

<!--
  The G3 package. Written BEFORE anything is run, and reviewed by a human before it is
  run. The point of this file is that "you're missing this case" and "that test does not
  actually prove the requirement" get said while they are still cheap.

  `marp verify plan` drafts it from .marp/task.md. `marp verify run` executes what was
  approved and appends the real results, failures included, verbatim.
-->

## What each test proves

| Requirement | Test | Tier | Proves |
| --- | --- | --- | --- |
| R1, R2 | `test_keyframe_confidence_is_the_score_at_each_keyframes_own_frame` | pipeline | The real tracker and accumulator produce frame-specific scores; every selected keyframe contains the score from the raw frame named by its `framenum`, with at least two different scores preventing a copied track-level value from passing. |
| R3 | `test_a_predicted_keyframe_keeps_an_explicit_null_confidence` | pipeline | A keyframe selected from ByteTrack's prediction-only tail retains an explicit null instead of borrowing a nearby detection's score. |
| R4 | Both new tests | pipeline/serialization | `build_observation` preserves the reduced keyframes and JSON serialization retains numeric confidence and null in the payload sent toward MARP_API. |
| R5 | `test_full_pipeline_produces_one_observation_for_one_animal` | pipeline regression | The existing registered `v3_dirpad/1` path still reduces a real accumulated track, labels its first and last keyframes, and preserves the reduction identity through observation shaping. |

**Choosing the tier is the decision that matters.** A rendering defect passes every
store-level check. A rule defect passes every browser test that never exercises it. A fix
reported as verified at a tier that structurally cannot observe the defect is how the same
bug gets reported twice.

## Requirements with no test

None.

<!-- Drafted by `marp verify plan` from the requirement ids the suite mentions.
     A requirement is "covered" here only in the sense that some test names it. Whether
     that test proves it is the thing a human is reviewing. -->

## Edge cases

- **Prediction without detection:** ByteTrack may carry a box through a gap. The keyframe
  keeps `confidence: None`, because no model score exists on that frame.
- **Several selected frames:** Distinct confidence values are compared by `framenum`, so a
  first, last, minimum, maximum, mean, or copied score cannot satisfy the assertion.
- **Padded geometry:** Tests obtain confidence from the raw frame but leave the existing
  directional padding path active, matching the settled A1 meaning.

## Regression coverage

Run:

```powershell
.venv312\Scripts\python -m pytest tests/test_tracking_pipeline.py -k "keyframe_confidence or predicted_keyframe or full_pipeline"
```

The two new tests reproduce issue #9 at the pipeline tier: before the repair, every reduced
keyframe lacks `confidence`. The existing full-pipeline test protects the reducer registration,
start/end labels, and observation handoff while this output field changes.

Then lint only the changed Python files:

```powershell
.venv312\Scripts\python -m ruff check src/marp_inference_worker/reduction/keyframes.py tests/test_tracking_pipeline.py
```

## Known gaps

- No GPU or real model run is planned. Confidence is already present before reduction, and
  this defect is entirely in the pure-Python output shaping after tracking.
- No MARP_API or database suite is planned. Its existing ingest maps numeric
  `keyframe.confidence` and stores null otherwise; issue #9 requires no API or schema change.
- The tests do not freeze exact padded coordinates. The source diff is additive at the output
  dictionary, while the existing pipeline regression keeps the padding path active.

## Manual steps

None. The defect and its null edge case are observable without hardware, media, or a database.

<!-- There is deliberately no "walkthrough videos" section here, and adding one back is a
     mistake. A narrated walkthrough is **not a verification step and never belongs in a
     plan** -- it is recorded only when the human specifically asks for one. A heading here
     invited every plan to promise a video nobody had asked for, which is why it is gone.
     The walkthroughs still assert as they go; that is about not filming a broken app, not
     about them being evidence a phase is done. -->

---

## Results

<!-- Appended by `marp verify run`. Real output, including failures, verbatim. -->
