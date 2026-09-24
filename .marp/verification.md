# Verification — marp-inference-worker#49: a track's species comes from the evidence

## What each test proves

| Requirement | Test | Tier | Proves |
| --- | --- | --- | --- |
| R1, R5 | `test_track_accumulator` › `test_a_predicted_frame_is_no_evidence` | unit | a predicted frame carries no class and does not vote |
| R2 | › `test_species_is_the_evidence_not_the_first_frame` | unit | a weak first frame no longer decides the track |
| R2 (A1) | › `test_species_is_weighted_by_confidence_not_counted` | unit | 11 confident frames beat 50 near-zero ones, where a majority vote would not |
| R5 | › `test_a_track_nothing_detected_is_unknown` | unit | an all-predicted track is still "Unknown" |
| A4 | › `test_a_tie_goes_to_the_class_seen_first` | unit | the decision is deterministic |
| A3 | › `test_another_class_score_is_not_reported_as_this_species` | unit | another class's score is dropped; the species' own is kept |
| R4 | › `test_the_running_species_holds_through_one_dissenting_frame` | unit | the running decision ignores one dissenting frame |
| R2, R3, R6 | `test_tracking_pipeline` › `test_a_track_is_recorded_as_its_evidence_not_its_first_frame` | real tracker + engine matcher | the observation and every keyframe carry the decided species, through the unchanged reduction |
| A3 | › `test_observation_confidence_is_null_when_that_frame_saw_another_species` | real tracker + engine matcher | the observation reports no score from another class |
| R4 | › `test_the_watch_window_label_holds_through_a_dissenting_frame` | real tracker + engine matcher | the label the watch window receives does not flip |

## Known gaps

- The watch window itself was not opened. The tests assert the label the engine hands it.
- A3 is applied to keyframe confidences as well as the observation's. A keyframe on a frame
  detected as another class now reports no score. That is the same answer, one level down.

---

## Results

**2026-09-24, branch `49-species-from-the-evidence`.**

Full `pytest`: 282 passed, 4 skipped (platform skips, as before).

**Red first.** With develop's `track_accumulator.py` and `tracking_engine.py`, 8 of the 10
new tests failed. The two that passed, the all-Unknown track and the tie, describe behaviour
that is meant to be unchanged.

**On this machine's GPU, once, not committed.** Job 6268's piece (MBARI benthic model,
frames 1500–6000) was run twice through the coordinator on this machine: job 13785 on
develop's worker and job 13786 on this branch.

```
 develop_obs | branch_obs | matched | species_changed | confidence_now_null
          67 |         67 |      67 |               6 |                   5

 Bony fishes     -> Flatfish
 Bony fishes     -> Sharks
 Rays and Skates -> Sea stars
 Sea fans        -> Feather stars and sea lilies
 Sea fans        -> Squat lobsters
 Sea stars       -> Feather stars and sea lilies

branch observations whose keyframes disagree: 0
```

Tracking is unchanged, so every track matched one for one. 6 of 67 were recorded as a
different species, and no observation's keyframes disagree with it.
