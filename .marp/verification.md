# Verification — marp-inference-worker#67, training as a job

Two repositories, so two suites. The worker half runs anywhere; the coordinator half needs
a database; and several requirements need a GPU and real data, which is why there is a
script rather than a test for those.

## What each test proves

| Requirement | Test | Tier | Proves |
| --- | --- | --- | --- |
| R1 — a spec can describe a training job | `test_job_spec.py::test_a_training_job_needs_no_video_or_range` | unit | A dataset-shaped spec validates at all; before this it could not be expressed. |
| R1 — and inference is unchanged | `test_job_spec.py::test_an_inference_job_is_still_refused_when_incomplete` | unit | Making three fields optional did not loosen the inference contract. Parameterised over `video`, `range`, `reduction`. |
| R1 — never both shapes | `test_job_spec.py::test_a_job_cannot_be_both_shapes` | unit | An incoherent spec is named rather than silently resolved toward one. |
| R2 — fetched and verified | `test_training_dataset.py::test_a_wrong_archive_is_refused_and_leaves_nothing_behind` | unit | A mismatched archive cannot become a training run, and leaves no entry the next attempt would trust. |
| R3 — the existing engine contract | `test_engine_contract.py::test_every_job_engine_implements_the_contract` | unit | `marp_training` is dispatchable through the same `preflight`/`run` as everything else. |
| R4 — epochs, not frames | `scripts/verify_training_run.py` | **real GPU** | One metric event per epoch, carrying Ultralytics' own metric names. |
| R6 — weights handed over | `scripts/verify_training_run.py` | **real GPU** | A `weights` artifact is published, with the sha256 MARP would have recorded. |
| R7 — the dataset cache is shared | `test_training_dataset.py::test_a_dataset_is_fetched_once_and_then_shared` | unit | A second job reuses the first job's copy. The cost of getting this wrong is #65. |
| R8 — no Jellyfin on the job path | `test_module_imports.py` | structural | The existing boundary still holds with the training package added. |
| R9 — one run, no phases | `scripts/verify_training_run.py` | **real GPU** | One `model.train()`, one set of weights out. |
| R12 — the YAML is made true here | `test_training_dataset.py::test_the_yaml_points_at_where_the_dataset_actually_is` | unit | `train`/`val` come out absolute and pointing at directories that exist. |
| R12 — on real data | `scripts/verify_training_run.py` | **real GPU** | rockfish5's own YAML, which carried absolute paths from the machine that built it. |
| A8 — the credential never lands on disk | `test_job_runner.py::test_the_fetch_credential_never_reaches_the_envelope` | unit | The token appears in no file anywhere in the job workspace. |
| MARP_API — queue a training job | `gpu-training-submit.test.js` | API + database | Seven cases, including that an inference submission is unaffected. |
| MARP_API — serve the dataset | `gpu-dataset-archive.test.js` | API + database | Whole file, closed range, open-ended range, past-the-end, conditional, and both 404s. |
| MARP_API — the run is recorded | `gpu-training-run-record.test.js` | API + database | Submit, lease, epochs, publish — through the real endpoints, ending in `training_runs` and `epochs` rows. |

## Requirements with no test

- **R5 — stop and pause at an epoch boundary.** The mechanism is shared with inference and
  covered there (`test_volunteer_control.py`), but nothing proves it for a *training* run,
  where the wait happens inside an Ultralytics callback rather than a frame loop. Proving
  it means starting a GPU run and stopping it part way.
- **R10 — a training job occupies the machine.** The runner decides slots and the engine
  has no say, so there is nothing in this repository to assert. It needs a worker offered
  two jobs at once.
- **R11 — the ten-minute throttle.** A short verification run improves on nearly every
  epoch, so what is exercised is "publishes when it improves", not "declines to publish
  twice inside ten minutes". A test for the second would have to control the clock.
- **R13 — the fetch does not stall the runner.** Covered only indirectly, by the
  credential reaching the child at all. Showing that heartbeats continue during a
  multi-gigabyte fetch needs two jobs and a slow network.

## Edge cases

- An archive with no `data.yaml`, and one whose members escape the cache directory
  (`filter="data"`), are both covered.
- A dataset recorded as packaged whose file is missing from storage answers 404 rather
  than throwing.
- A replayed event batch does not write an epoch twice. This one matters: a worker
  retrying after a network drop sends exactly that.

## Regression coverage

- `test_an_inference_job_is_still_refused_when_incomplete` and `still queues an inference
  job exactly as before` are the pair that would catch this work loosening inference.
- `test_every_job_engine_implements_the_contract` pins the registered engine set, so
  gaining or losing one is a deliberate edit rather than a silent change.

## Known gaps

- **Nothing has trained on a second machine.** The run below was on ABYSS, which also
  holds the API, the dataset and the weights, so the fetch was local. A volunteer's
  machine pulling 4.9 GB over a home connection has not happened.
- **No model has been registered from the weights.** The worker publishes them; turning a
  published artifact into an `ml_models` row is still `MARP_API#265`.
- **The full loop has not been run end to end.** The API half is proven by its own suite
  and the engine by the script below, but no single run has gone submit → lease → train →
  publish with a real worker process in the middle.
- **The ingest path is untouched**, correctly: a training job produces no observations, and
  `INGESTIBLE_JOB_KINDS` already excluded it.

## Manual steps

```
python scripts/verify_training_run.py \
  --archive <packaged .tar> --weights <starting .pt> --epochs 2 --batch 8
```

Needs a CUDA device, the packaged dataset and starting weights. Several minutes, and about
5 GB of disk for the unpacked cache.

## Results

### The first real training run, 2026-10-20, ABYSS (RTX 4080 SUPER, 16 GB)

`rockfish5` (9,879 train / 1,966 eval, 11 classes) from `rockfish2.pt`, 2 epochs, batch 8,
imgsz 640. **It trained**, on the real archive and the real weights, in 1,194 s.

```
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95
                   all       1966       2050       0.91      0.851      0.925       0.77
    Quillback Rockfish        304        304          1      0.997      0.995      0.836
         Rosy Rockfish        164        164      0.995          1      0.995      0.932
    Vermilion Rockfish        632        632      0.988      0.932      0.991      0.839
        Brown Rockfish        395        395      0.964      0.934      0.988      0.914
       Gopher Rockfish        130        130      0.699      0.377      0.615      0.409
```

What the engine reported: 6 progress events, one metric event per epoch, artifacts
`['checkpoint', 'weights']`, no ignored params, final weights 136,708,218 bytes,
sha256 `2af04a7044c100d7…`. The dataset unpacked from the archive, its `data.yaml` was
rewritten to absolute paths, and Ultralytics read it without being told anything else —
R2, R9 and R12 on real data.

Gopher Rockfish at mAP50 0.615 against 0.99 for Quillback is the data, not the pipeline.

### And the run failed its own check, which is why it was worth running

`epochs_completed` came back as **3 for a 2-epoch run**, and progress read `3/2 epochs`.
Ultralytics calls `on_fit_epoch_end` once per epoch and then **once more**, after it
validates the best weights, with `trainer.epoch` already advanced. The engine counted that
as another epoch.

Left alone it would have written an `epochs` row for an epoch that never ran and a
`total_epochs` that disagreed with the run — in the table the ML Dashboard reads. Nothing
in the unit suite could see it: it needs a real Ultralytics trainer to fire the hook.

Fixed by refusing to count an epoch past the requested total or one already counted, while
keeping the final validation's metrics as the run's result. Covered now by
`test_the_final_validation_is_not_counted_as_an_epoch`, which replays the exact sequence
the run produced, plus a repeat-index variant and a positive control.

### Confirmation run

A second run after the fix, same data and weights, 1 epoch.
