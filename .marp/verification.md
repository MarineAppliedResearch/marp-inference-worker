# Verification — MarineAppliedResearch/marp-inference-worker#3

Worker half of MARP's distributed GPU compute: the dial-out loop, the engine contract, and
the detect → track → reduce pipeline.

**This plan was written on a machine with no GPU**, and that shapes everything below: every
requirement that could be proved without one is proved, and the ones that could not are
named in *Known gaps* rather than claimed. The tier that matters most below a GPU is not
`unit` — it is `runner`, which launches a real child process over a real pipe, because
that is the only tier below a GPU that can watch a job be cancelled.

**A CUDA machine and a reachable Jellyfin server have since become available**, and some of
those gaps are now closed. *Results — real model over real video* at the end of this file
is the authority on what has actually been observed; where it and *Known gaps* disagree,
it wins, and the gap rows it closes say so.

## Tiers

| Tier | What it is | Cost |
| --- | --- | --- |
| `unit` | one module, in-process | milliseconds |
| `import` | one module imported first in a clean interpreter, one subprocess each | ~10 s total |
| `runner` | the real job loop against a fake coordinator: real child process, real pipe, real heartbeats, real stop file. Mock engine, so no GPU and no model. | seconds |
| `pipeline` | the real vendored ByteTrack, the real accumulator, the real ported reduction, over synthetic detections. No GPU. | seconds |
| `source` | a structural assertion over the code itself — an import graph, a syntax tree, a pin | milliseconds |
| `gpu` | real YOLO, real video, real CUDA | minutes, and **not in CI** — run by hand, see *Results — real model over real video* |
| `media` | the seek analysis over a fake capture, in-process. No stream, no GPU. | milliseconds |

`source` needs a word of justification, because a test that reads code rather than running
it is usually a smell. Three things in this task are genuinely structural claims and
cannot be observed behaviourally: that a reduction was ported *verbatim*, that push is
*unrepresentable*, and that an engine *cannot* reach the coordinator. Each is an assertion
about what does not and cannot happen, and a behavioural test can only ever sample.

## What each test proves

| Requirement | Test | Tier | Proves |
| --- | --- | --- | --- |
| R1 | `test_job_runner.py::test_worker_never_receives_a_push_address` | source | No field of the job spec or attempt envelope is a callback address. The one `url` in the whole contract is the model artifact locator — an outbound fetch — and the test asserts that direction explicitly. |
| R1 | `test_status.py::test_pause_and_resume_change_what_status_reports` | unit | The loopback API serves the operator: pause and resume, with the reason recorded. |
| R1 | `test_engine_contract.py::test_no_engine_imports_the_coordinator_or_the_runner` | source | The outbound-only boundary is closed at the import graph, not just by convention. |
| R2 | `test_job_runner.py::test_worker_enrols_with_its_real_discovered_hardware` | runner | Nothing about the machine is passed in. Slots, CUDA devices, engines, reductions and the full hardware snapshot are all read off the host at enrolment. |
| R3 | `test_job_runner.py::test_identity_survives_a_restart` | runner | A second runner over the same state directory is the same worker and does **not** re-enrol. |
| R3 | `test_job_runner.py::test_re_enrolment_keeps_the_local_id` | runner | A forced re-enrolment carries the same machine-local id, so MARP recognizes the machine rather than accumulating duplicates. |
| R4 | `test_job_runner.py::test_job_runs_in_a_child_process_and_reports_success` | runner | Poll → lease → run → report, end to end. All three lease fields ride on the terminal report. |
| R4 | `test_job_runner.py::test_progress_reaches_the_coordinator_and_advances` | runner | Heartbeats carry progress that **increases**. Asserts movement, not presence — a stuck counter fails. |
| R5 | `test_job_runner.py::test_cancel_in_a_heartbeat_stops_a_running_job` | runner | **The important one.** A heartbeat answering `cancel` reaches a live child process mid-run, stops it, and it reports itself cancelled with the work it had done. Asserts `0 < frames_processed < 2000`, so a job that had already finished cannot pass. |
| R5 | `test_job_runner.py::test_abandon_in_a_heartbeat_also_stops_the_job` | runner | `abandon` stops a running job too — it is what a lease-epoch mismatch is answered with. |
| R5 | `test_job_context.py::test_should_stop_is_false_until_the_stop_file_appears` | unit | The engine-side mechanism, in isolation. Named separately because it proves the mechanism and *not* the wiring. |
| R5 | `test_job_context.py::test_should_stop_stays_true_once_set` | unit | A stop is not retractable, so an unwinding engine cannot be told to carry on. |
| R6 | `test_job_runner.py::test_slots_limit_how_many_jobs_run_at_once` | runner | Two slots take two jobs on **distinct** slot indices and refuse a third. |
| R6 | `test_device.py::test_auto_resolves_to_the_slots_own_gpu` | unit | Slot 1 resolves to `cuda:1`, so two jobs do not both land on GPU 0. |
| R7 | `test_engine_contract.py::test_every_job_engine_implements_the_contract` | unit | One way in. Every dispatchable engine has `engine_name`, `describe`, `preflight`, `run`. |
| R7 | `test_job_context.py::test_context_offers_exactly_the_six_documented_calls` | unit | *Exactly* the six. A seventh public call would be a seventh thing an engine could learn. |
| R7 | `test_engine_contract.py::test_context_protocol_is_the_only_thing_engines_are_given` | unit | Same assertion at the protocol engines are typed against, which can drift from the implementation because a Protocol is structural. |
| R7 | `test_engine_contract.py::test_no_engine_reads_a_token_or_a_coordinator_address` | source | No engine reads `os.environ`, and none mentions a token or coordinator. An engine that could read the token would be inside the trust boundary. |
| R7 | `test_engine_contract.py::test_model_manager_no_longer_probes_for_method_names` | source | The two `hasattr` reach-throughs are gone and the typed accessor replaced them. |
| R7 | `test_engine_contract.py::test_frame_only_engine_cannot_be_dispatched_a_job` | unit | The capability split is real and enforced by `isinstance`, both directions. |
| R8 | `test_tracking_pipeline.py::test_infer_stream_is_a_generator` | source | `infer_stream` and `iter_frame_range` are both generator functions, `infer_stream` feeds Ultralytics one frame at a time, and the tracking engine consumes the reader **without** listing it — the mistake that would quietly undo all of it. |
| R9 | `test_job_runner.py::test_two_piece_split_covers_every_frame_exactly_once` | runner | The half-open convention, asserted rather than documented. Two adjacent pieces `[0,300)` and `[300,600)` cover `0..599` exactly once: no frame in both, none missing, and the first piece does **not** touch frame 300. |
| R9 | `test_job_runner.py::test_malformed_spec_is_rejected_without_launching_anything` | runner | An inverted range is refused at the schema, before a child process exists. |
| R9 | `test_job_runner.py` (all job tests) | runner | Every spec carries a range, including in the fixture — nothing special-cases a whole video. |
| R9 | `test_seek_verification.py::test_a_seek_that_lands_short_is_reported_as_an_offset` | media | A capture that *reports* frame 300 while decoding frame 267 is caught and the offset named. Believing the reported position is the failure this exists for, and no other test in this repository can see it. |
| R9 | `test_seek_verification.py::test_locating_a_run_narrows_what_locating_one_frame_cannot` | media | **Why the comparison is by run.** On video that repeats inside a window, one frame occurs at five indices and a run that reaches past the window occurs at one. A single-frame comparison invents offsets that are not there — it did, on the first real measurement. |
| R9 | `test_seek_verification.py::test_a_short_landing_inside_a_repeating_stretch_is_still_caught` | media | The dangerous case: the seek lands short *and* the frame it landed on is pixel-identical to the target. Still reported as an offset. |
| R9 | `test_seek_verification.py::test_summarize_will_not_pass_on_ambiguous_results_alone` | media | A set in which nothing was unambiguous must not report green — it says nothing about the decoder. |
| R9 | `test_seek_verification.py::test_summarize_of_nothing_does_not_pass` | media | A measurement that ran on nothing does not look green. |
| R9 | `scripts/verify_seek_accuracy.py` | gpu/media | The measurement itself, against a real stream. This is manual step 3, as a command rather than a procedure. |
| R10 | `test_tracking_pipeline.py::test_full_pipeline_produces_one_observation_for_one_animal` | pipeline | The three stages, joined, with the **real** ByteTrack: id assigned, track gathered, reduced to labelled keyframes, shaped as an observation. |
| R10 | `test_tracking_pipeline.py::test_vendored_bytetrack_is_importable_and_usable` | pipeline | The vendored tracker actually tracks. Not skipped when absent — see *Regression coverage*. |
| R10 | `test_track_accumulator.py::test_accumulated_track_is_in_the_shape_the_reduction_reads` | unit | The join between stage 2 and stage 3, which would otherwise fail only on a real job. |
| R10a | `test_track_accumulator.py::test_track_still_open_at_range_end_is_closed_there` | unit | A track in frame at the seam ends anyway, marked `range_end` so the truncation is attributable. |
| R10a | `test_track_accumulator.py::test_range_end_leaves_nothing_to_hand_forward` | unit | Nothing survives the seam to be carried or stitched. |
| R10a | `test_track_accumulator.py::test_two_ranges_produce_two_independent_observations` | unit | One animal crossing a seam becomes two observations — asserted as the accepted outcome. Both carry track id 1 from two independent trackers, which is *why* ids must never be compared across ranges. |
| R10b | `test_keyframe_reduction.py::test_ported_reduction_is_the_line_814_definition` | source | The ported reduction is syntax-tree-identical to the line-814 legacy definition, docstrings excluded. |
| R10b | `test_keyframe_reduction.py::test_ported_directional_pad_is_the_line_930_definition` | source | Same for `_apply_directional_pad`. |
| R10b | `test_keyframe_reduction.py::test_reduction_defaults_match_the_live_definition` | unit | `vel_window=3`, `speed_gain=0.25` — the numbers that distinguish line 814 from line 628. Fails loudly if the wrong variant is ever wired in. |
| R10b | `test_keyframe_reduction.py::test_registry_refuses_an_unknown_reduction` | unit | An unknown name or version is refused, **not** substituted. A fallback would attribute an observation to a rule that did not produce it. |
| R10b | `test_tracking_pipeline.py::test_full_pipeline_...` | pipeline | The reduction name and version travel on the observation. |
| R11 | `test_job_runner.py::test_artifact_is_uploaded_only_when_the_coordinator_asks` | runner | Hash offered first; bytes sent only when the coordinator says it lacks them. |
| R11 | `test_job_runner.py::test_artifact_is_not_uploaded_when_already_held` | runner | Nothing is sent when it already has it — the saving is real. |
| R11 | `test_job_context.py::test_publish_artifact_hashes_the_real_bytes` | unit | The hash is computed by the process that wrote the file, and no detections are in the event. |
| R11 | `test_tracking_pipeline.py::test_results_are_written_as_one_json_object_per_line` | unit | The results file streams both ways; a single JSON array would have to be held whole. |
| R12 | `test_status.py::test_status_reports_unenrolled_worker_honestly` | unit | An unenrolled worker says so instead of being given a fabricated id. |
| R12 | `test_status.py::test_status_reflects_a_running_job` | unit | The job list is the runner's, and slot accounting is derived from it so the two cannot disagree. |
| R12 | `test_status.py::test_status_lists_the_engines_actually_registered` | unit | Read from the registry, including whether each needs a GPU. |
| R12 | `test_status.py::test_status_version_is_not_a_literal` | unit | The version is the installed package's. |
| R13 | `test_model_cache_hashing.py::test_wrong_hash_is_refused` | unit | **The test that would have caught the original defect.** Good bytes, wrong stated hash — the old code accepted it silently. |
| R13 | `test_model_cache_hashing.py::test_an_already_cached_artifact_is_verified_not_assumed` | unit | The early-return path is guarded too — the branch the defect was most dangerous in. |
| R13 | `test_model_cache_hashing.py::test_a_mismatched_artifact_is_removed_so_a_retry_can_refetch` | unit | A bad file is not left cached to fail identically forever. |
| R13 | `test_model_cache_hashing.py::test_download_uses_a_timeout` | source | A timeout is passed, and `urlretrieve` — which cannot take one — is no longer imported. |
| R13 | `test_model_cache_hashing.py::test_download_lands_on_a_partial_name_first` | source | An interrupted fetch leaves nothing a later run would trust. |
| R14 | `test_device.py::test_auto_refuses_rather_than_falling_back_to_cpu` | unit | `auto` on a GPU-less machine is a **refusal**, with an actionable message. Real behaviour on this machine, not simulated. |
| R14 | `test_device.py::test_missing_device_is_treated_as_auto_and_also_refuses` | unit | The rule cannot be sidestepped by omitting the field. |
| R14 | `test_device.py::test_explicit_cuda_index_is_checked_against_what_exists` | unit | `cuda:3` on a two-GPU machine fails now, not later. |
| R14 | `test_engine_contract.py::test_tracking_engine_refuses_a_gpu_job_on_a_machine_without_one` | unit | Preflight refuses before any work. Branches on the real device count, so it is meaningful on both kinds of machine. |
| R14 | `test_job_runner.py::test_job_that_cannot_run_is_refused_before_it_starts` | runner | Reported failed with the reason named, and the slot is freed. |
| R15 | `test_job_runner.py::test_a_job_in_flight_at_restart_is_reported_lost` | runner | A record left by a killed worker is reported `lost` with its lease and progress, the record is cleared, and the worker goes on to take work. |
| R15 | `test_job_runner.py::test_in_flight_record_is_written_and_cleared_around_a_job` | runner | Written before launch, cleared on completion — so a clean run does not report a completed job lost. |
| R16 | `test_engine_contract.py::test_ultralytics_environment_is_set_by_the_worker_not_the_engine` | source | All three variables set in the child entry point, and asserted to be applied **before** the engine registry import — the only place it can work, since Ultralytics caches them at import. |
| R16 | `test_engine_contract.py::test_ultralytics_version_is_pinned_exactly` | source | An `==` pin on 8.4.x in `pyproject.toml`, and the installed version matches it, so the pin is not aspirational. |
| R16 | `test_job_context.py::test_metrics_mapping_is_not_schemad` | unit | Arbitrary metric names survive — Ultralytics' names differ by task and version. |

## Requirements with no test

None. R1–R16 each have at least one row above.

Two are proved only structurally and are weaker for it, which is stated rather than hidden:

- **R8** is proved by `source`. That the generator is *not* materialized is provable by
  reading the code; that a ten-hour video does not exhaust RAM is a `gpu` claim.
- **R1**'s no-inbound-listener property is proved by the absence of a field and the absence
  of an import. Nothing here observes a real coordinator failing to connect back.

## Edge cases

| Case | Traces to |
| --- | --- |
| Track exactly at the 30-frame minimum span is **kept** | An off-by-one to `<=` silently drops a whole category of real observations. |
| Empty frame produces a `(0, 5)` array, not `[]` | An empty list gives ByteTrack's own indexing the wrong shape and raises, rather than reporting no tracks. |
| Unknown heartbeat action treated as `continue` | An older worker against a newer coordinator must keep working, not stop on every job. |
| Unrecognized tracker params ignored rather than set | A typo in a job spec must not become a tracker setting nobody meant. |
| Hash comparison case-insensitive | Upper-case hex carries no information; refusing it would be a false alarm. |
| Spec with no sha256 is measured and reported **unverified** | The frame routes allow it; the job runner requires a hash. The distinction must not be blurred into "verified". |
| Video shorter than its range ends cleanly | A range can be clipped by the true length. Reported as `frames_short_of_range`, not as an error. |
| Frame rate reported as 0 | Seen on Jellyfin streams. Floored to 1, or every timecode is an infinity. |
| Failed heartbeat does not kill a running job | The network may return before the lease expires; killing throws away valid work. |
| Track with no class match on a frame labelled `Unknown`, not mislabelled | A predicted track with no detection behind it has no known class. |
| Observation-time fallback to the middle frame | An animal that never crosses the counting line was still seen. |

## Regression coverage

Three defects were found during this work. Each has a named test at a tier that can see it.

1. **The vendored ByteTrack could not run at all.** It uses `np.float`, `np.int` and
   `np.bool`, removed in numpy 1.24. The *import succeeded* and the failure appeared only
   on the first frame with something on it, inside `STrack.__init__`.
   → `test_tracking_pipeline.py::test_vendored_bytetrack_is_importable_and_usable`, which
   asserts a real `update()` call rather than the import. Deliberately **not** skipped when
   ByteTrack is missing: a skipped suite looks green.

2. **A circular import between the engines and the models package.**
   `models/__init__.py` re-exports `model_manager` → `engine_registry` → the engine
   modules, so an engine importing `ModelSpec` at runtime closed the loop. It was invisible
   under normal collection order and appeared only when a test imported an engine module
   first — breakage whose existence depended on test ordering.
   → `test_module_imports.py`, which imports each of 27 modules **first** in its own clean
   interpreter. One subprocess per module is slower than a loop and is the only way to test
   import order rather than whatever order pytest produced.

3. **An artifact event's `role` field was named `kind` and silently clobbered the event
   type.** `_emit` merges its fields over `{"seq", "kind", "at"}`, so the artifact event
   arrived with `kind` set to `"observations"` and the parent could not tell an artifact
   event from a log one.
   → `test_job_context.py::test_publish_artifact_hashes_the_real_bytes`, which asserts
   `kind == "artifact"` **and** `role == "observations"` separately.

4. **The frame-range convention diverged between the worker and the coordinator.** This
   worker read the range as half-open; MARP_API had implemented both bounds inclusive,
   which drops one frame at every piece boundary — silently, because each piece looks
   complete on its own. A ten-hour video in ten pieces would have lost nine frames with
   every other test on both sides still green. Settled half-open, `[start_frame,
   end_frame)`, on 2026-09-09; MARP_API is being corrected to match.
   → `test_job_runner.py::test_two_piece_split_covers_every_frame_exactly_once`, at the
   `runner` tier because the schema can only reject an inverted range — whether a job
   actually processes `end_frame` is a question about what the engine did, so it is read
   off two real jobs' results files.

   Checked for vacuity rather than assumed: mutating the mock engine to
   `range(start, end + 1)` fails it twice over, independently, on
   `assert summary["frames_processed"] == 300` (`301 == 300`) and on
   `assert max(first) == 299` (`300 == 299`). Both the count and the frame indices catch
   it, so relaxing either one still leaves the convention guarded.

Two of the fixed pre-existing defects also get regression tests, because the fix is
otherwise unobservable — the broken and fixed code behave identically on correct input:

- `model_cache` computing a sha256 and never comparing it → the whole of
  `test_model_cache_hashing.py`.
- `/status` returning a fixed literal → the whole of `test_status.py`, which replaces the
  old test that asserted that literal and would have passed on any worker in any state.

## Known gaps

Stated plainly. **G4 is not satisfied for the rows below** and cannot be in this
environment.

**Needs a GPU — nothing here observes any of it:**

- ~~Real YOLO inference. `infer_stream` has never run against a real model. Its
  normalization of Ultralytics `Results` objects — `boxes.xyxy`, `boxes.conf`, `boxes.cls`,
  `orig_shape` — is written against the 8.4 API and is **unverified against a real Results
  object**.~~ **Closed 2026-09-09**: verified against ultralytics 8.4.145 on CUDA. All four
  attributes exist and carry what the code assumes. See the results section.
- `load_weights` placing a model on a CUDA device, and `unload_all` actually releasing VRAM.
- Device resolution against real hardware. Every `test_device.py` case with more than zero
  GPUs monkeypatches the count; the *zero* case is the only one that is real here.
- Memory behaviour over a long range. R8 is proved structurally, not measured.
- Multi-GPU. The decision to read progress from `results.csv` where Ultralytics' re-exec
  loses callbacks is **not implemented and not tested** — it is a training-path concern and
  Milestone 1 is inference.

**Needs a real coordinator:**

- Every coordinator call is exercised against `FakeCoordinator`. The real HTTP shapes of
  the five `/api/v2/gpu/…` routes are **unverified** — A1 is a written agreement, not a
  tested one. The first end-to-end run against MARP_API is where a field-name mismatch will
  surface.
- Long-poll behaviour: a 204 on timeout, and the interval fallback. The fake answers
  immediately, so the timeout path is untested.
- `report_result` idempotency on retry.
- The artifact upload target's real shape (`url`, `method`, `headers`).

**Needs a real Jellyfin server:**

- `VideoSourceResolver` resolving a job's `source_name` to a stream. Every runner test
  supplies `video_source_url` directly. The resolver is pre-existing and unchanged.
- ~~OpenCV opening a Jellyfin stream and seeking to a range start. **Seeking is the
  specific risk**: `CAP_PROP_POS_FRAMES` on a long-GOP stream can land on the wrong frame,
  which would silently offset every observation in the range.~~ **Closed 2026-09-09** for
  direct-played MP4 over Jellyfin's `?static=true` endpoint: measured exact on 22 targets
  across two videos, one of them 38,159 frames. It is **not** closed for a transcoded
  stream, which nothing here has yet met — see the results section.

**Not covered for other reasons:**

- Killing a wedged child after the stop grace period. The path exists; no test wedges a
  child for 60 s to reach it.
- A child dying without a terminal event (OOM kill, native crash). The `stderr` fallback is
  written and untested.
- Two workers on adjacent ranges of one real video. The seam is proved with two
  accumulators standing in for two workers, which is the logic but not the deployment.
- `worker_main.main()` has never been run. The token/address reads and the loopback bind
  are unexercised — `uvicorn.run` is called with `host="127.0.0.1"` and no test asserts the
  socket is unreachable from off-machine.
- Ruff is **not clean**, and was not before this work: pre-existing files carry 35 findings
  under the configured rules. Two findings remain in files touched here and are left
  deliberately — one in the pre-existing `frame_source.py`, and one unused unpacked variable
  inside the *verbatim-ported* `pick_observation_time`, where changing it would break the
  port.

## Manual steps

For the GPU machine, in order. Each says what to expect.

1. **Build the environment.**
   ```
   py -3.12 -m venv .venv312
   .venv312\Scripts\python -m pip install -e ".[dev]"
   ```
   `cython-bbox` has no Windows wheel and compiles. Expect it to succeed with the Visual
   Studio "Desktop development with C++" workload installed, and to fail with
   `error: Microsoft Visual C++ 14.0 or greater is required` without it.

2. **Confirm the GPU is visible.**
   ```
   .venv312\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
   ```
   Expect `True` and a non-zero count. Then run `pytest -q` — every `test_device.py` case
   still passes, but now takes the GPU branch of
   `test_tracking_engine_refuses_a_gpu_job_on_a_machine_without_one`, which asserts the
   preflight does **not** refuse. If it refuses on a GPU machine, the check is too strict.

3. **Frame-range seek accuracy.** The highest-risk untested thing, and now a command:
   ```
   .venv312\Scripts\python scripts\verify_seek_accuracy.py <jellyfin_item_id>
   ```
   It decodes the stream sequentially as ground truth, then seeks to each target on a
   freshly opened capture and reports where each landed. Expect `PASSED` and an exit code
   of 0. A `verdict` of `offset` on any target means `CAP_PROP_POS_FRAMES` is landing
   short and **every observation in every non-zero range is offset** — stop and report it.
   `exact_ambiguous` is not a failure: it means the video repeats over the whole decoded
   run, which is a property of the content.

   Then the same question through two real jobs, which is what actually matters: one over
   `[0, 300)` and one over `[300, 600)`. The ranges are half-open and share the bound 300,
   so the first job must not process it — in its results the highest `framenum` must be
   ≤ 299, and in the second's the lowest must be ≥ 300. The convention is asserted in
   `test_two_piece_split_covers_every_frame_exactly_once`; these two steps check the
   *decoder* honours it against a real stream, which no in-process test can.

   Results of both, on a real CAMPA2021 clip: *Results — real model over real video*.

4. **A real inference job.** Point the worker at MARP with its token, queue one tracking
   job over a short range, and watch `/status` on the worker machine.
   Expect: progress advancing, `nvidia-smi` showing one process on slot 0's GPU, and a
   results file whose `sha256` the coordinator reports as verified.

5. **Cancel a real job.** Queue a job over a long range, let it reach a few hundred frames,
   cancel from MARP.
   Expect: it stops within one heartbeat interval, reports `cancelled`, and the results file
   contains the observations found **before** the cancel — not zero, and not all of them.

6. **Kill the worker mid-job.** Queue a job, let it start, kill the worker process
   outright. Restart it.
   Expect: the job is reported `lost` on startup, `data/worker/in-flight.json` is gone
   afterwards, and the worker takes new work without being reconfigured.

7. **Refuse an impossible job.** Queue a job naming a reduction this worker does not have.
   Expect: refused immediately, with `v9_imaginary` (or whatever was named) in the reason,
   and no child process started.

## Walkthrough videos

None. Nothing in this task has a visual surface — the worker is a headless loop and its
only UI is a JSON endpoint on loopback. A video would narrate log lines, which is exactly
the pattern the doctrine warns about.

---

## Results

Run on the development machine: Windows 11, Python 3.12.10, **no NVIDIA GPU**,
torch 2.14.0 (CPU), ultralytics 8.4.145, numpy 2.5.3, ByteTrack vendored + cython-bbox
0.1.5 built against MSVC 14.51.

Command: `.venv312\Scripts\python -m pytest -q`

```
........................................................................ [ 52%]
..................................................................       [100%]
============================== warnings summary ===============================
.venv312\Lib\site-packages\fastapi\testclient.py:1
  StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is
  deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

.venv312\Lib\site-packages\starlette\testclient.py:53
  DeprecationWarning: The anyio.abc.BlockingPortal alias is deprecated, use
  anyio.from_thread.BlockingPortal instead.
    _PortalFactoryType = Callable[[], AbstractContextManager[anyio.abc.BlockingPortal]]

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
138 passed, 2 warnings in 22.44s
```

Both warnings are from installed libraries, not from this code.

**Failures seen and fixed during the run**, recorded because the doctrine asks for real
results including failures:

- `test_status.py::test_status_returns_worker_status_contract` — failed with
  `AssertionError: assert None == 'dev-worker'`. Expected: it asserted the hard-coded
  literal R12 required deleting. The test was replaced, not adjusted.
- `test_keyframe_reduction.py` — the two verbatim-port comparisons failed on docstrings,
  which `ast.unparse` preserves and which the port replaced with this repository's `#`
  comment style. A recursive docstring stripper was added so the comparison is of the
  arithmetic, not the prose.
- `test_job_context.py::test_context_satisfies_the_job_context_protocol` — failed with
  `ImportError: cannot import name 'BaseEngine' from partially initialized module`. A real
  circular import, fixed at the root and now guarded by `test_module_imports.py`.
- `test_job_runner.py` — 13 failures, all from the fixture. Two were the runner behaving
  correctly and the test being wrong: the model fetch is mandatory and hash-verified, and
  three jobs sharing one model filename meant the last job's bytes were on disk when the
  first two were checked, so both were refused on a hash mismatch. The check was right and
  the fixture was wrong.
- `test_model_cache_hashing.py::test_download_uses_a_timeout` — asserted `"urlretrieve" not
  in` the module source and matched the module's own comment explaining its removal.
  Narrowed to the import and the call.

Ruff, for the record — `ruff check src/marp_inference_worker tests`:

```
Found 107 errors.
```

Of those, 35 are in pre-existing files under the same rules, so the repository has never
been clean under this configuration. Restricted to real correctness rules
(`--select F,E9,RUF015,RUF059,PLW1510,SIM115`) after fixes: **2 findings**, both left
deliberately and named in *Known gaps*.

---

## Results — real model over real video

Run 9 Sep 2026 on the development machine, which by then had a GPU and a reachable video
server: Windows 11, Python 3.12.10, **NVIDIA GeForce RTX 5060 Laptop GPU**,
torch 2.11.0+cu128, torchvision 0.26.0+cu128, ultralytics 8.4.145, numpy 2.5.3,
OpenCV 5.0.0 (FFMPEG backend), ByteTrack vendored + cython-bbox built against MSVC 14.51.

Model: **stock `yolov8n`**, not the MARP fish model. Deliberate. Its classes are COCO, so
it was expected to find no fish, and the point of the run was the machinery — decode, seek,
detect → track → reduce, the reduction, and the results file — not the biology.

`yolov8n.pt`, 6,549,796 bytes,
sha256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`, fetched
deliberately from `ultralytics/assets` rather than by an implicit download inside a job,
because the child process sets `YOLO_OFFLINE` (R16).

Video: CAMPA2021 / Dive 119 / `20210614_153127_Fwd`, Jellyfin item
`0580f9fc5d3e0633998c3b02a236358c`. 39.168 s, 1920x1080 h264, 25 fps, 979 frames.
The long-video seek check used `20210614_153913_Fwd` from the same dive, 38,159 frames.

No coordinator was involved in any of this. Jobs were driven through the real child entry
point, `python -m marp_inference_worker.jobs.child_main <envelope.json>`, with the envelope
written by hand — which is exactly the contract `run(ctx, spec)` was built to allow.

### Direct play, not transcode

Established rather than assumed, because a transcoded stream need not have the same frame
count as the source and "frame 300" would then mean something different to every viewer.
Four facts, all from the running server:

- `PlaybackInfo` reports the source with `Protocol: File`, `SupportsDirectPlay: true`, and
  **both `TranscodingUrl` and `DirectStreamUrl` empty** — Jellyfin saw nothing to convert.
- The worker's stream url is the `?static=true` endpoint, which serves the original file.
- `HEAD` on it answers `Content-Length: 4758539`, **byte-for-byte the `Size` Jellyfin
  reports for the source file**, with `Accept-Ranges: bytes` and no `Transfer-Encoding`.
- A `Range: bytes=0-15` request answers `206` with `content-range: bytes 0-15/4758539` and
  the bytes `000000206674797069736f6d…` — an unfragmented MP4 `ftyp isom` box.

A transcode would be chunked, without a content length, without byte ranges, and not the
size of the source. This also explains *why* the seek is exact: ffmpeg can seek by byte
offset in a seekable MP4 with an index, decode from the correct keyframe, and discard up to
the requested frame.

**Frame rate reported as 0 did not occur here** — this stream reports 25.0, matching
Jellyfin's own `AverageFrameRate`. The floor-to-1 guard in `open_video` was therefore not
exercised, and remains a gap.

### Seek accuracy — the headline

`CAP_PROP_POS_FRAMES` landed on **exactly** the frame it was asked for, on every target
tried, on both videos. 22 targets in total. No offset, constant or varying.

The first measurement said otherwise and was wrong, which is worth recording because the
mistake is easy and the wrong answer is alarming. Mapping each decoded frame to *the first
ground-truth index with the same pixel hash* produced this:

```
 target  landed  offset
    250     250      +0
    299     267     -32
    300     267     -33
    301     267     -34
    400     271    -129
    500     500      +0
distinct offsets: [-129, -34, -33, -32, -17, 0]
VERDICT: offset VARIES by target -- not correctable by a constant.
```

That is a measurement artefact, not a decoder fault. **547 of this clip's 979 frames are
pixel-identical to another frame** — only 432 are distinct — in 148 runs of identical
consecutive frames, with a 16-frame period from about frame 250 onward. The seek was
landing on frame 300; frame 300 is simply the same picture as frame 267.

Comparing a *run* of decoded frames instead of one frame settles it:

```
 target  exact?  reported  candidate sequential indices for the frame we got
      1     YES       1.0  [1]
     25     YES      25.0  [25]
    100     YES     100.0  [100]
    150     YES     150.0  [150]
    200     YES     200.0  [200]
    250     YES     250.0  [250]
    299     YES     299.0  [267, 268, 269, 283, 284, 285, 299, 300, 301, 315, ...]
    300     YES     300.0  [267, 268, 269, 283, 284, 285, 299, 300, 301, 315, ...]
    301     YES     301.0  [269, 285, 301, 317, 333, 349, 365, ...]
    302     YES     302.0  [270, 286, 302, 318, 334, 350, 366, 382, 398]
    400     YES     400.0  [271, 272, 273, 287, 288, 289, 303, 304, 305, 319, ...]
    500     YES     500.0  [500]
    600     YES     600.0  [600, 616, 632, 648, 664, 680, 696, 712]
    700     YES     700.0  [588, 604, 620, 636, 652, 668, 684, 700, 716]
    750     YES     750.0  [750]
    900     YES     900.0  [883, 884, 885, 899, 900, 901]
    978     YES     978.0  [930, 946, 962, 978]

exact on 17/17 targets
```

Every remaining candidate sits at the target plus or minus a multiple of 16, which is the
content's own period — not something a broken seek would produce. Where the content is
unique the answer is a singleton and unambiguous, and it is exact there too.

The 25-minute video removes the ambiguity entirely, because every one of its frames is
distinct. Each seek was followed by 40 decoded frames and that run located in the ground
truth:

```
ground truth: 9045 frames, 9045 distinct, 199s

seek    300: reported     300.0  first-frame exact=True  run of 40 occurs at [300]
seek   1500: reported    1500.0  first-frame exact=True  run of 40 occurs at [1500]
seek   3000: reported    3000.0  first-frame exact=True  run of 40 occurs at [3000]
seek   6000: reported    6000.0  first-frame exact=True  run of 40 occurs at [6000]
seek   9000: reported    9000.0  first-frame exact=True  run of 40 occurs at [9000]
```

`iter_frame_range(capture, geometry, 300, 310)` was checked directly: its first frame's
`index` is 300 and its pixels are sequential frame 300.

The analysis is now `media/seek_verification.py` with `scripts/verify_seek_accuracy.py`
over it, so manual step 3 is a command. Against the same clip:

```
  target   reported          verdict  candidates
       1        1.0            exact  [1]
     100      100.0            exact  [100]
     299      299.0  exact_ambiguous  [267, 283, 299, 315, 331, 347, 363]
     300      300.0  exact_ambiguous  [268, 284, 300, 316, 332, 348, 364]
     301      301.0  exact_ambiguous  [269, 285, 301, 317, 333, 349, 365]
     600      600.0  exact_ambiguous  [600, 616, 632, 648, 664, 680]
     900      900.0            exact  [900]

verdicts        : {'exact': 3, 'exact_ambiguous': 4}
distinct offsets: [-32, -16, 0, 16, 32, 48, 64, 80]
PASSED
```

### The real Results object

`infer_stream`'s normalization was written against the 8.4 API and had never met a real
`Results`. **It is correct.** On ultralytics 8.4.145, `result.boxes` is present,
`boxes.xyxy` is `(17, 4)`, `boxes.conf` is `(17,)`, `boxes.cls` is `(17,)`, and
`orig_shape` is `(1080, 1920)` — height first, which is the order `_run_frame_prediction`
assumes. `model.names` is a plain `dict` of 80 COCO ids, so the `.get(class_id, ...)`
lookups work. Nothing had to be changed.

### What stock yolov8n does on ROV video, and why it cannot form a track

Over all 979 frames at `conf=0.01`, the **highest detection confidence anywhere is
0.1797**. Not one frame has a detection above 0.2. The classes are what COCO has to offer
a seafloor: `clock` 1141, `person` 852, `tv` 310, `refrigerator` 238, `surfboard` 176,
`elephant` 172.

At MARP's real settings — `confidence=0.15`, `track_thresh=0.3` — a job over `[0, 300)`
sees **2 detections and produces 0 observations**. That is the honest result and it is not
a failure.

It is worth being precise about *why* no track forms, because it is structural and not a
threshold to nudge. Two gates in the vendored ByteTrack:

- `det_thresh = args.track_thresh + 0.1` (`byte_tracker.py:154`), and a new track is only
  started when its score clears `det_thresh` (`:266`).
- `matching.fuse_score` multiplies IoU similarity by the detection score
  (`matching.py:173`), and `linear_assignment` rejects a cost above `match_thresh`. With a
  score of 0.18 the best achievable fused similarity is 0.18, so the cost is at least 0.82
  and can never clear MARP's `match_thresh=0.7`. **No association across frames is
  possible at any IoU.** Lowering `confidence` alone therefore changes nothing: at
  `track_thresh=0.0`, 979 detections over 300 frames still produced exactly one track,
  which was seen once and aged out.

So the reduction was exercised on real video under a deliberately artificial tracker
configuration — `confidence=0.01`, `track_thresh=0.0`, `match_thresh=0.97` — whose only
purpose was to let COCO-class noise associate. **No code was changed to achieve it**; those
three values are ordinary job params. Everything below is therefore evidence about the
machinery and about nothing else.

### The two-piece seam, on real video

Two runs, `[0, 300)` and `[300, 600)`, same settings, same video.

| | `[0, 300)` | `[300, 600)` |
| --- | --- | --- |
| outcome | `succeeded` | `succeeded` |
| device | `cuda:0` | `cuda:0` |
| frames_expected / processed | 300 / 300 | 300 / 300 |
| frames_short_of_range | 0 | 0 |
| detections_seen | 979 | 1215 |
| observations | 1 | 1 |
| results sha256 | `47267c06…daf191` | `c336d3bd…fb1a70` |
| results bytes | 3329 | 734 |
| keyframe `framenum` range | **0 … 299** | **300 … 413** |
| track_id | 1 | 1 |
| track_end_reason | `range_end` | `range_end` |

**The seam holds.** The first piece's highest `framenum` is 299 — it never processed frame
300 — and the second piece's lowest is exactly 300. The last metrics event of the first run
is `step=299` and the first of the second is `step=329`, consistent with the same thing.
Both pieces carry `track_id: 1` from two independent trackers, which is exactly why R10a
forbids comparing ids across ranges.

### A results record, verbatim

The whole of the second piece's `observations.jsonl`, one JSON object per line as R11 says:

```json
{"comname": "clock", "count": 1, "tc": "00:00:14", "frame": "7", "video_source": "20210614_153127_Fwd.mp4", "jellyfin_item_id": "0580f9fc5d3e0633998c3b02a236358c", "mediaPosition": "00:00:14.279", "actualPosition": "00:00:14.279", "keyframes": [{"subset": "1", "comname": "clock", "type": "start", "framenum": 300, "x": 0.8039578912681559, "y": 0.2955575830743181, "width": 0.0811110492599446, "height": 0.13830153021656155}, {"subset": "1", "comname": "clock", "type": "end", "framenum": 413, "x": 0.8039542095078498, "y": 0.2955645023890501, "width": 0.08111274314254842, "height": 0.13830199240320623}], "reduction": {"name": "v3_dirpad", "version": "1"}, "track_id": 1, "track_end_reason": "range_end", "observation_frame": 357}
```

The first piece's record is the same shape with 16 keyframes labelled `start`,
`middle` x14, `end`, at framenums 0, 175-226 and 299 — so the reduction really did pick
keyframes rather than pass the track through, and `reduction: {name: v3_dirpad, version:
1}` travels on the row as R10b requires.

### Two things noticed and deliberately left alone

**Normalized keyframe boxes can exceed 1.0.** Some keyframes in the first piece carry a
`width` of 1.2236 — wider than the frame. The engine normalizes ByteTrack's `track.tlbr`,
which is a Kalman *prediction* and is not clipped to the picture.

This is **not new**: `src/old_scripts/object_tracking_live.py:407-410` normalizes
`track.tlbr` exactly the same way and just as unclipped, so MARP's existing observations
already contain values like these and the port is faithful. It is therefore not a new
implementation decision to surface at the gate — but it is worth knowing before anyone
"fixes" it, because clamping would change what is stored and an unclamped value does carry
information: the tracker believed the animal extended past the frame edge.

One small divergence from that line while looking at it: the legacy script does
`map(int, track.tlbr)` and normalizes truncated integers; the worker keeps floats. A
sub-pixel difference in a normalized coordinate, not a semantic one, and the float is the
better value.

**Ultralytics prints to the child's stdout.** The first ultralytics import in a fresh
`YOLO_CONFIG_DIR` writes three lines beginning `Creating new Ultralytics Settings v0.0.8
file` — and since R16 gives every job its own config directory, that happens on **every**
job, on the same stream the child's event protocol uses. It is handled: `_read_events`
records a non-JSON line as stray output rather than failing, and `runner.py:868` collects
it. Verified by seeing the lines arrive and the job still succeed. Noted because it looks
alarming in a log and because anything that tightens that reader would break every job.

### What this proves, and what it does not

Proved, on real hardware against a real stream:

- Jellyfin direct play, opened by OpenCV, with correct geometry.
- Seeking to a frame lands on that frame, on a 979-frame clip and on a 38,159-frame one.
- The half-open range convention is honoured by the *decoder*, not merely by the schema.
- `infer_stream` against a real `Results` object, on CUDA, one frame at a time.
- detect → track → reduce end to end, with the real vendored ByteTrack.
- `reduce_to_keyframes_v3_dirpad` on a real track, producing start/middle/end keyframes.
- The results file, written as JSONL, hashed by the process that wrote it, and handed over
  through `ctx.publish_artifact` with `role: observations`.
- `run(ctx, spec)` driven with no coordinator at all, through the real child entry point.
- The per-job Ultralytics environment (R16) applied before the first ultralytics import.

**Not** proved, and not claimed:

- **Nothing about detection quality.** Stock yolov8n found no fish, as expected, and every
  observation above came from COCO noise under an artificial tracker configuration. The
  MARP fish model has still never run here.
- The tracker's real settings producing a real track. Structurally impossible with this
  model on this video, per the fuse_score arithmetic above.
- Anything coordinator-mediated: lease, heartbeat, progress upstream, cancel, terminal
  report, artifact hand-off. Another agent held MARP_API and its database for the duration,
  so no `/api/v2/gpu/...` route was called. Manual steps 4, 5, 6 and 7 remain open.
- A transcoded stream. Every measurement above is against direct play, and a transcode is
  the case where frame numbering could differ from the source's.
- A frame rate reported as 0. This stream reports 25.
- The model cache's hash verification (R13). Driving the child entry point directly skips
  the parent runner's fetch-and-verify, so the sha256 above was recorded but not checked by
  the worker.

### Test suite

`.venv312\Scripts\python -m pytest -q`, before this work and twice:

```
166 passed, 2 warnings in 69.68s (0:01:09)
166 passed, 2 warnings in 67.34s (0:01:07)
```

After adding `test_seek_verification.py`:

```
180 passed, 2 warnings in 93.97s (0:01:33)
```

The 14 new tests were each checked for vacuity by mutating what they guard. All five
mutations tried went red, including reintroducing the original defect — comparing one frame
instead of a run — which fails three of them:

```
--- mutation: locate_run compares one frame, not the run ---   3 failed, 11 passed
--- mutation: locate_run returns only the first match ---      5 failed,  9 passed
--- mutation: verdict calls an ambiguous match exact ---       1 failed, 13 passed
--- mutation: summarize no longer requires an exact ---        1 failed, 13 passed
--- mutation: summarize no longer fails on an offset ---       1 failed, 13 passed
--- mutation: measure_seek leaks the capture ---               1 failed, 13 passed
```

Two failures were seen and fixed while writing them, recorded because the doctrine asks for
real results including failures:

- `test_a_seek_that_lands_short_is_reported_as_an_offset` and
  `test_summarize_fails_when_any_seek_landed_elsewhere` both failed with an unexpected
  extra offset. The test fixture built frames with a fill of `value % 256`, so a
  400-frame "unique" video contained 144 duplicate frames. The fixture was wrong in
  exactly the way the module under test exists to catch.
- `test_a_short_landing_inside_a_repeating_stretch_is_still_caught` asserted
  `offsets == [-32]` and got `[-48, -32]`, because the run it decodes stays inside the
  repeating stretch and so also matches one period earlier. The assertion was wrong, not
  the code: what matters is that the target is *not* among the matches. Narrowed to that.

Ruff on the three new files reports 3 findings, all of them the repository's own house
style rather than anything new: two `I001` from the comment-per-import-group convention
`AGENTS.md` requires, and one `UP035` for `from typing import Iterable` where the rest of
the tree does the same. The one finding unique to the new code, `PIE808`, was fixed.
