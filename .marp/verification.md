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
.\.venv312\Scripts\python -m pytest tests/test_tracking_pipeline.py -k "keyframe_confidence or predicted_keyframe or full_pipeline"
```

The two new tests reproduce issue #9 at the pipeline tier: before the repair, every reduced
keyframe lacks `confidence`. The existing full-pipeline test protects the reducer registration,
start/end labels, and observation handoff while this output field changes.

Then lint only the changed Python files:

```powershell
.\.venv312\Scripts\python -m ruff check src/marp_inference_worker/reduction/keyframes.py tests/test_tracking_pipeline.py
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

Run 2026-09-13 on Windows with CPython 3.12.11.

### Environment setup and command corrections

The repository had no `.venv312`, and the first approved PowerShell command omitted the
required `.\` executable prefix. These attempts failed before Python or pytest started:

```text
The module '.venv312' could not be loaded. For more information, run 'Import-Module .venv312'.
```

After correcting the prefix, the missing environment was explicit:

```text
The term '.\.venv312\Scripts\python' is not recognized as a name of a cmdlet, function, script file, or executable program.
```

The documented environment was then created with Python 3.12.11 and the repository's
declared `[dev]` dependencies. A sandboxed pytest attempt collected all three selected tests;
the two new tests passed, while the existing `tmp_path` regression could not set up because
the command sandbox denied access to pytest's temp directory:

```text
tests\test_tracking_pipeline.py E..                                      [100%]
E       PermissionError: [WinError 5] Access is denied: 'C:\\Users\\isaac\\AppData\\Local\\Temp\\pytest-of-isaac'
============ 2 passed, 13 deselected, 2 warnings, 1 error in 7.70s ============
```

Giving the same approved pytest command access to a workspace-local temp directory resolved
that environmental failure.

### Focused pipeline verification — PASS

Command:

```powershell
.\.venv312\Scripts\python -m pytest -p no:cacheprovider --basetemp .marp\local\pytest-9 tests/test_tracking_pipeline.py -k "keyframe_confidence or predicted_keyframe or full_pipeline"
```

Output:

```text
============================= test session starts =============================
platform win32 -- Python 3.12.11, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\isaac\Documents\Workspace\MARP\marp-inference-worker
configfile: pyproject.toml
plugins: anyio-4.15.1
collected 16 items / 13 deselected / 3 selected

tests\test_tracking_pipeline.py ...                                      [100%]

====================== 3 passed, 13 deselected in 2.25s =======================
```

### Changed-file lint — FAIL, pre-existing findings only

Command:

```powershell
.\.venv312\Scripts\python -m ruff check src/marp_inference_worker/reduction/keyframes.py tests/test_tracking_pipeline.py
```

Ruff exited 1 with these exact findings:

```text
I001  src\marp_inference_worker\reduction\keyframes.py:25:1  Import block is un-sorted or un-formatted
UP035 src\marp_inference_worker\reduction\keyframes.py:31:1  Import from `collections.abc` instead: `Callable`
I001  tests\test_tracking_pipeline.py:19:1                  Import block is un-sorted or un-formatted
RUF046 tests\test_tracking_pipeline.py:289:13               Value being cast to `int` is already an integer
RUF012 tests\test_tracking_pipeline.py:547:13               Mutable default value for class attribute
Found 5 errors.
[*] 4 fixable with the `--fix` option.
```

The same Ruff command was run against copies of both files from `origin/develop`. It returned
the same five codes on the same source statements:

```text
I001  baseline/keyframes.py:23:1               Import block is un-sorted or un-formatted
UP035 baseline/keyframes.py:29:1                Import from `collections.abc` instead: `Callable`
I001  baseline/test_tracking_pipeline.py:19:1   Import block is un-sorted or un-formatted
RUF046 baseline/test_tracking_pipeline.py:289:13 Value being cast to `int` is already an integer
RUF012 baseline/test_tracking_pipeline.py:547:13 Mutable default value for class attribute
Found 5 errors.
[*] 4 fixable with the `--fix` option.
```

No Ruff finding points to an issue #9 addition. The unrelated baseline findings were left
unchanged under the task's scope rule.
