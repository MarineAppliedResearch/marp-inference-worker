# Verification - marp-inference-worker#65, worker storage

The suite covers the decisions. The thing that settles R4 is a measurement against a real
Chrome, recorded at the bottom, because what Chromium fetches for a new profile is not
something a unit test can observe.

## What each test proves

| Requirement | Test | Tier | Proves |
| --- | --- | --- | --- |
| R1 - the profile is per slot | `test_watch_display` - *a slot's profile is reused by every job that runs in it* | unit | Two attempts in one slot resolve to one directory, and it is outside the workspace so removing a finished job cannot take it. |
| R2 - windows stay distinguishable | same file - *two slots never share one profile* | unit | Four slots, four paths. This is why it is per slot and not one for the worker. |
| R4 - no background fetching | same file - *the window fetches nothing it cannot use* | unit | The flags are present, including the one that actually stops the model store, and the cache is capped. |
| R4 - occlusion still disabled | same file - *chromium keeps occluded watch windows rendering...* | unit | Asserted by membership now, because the features are one combined flag. |
| R5 - an accepted result takes its workspace | `test_job_runner` - *an accepted result takes its workspace with it* | runner | A real job through a real child process; the workspace exists while it runs and is gone once MARP takes the result. |
| R5 - a refused result keeps it | same file - *a refused result keeps the workspace it still needs* | runner | The coordinator refuses, the slot frees, and the directory survives for the retry. |
| R6 - leftovers pruned at start | same file - *leftover workspaces are pruned at start* | runner | Driven through `run_forever()`, so it proves the ordering against `report_lost_job()` and not just the method. |
| R7 - nothing outside the root | same file - *tidying never reaches outside the workspace root* | runner | A path outside the root is refused and noted rather than removed. |

Both runner tests were confirmed red before the fix, by mutation: disabling the
`_discard_workspace` call failed *an accepted result takes its workspace with it*, and
removing the `_prune_workspaces()` call failed *leftover workspaces are pruned at start*.

## Requirements with no test

- **R3 - orphan cleanup covers the profile root.** The call is wired and `close_orphaned_windows`
  has its own tests, but nothing asserts that `run_forever` sweeps *both* roots. It matters
  more than it used to: a ghost holding `slot-0` now captures the next launch instead of
  merely looking wrong. Worth adding.

## Known gaps

Both of the gaps this file first recorded -- no real inference job, and whether a
long-lived profile creeps -- were closed by the installed run below. What is left:

- **One slot only.** Two windows at once are covered by the script, but no real job has
  run concurrently with another on this machine.
- Nothing measures a worker's disk over a *day*, which is the number the issue quotes.
  What is measured is one real job and the state a hundred of them left behind.
- **The install still grows by version.** `installation/updater.py` keeps every installed
  version side by side, and after upgrading to 0.1.5 the machine held
  `versions/0.1.4-cu128` at 5,808 MB beside `versions/0.1.5-cu128` at 5,808 MB -- a
  complete second copy of the Python runtime with torch (4,992 MB) and of Chromium
  (807 MB). Confirmed as two real directories rather than the junction double-count
  `AGENTS.md` warns about: different inodes, neither a reparse point. That is named in #65
  as secondary and separate, and on these numbers it is four times the problem the
  workspaces were.

## Results

### The suite, 2026-10-21

```
375 passed, 4 skipped, 1 failed in 62.20s
FAILED tests/test_status.py::test_status_lists_the_engines_actually_registered
```

That failure is **not from this change**: it fails identically on `develop` at `c31dc4b`,
the commit this branch starts from. `/status` answers `engines` in a shape the test does
not expect (`KeyError: 'engine'`). Named and left alone.

### The installed worker, on real jobs, 2026-10-21

**This is the measurement that counts, and it is twelve times better than the synthetic
ones below, because those under-measured the problem.**

ABYSS had been running the pilot build, 0.1.4-cu128, which predates this change. It held
three leftover job workspaces:

```
1429 MB  state/jobs
 482 MB    state/jobs/12753
 482 MB    state/jobs/12754
 466 MB    state/jobs/12755
```

A real workspace is **482 MB**, not the 70-120 MB #65 recorded off a checkout's spare
copies, and the profile is all of it -- `work`, `ultralytics`, `envelope.json` and the
render proof together are under 2 MB, exactly as the issue said. What fills it:

| | 0.1.4, per job | 0.1.5, one per slot |
| --- | --- | --- |
| `component_crx_cache` | 237 MB | 1 MB |
| `SODALanguagePacks` | 194 MB | absent |
| `SODA` | 35 MB | absent |
| `OnDeviceHeadSuggestModel` | 8 MB | absent |
| `optimization_guide_model_store` | -- | absent |
| everything else | ~12 MB | 7 MB |
| **total** | **482 MB** | **8 MB** |

**SODA is Chrome's on-device speech recognition** -- live-caption models, a
speaker-diarization encoder, a 48 MB rank-candidates model. Fetched per job, on a
volunteer's machine, by a window that shows a silent video feed and has no microphone and
no captions. It is the single largest thing in a real profile and **it never appeared in
any synthetic measurement**, because a window that is up for 75 seconds and idle does not
trigger it. That is why the numbers below are too kind to the old behaviour.

After installing 0.1.5 built from `develop` at `9ae9da1`:

```
BEFORE   active 0.1.4-cu128   jobs: 3 dirs, 1429 MB
AFTER    active 0.1.5-cu128   jobs: 0 dirs,    0 MB
```

The start-up prune reclaimed **1.4 GB on a real machine, first run**. Then one real job
through the installed worker -- job 21746, 6,000 frames of Jellyfin video, attempt 19200,
succeeded and ingested at 12:11:16:

```
12:08:36  starting    0/6000   profile=-    jobs=1MB  soda=0
12:09:37  inferring 2130/6000   profile=5MB  jobs=1MB  soda=0
12:10:38  inferring 4470/6000   profile=9MB  jobs=1MB  soda=0
12:11:09  inferring 5790/6000   profile=9MB  jobs=1MB  soda=0
          succeeded  6000/6000  through=8000
          profile: slot-0, 8 MB.  SODA present: 0.  jobs left: 0 dirs, 0 MB
```

So on a real workload: **8 MB against 482 MB, once per slot rather than once per job**,
and nothing left behind afterwards.

### A real window, a real profile, 2026-10-21

**Synthetic, and it understates the fix by about twelve times.** Kept because it is
repeatable on any machine with no worker installed and no GPU, and because it is what
proves the window still draws. Do not quote its numbers as the size of the problem -- the
installed run above is the one to quote.

`scripts/verify_watch_storage.py` on this machine, against the Chrome at
`C:\Program Files\Google\Chrome\Application\chrome.exe`. Three real watch windows in
sequence, each drawing a frame and left up for 75 seconds, then a real job through the
real runner. The baseline window is driven by the same code with the new flags stripped
back out, so the comparison is made by the script rather than quoted from elsewhere.

```
0. BASELINE -- the flags as they were before #65
   drew a frame True
   profile is   61.3 MB
         46.0 MB  optimization_guide_model_store
          8.9 MB  Default

1. First job in slot 0 -- a profile that has never existed
   drew a frame True
   profile is   11.2 MB
          4.9 MB  Default
          4.0 MB  BrowserMetrics

2. Second job in slot 0 -- the same profile, reused
   drew a frame True
   profile is   11.2 MB
   same directory as the first: True

3. Where it all lives
      11.2 MB  state/chromium-profiles
       0.0 MB  state/jobs

4. The workspace lifecycle, through the real runner
   before start  ['attempt-a', 'attempt-b', 'attempt-old-1', 'attempt-old-2', 'attempt-old-3']
   after prune   []
   job running   ['attempt-demo']
   reported      succeeded
   after result  []
   workspace survived: False
   last notice   removed 5 leftover job workspace(s)
```

And two slots at once, which a single-slot run cannot see:

```
5. Two slots at once -- the case per-slot profiles exist for
   slot 0 profile  slot-0
   slot 1 profile  slot-1
   different       True
   two processes   True  (30244, 8168)
   both drawing for 30s -- two windows should be on screen
   slot 0 window alive True
   slot 1 window alive True
   closing slot 0 only...
   slot 0 window alive False
   slot 1 window alive True
   slot 1 still draws  True
   profiles now    ['slot-0', 'slot-1']
         15.2 MB  slot-0
         11.2 MB  slot-1
```

**Closing one job's window left the other's running and still drawing.** That is the whole
case for per slot rather than one profile for the worker: two `--user-data-dir`s means two
processes, so `_kill_by_profile()` reaches exactly one window. Behind a single shared
profile both windows would be one process and ending either job would take both down.

Five things that are settled by this and were not settled by the suite:

- **The window still draws.** `drew a frame True` on all three, so nothing in the new
  flags stops the page rendering -- which was the plausible way this change could break
  the feature rather than merely shrink it.
- **11.2 MB against 61.3 MB** for the same window with the old flags, and the whole of the
  difference is `optimization_guide_model_store`.
- **A second job in the same slot costs nothing.** Identical directory, identical size.
  Under the old behaviour that second job was another 61 MB.
- **`state/jobs` is 0.0 MB** at the end, with its five leftovers -- including the two the
  watch windows above had just made -- pruned at start, and the demo job's own workspace
  gone once MARP took the result.

The baseline number is if anything flattering to the old behaviour: it is the size at 75
seconds, and the headless comparison below shows a baseline profile still climbing at that
point, reaching 102 MB by three minutes, while a flagged one is flat from 90 seconds.

### What a profile actually costs, measured headless

**Synthetic, and the oldest of the three.** Its value is the A/B that found which flag
mattered, not its absolute numbers.

Real Chrome (`C:\Program Files\Google\Chrome\Application\chrome.exe`), headless, three
fresh profiles against `about:blank`, sampled every 15s for three minutes.

| | at 15s | at 180s |
| --- | --- | --- |
| today's flags | 59 MB | **102 MB**, still climbing |
| plus background-networking and component-update off | 59 MB | 63 MB |
| plus the optimization-guide features off | 13 MB | **17 MB** |

**The two obvious flags are not enough, and that is the finding.** With background
networking *and* the component updater both disabled, `optimization_guide_model_store` was
still 47 MB -- most of what remained. Naming `OptimizationHints`,
`OptimizationGuideModelDownloading` and `OptimizationGuideOnDeviceModel` is what stops it.

What the first two did remove: Safe Browsing 19 MB, `component_crx_cache` 8 MB,
`OnDeviceHeadSuggestModel` 8 MB.

So a profile is **17 MB against 102 MB**, and it is built once per slot rather than once
per attempt. Against the issue's measured rate -- 86 to 101 attempts in twelve hours, 70 to
120 MB each -- that is roughly 14-24 GB a day becoming tens of megabytes for the life of
the worker.

## MARP_API#287: the habitat trainer on a GPU, 2026-11-10

`scripts/verify_habitat_training.py`, run by hand on the RTX 4080 SUPER in this machine,
the `marp_habitat_training` engine from this branch with the recipe's defaults (3 frozen
passes, 10 fine-tune passes, seed 1, input 518).

**The data is the feasibility session's own frames**, packed read-only into a habitat archive
(Isaac's choice, 2026-11-10), so the result can be set beside the feasibility run on the same
crop: 21,457 training, 5,163 tuning and 6,016 test frames, whole dives per split, the habitat
crop x 0.25-0.75, y 0.5-1.0. DINOv2 ViT-B/14 was built from the `model.safetensors` that
`seed-base-models.js --only dinov2-vitb14` registers, offline.

The GPU was shared for the whole run with the feasibility session's own seed-2 training and
another script, so the time below is not the trainer's own.

| | this run | feasibility run, same crop, seed 1 |
| --- | --- | --- |
| frozen passes, tuning macro F1 | 0.735 / 0.741 / 0.743 | 0.738 / 0.742 / 0.746 |
| best fine-tune pass, tuning macro F1 | 0.796 (pass 4) | 0.798 |
| **test macro F1** | **0.734** | **0.728** |
| Boulder, test F1 | 0.556 | 0.55 |

Per substrate on the test dives, at the cut-offs chosen on the tuning dives:

| | F1 | precision | recall | frames with it | cut-off |
| --- | --- | --- | --- | --- | --- |
| Rock | 0.935 | 0.898 | 0.976 | 3,999 | 0.30 |
| Mud | 0.656 | 0.548 | 0.816 | 925 | 0.15 |
| Sand | 0.841 | 0.881 | 0.805 | 2,598 | 0.90 |
| Cobble | 0.684 | 0.606 | 0.785 | 958 | 0.75 |
| Boulder | 0.556 | 0.548 | 0.565 | 407 | 0.80 |

Mud's 0.15 is the tuned cut-off; a run applies 0.35, the settled 0.2 raise, which
`db/habitat-rules.js` adds when the model is leased.

What it showed:

- **The trainer reproduces the feasibility result.** Same frames, same crop, same seed: test
  0.734 against 0.728, every frozen pass within 0.003, the best tuning pass within 0.002.
  The feasibility session's five seeds on this crop afterwards scored 0.721 to 0.738 as single
  models (0.737 averaged), so 0.734 is inside the spread rather than a lucky draw.
- **The kept model is the best tuning pass, not the last.** Tuning F1 peaked at pass 4 and
  fell to 0.786 by pass 10 as the loss went to 0.017; the published weights are pass 4's.
- **All three files were published and say what they should**: `weights` (346 MB),
  `decision_rules` with the substrates, cut-offs, crop and input size, and `final_metrics` in
  the `multilabel` family on the test split.
- Took 5,253 s (88 min) with the GPU shared.

Run once the first time and failed at the first pass on Windows: the frame loader class was
local to a function and could not be handed to a worker process. Fixed by making it a module
class, with `tests/test_habitat_training.py` now loading through a worker process (shown
failing before the fix). This run is the second.

Not covered here: a habitat run of the trained model over real video (`marp_habitat`), which
needs the model registered in MARP and a video to run on.
