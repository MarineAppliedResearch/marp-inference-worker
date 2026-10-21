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

- **No real inference job has run with these flags.** The windows below are real, draw a
  real frame and are closed the way a job closes one, but the frames come from the script
  rather than from a model over real video. What that leaves unproven is the window under
  load, not the window.
- **One slot only.** Nothing has opened two windows at once, which is the case the
  per-slot decision exists for.
- Nothing measures a worker's disk over a day, which is the number the issue is about.

## Results

### The suite, 2026-10-21

```
375 passed, 4 skipped, 1 failed in 62.20s
FAILED tests/test_status.py::test_status_lists_the_engines_actually_registered
```

That failure is **not from this change**: it fails identically on `develop` at `c31dc4b`,
the commit this branch starts from. `/status` answers `engines` in a shape the test does
not expect (`KeyError: 'engine'`). Named and left alone.

### A real window, a real profile, 2026-10-21

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

Four things that are settled by this and were not settled by the suite:

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
