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

- **Nothing has run a real watch window with these flags.** The measurement below used the
  same Chrome and the same flags, headless, against `about:blank`. It settles what gets
  fetched; it does not settle that the window still draws. That needs a GPU job.
- **The profile is now reused after a force-kill**, which no run here has exercised twice.
  `--hide-crash-restore-bubble` is there for it, untested.
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

### What a profile actually costs, measured

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
