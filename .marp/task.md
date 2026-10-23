---
task: MarineAppliedResearch/MARP_API#268
repos: [marp-inference-worker, MARP_API]
status: implementing
needs: []
---

## Goal

The ML Dashboard's training page can now queue a training job, and its forms offer four
Ultralytics arguments the worker refuses: the `box`, `cls` and `dfl` loss weights, and
`mask_ratio`. A job that sets them trains at the defaults and the record says they were
ignored.

Separately, three of `training_runs`' columns in MARP_API are never filled. Two of them
can only be filled by this worker, because it is the only thing that knows the answer:
`compute_device`, which is whatever the worker resolved rather than whatever the job
asked for, and `train_script_commit`, which is which build of this worker ran.

When this is done a training job can set the loss weights and have them honoured, and
every training attempt tells MARP which GPU it used and which build of the worker
produced it.

`loss_function` and `augmentation` are explicitly out of scope: they are not parameters
anywhere today and need designing before anything can report them.

## What was found

**The loss weights are safe to accept and the exclusions are not arbitrary.** The
allow-list exists because a handful of `model.train()` arguments change what a result
*means* -- `project` and `name` decide where weights land, `resume` reinterprets the whole
call, `device` is the worker's decision because a job is pinned to a slot. `box`, `cls`
and `dfl` are gains on the three terms the detection loss is summed from. They change how
well a run trains, never where its output goes or how it is read.

**`mask_ratio` is segmentation-only and inert elsewhere, not harmful.** Read from
Ultralytics 8.4.145 rather than from documentation: it is handed to the `Format` transform
for every task, at `data/dataset.py:328`, but read only under `return_mask`, which comes
from the dataset's `use_segments`. A detection run that sets it trains exactly as if it
had not.

**Types are the library's, not a guess.** `ultralytics.cfg` lists `box`, `cls` and `dfl`
in `CFG_FLOAT_KEYS` and `mask_ratio` in `CFG_INT_KEYS`; defaults are 7.5, 0.5, 1.5 and 4.

**There is exactly one structured channel from an attempt to MARP, and it is the settings
report.** The terminal result route records an outcome, a failure reason and artifact
hashes and nothing else (`coordinator_client.report_result`). The engine's own run summary
reaches the coordinator only inside a final log event's `terminal` field
(`runner._report_terminal_detail`), where no column can be read out of it. The `settings`
event is the one thing already shaped as typed, named, sourced values, and MARP already
writes it to `gpu_attempt_settings` and `gpu_attempt_ignored_params`
(MARP_API#232).

## Requirements

- **R1** A training job may set `box`, `cls` and `dfl`, cast to float, and they reach
  Ultralytics rather than being reported ignored.
- **R2** A training job may set `mask_ratio`, cast to int.
- **R3** Widening the allow-list does not admit `project`, `name`, `resume` or `device`.
- **R4** Every training attempt reports `compute_device` -- the device `resolve_device()`
  settled on, which a job's request is only one input to -- in its settings report, with
  `source: "worker"`.
- **R5** Every training attempt reports `train_script_commit` the same way, and always has
  an answer: a commit sha from a checkout, the package version from an installed wheel.
- **R6** A run started from a checkout with uncommitted changes says so, so a provenance
  record never claims a reproducibility it does not have.
- **R7** Every training attempt reports the `optimizer`, `lr0` and `momentum` Ultralytics
  actually resolved, with `source: "resolved"`, in a second settings report sent before
  the first epoch.
- **R8** A value that cannot be read off the live optimizer is omitted, never filled in
  from a default or from what was requested.

## The contract this changes

MARP_API should expect two new entries in a **training** attempt's settings report, in the
shape it already accepts:

```json
{
  "kind": "settings",
  "payload": {
    "engine": "marp_training",
    "settings": {
      "epochs":              {"value": 100,      "type": "int",   "source": "job"},
      "box":                 {"value": 9.0,      "type": "float", "source": "job"},
      "compute_device":      {"value": "cuda:0", "type": "str",   "source": "worker"},
      "train_script_commit": {"value": "1a2b…",  "type": "str",   "source": "worker"}
    },
    "ignored": {}
  }
}
```

A **second** settings report follows, from `on_train_start`, before epoch 1:

```json
{
  "kind": "settings",
  "payload": {
    "engine": "marp_training",
    "settings": {
      "optimizer": {"value": "AdamW", "type": "str",   "source": "resolved"},
      "lr0":       {"value": 0.002,   "type": "float", "source": "resolved"},
      "momentum":  {"value": 0.9,     "type": "float", "source": "resolved"}
    },
    "ignored": {}
  }
}
```

Three things the API side needs to know:

- **`source: "worker"` is the discriminator.** Everything else in `settings` is
  `"job"`. A reader must not treat `compute_device` as something the submitter chose; it
  is what the worker resolved, which a job's request is only one input to.
- **`train_script_commit` has two forms and they are distinguishable.** Forty hex
  characters is a git commit, optionally suffixed `+dirty`. Anything else is the worker's
  package version, which is the honest answer from a volunteer's wheel.
- **`source: "resolved"` means Ultralytics decided it, and it overrides the `"job"` entry
  of the same name.** `optimizer` and `lr0` appear in both reports when the job set them.
  The resolved one is what ran. They differ whenever the job left `optimizer` at `auto` --
  and `auto` also discards an `lr0` the job *did* send, which Ultralytics says in its own
  log line, so the two disagreeing is normal rather than a fault.
- **The second report is a second event, and MARP_API treats the first settings report as
  the record.** Taking these values therefore needs a deliberate change on the API side:
  either merge later reports when `source` is `"resolved"`, or read the resolved values
  off the event. This is the one part that does not work by itself.
- **An absent key means "could not be read", never "the default".** Nothing in the
  resolved report is reconstructed.
- **Nothing else changes.** No new event kind, no new route, no change to the result
  payload. The names were chosen to match the `training_runs` columns they are for.

## Open assumptions

- [x] **A1 - API contract - blocking** - answered by the brief and by what is already
  built: the two facts ride in the existing `settings` event rather than in a new channel.
  The result route cannot carry them and the terminal log line cannot be read as a column.
- [x] **A2 - behavioural - non-blocking** - `mask_ratio` is accepted although this engine
  describes itself as `detect` and Ultralytics will ignore it there. Accepted anyway
  because the dashboard offers it on its segmentation form and a job that sets it should
  have the value recorded rather than filed as unrecognised. It is documented as
  segmentation-only at the allow-list.
- [x] **A4 - API contract - non-blocking** - the resolved values go in a *second* settings
  report rather than being folded into the first. The first is sent before the model is
  loaded, so none of them exist yet, and moving it would lose the property that an attempt
  dying during setup still carries its record (MARP_API#232). The cost is that MARP_API's
  "first report is the record" rule has to learn about `source: "resolved"`.
- [x] **A5 - scientific/data-meaning - non-blocking** - `momentum` is reported although no
  `training_runs` column asks for it. `optimizer='auto'` overrides momentum exactly as it
  overrides lr0, so recording two of the three would leave the record quietly wrong about
  the third. The settings catalogue registers names it does not know, so an unused entry
  costs nothing and a missing one cannot be recovered later.
- [x] **A3 - environment - non-blocking** - `train_script_commit` falls back to the
  package version rather than being left empty when there is no git. A volunteer's
  installed worker is the common case, and an empty column for every volunteer run is the
  thing this requirement exists to stop.

## Not covered

No real training run. Training needs a CUDA device; what is proven here is the plumbing --
which settings reach the trainer and what the attempt reports -- exercised through `run()`
with a stubbed `ultralytics`. A run on the hardware is the human's.
