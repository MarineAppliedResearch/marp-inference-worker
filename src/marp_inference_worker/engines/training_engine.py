# training_engine.py
# Created: 2026-10-20
# Author: Isaac Travers
#
# YOLO training engine for the MARP Inference Worker.
# One training job is one ordinary Ultralytics run over one dataset: start from
# the weights the job names, train for the epochs it asks for, and hand the best
# weights back as they improve.
#
# There are no curriculum phases here and there must not be. The legacy script
# trains in three -- crops, then mixed, then frames -- and still contains the
# whole pipeline, but that was an experiment in curriculum learning that did not
# work well and has been retired (#67 R9). Anything ported from that file should
# bring the knowledge and leave the phases.
#
# What belongs here: building the run, watching it, and deciding what leaves the
# machine. What does not: fetching anything. The dataset and the starting weights
# are already on local disk by the time run() is called, because the parent
# process holds the only credential and does every fetch (#67 A8).

# json writes the two end-of-run reports to disk before they are published.
import json

# math screens NaN and infinity out of those reports. Ultralytics produces
# both, and json.dumps writes them as bare NaN/Infinity, which is not valid
# JSON -- so one unmeasured class would make the whole artifact unparseable.
import math

# time rate-limits how often weights are published.
import time

# Path types the dataset, the weights and everything published.
from pathlib import Path

# Any and Mapping type the free-form spec and params.
from typing import Any, Mapping

# The contract every engine is invoked through.
from marp_inference_worker.engines.base_engine import (
    BaseEngine,
    JobContext,
    JobUnrunnable,
)

# Device selection is the worker's decision, not the engine's.
from marp_inference_worker.system.device import (
    cuda_device_count,
    describe_cuda_devices,
    resolve_device,
)

# Which build of this worker produced the run, for the attempt's record.
from marp_inference_worker.system.provenance import worker_revision


# How long to wait between publishing improved weights, in seconds.
#
# Early epochs improve on almost every pass and the weights are around 130 MB,
# so an unthrottled "publish every best" sends gigabytes of files that are
# superseded minutes later. Ten minutes is far shorter than the window in which
# losing a run would actually hurt, and far longer than the burst at the start
# of training. The final best is published regardless of this (#67 R11).
_PUBLISH_INTERVAL_S = 600.0


# The Ultralytics arguments a job may set, and what they mean to us.
#
# An allow-list rather than passing `params` through: `model.train()` takes well
# over a hundred arguments, several of which quietly change what the result
# means -- `project` and `name` decide where weights land, `resume` reinterprets
# the whole call, `device` is the worker's decision and not the submitter's.
# Anything not listed is reported as ignored rather than silently dropped, which
# is the same contract inference already keeps (MARP_API#232).
_ALLOWED_TRAINING_PARAMS = {
    "epochs": int,
    "imgsz": int,
    "batch": int,
    "patience": int,
    "workers": int,
    "lr0": float,
    "lrf": float,
    "momentum": float,
    "weight_decay": float,
    "warmup_epochs": float,
    "optimizer": str,
    "seed": int,
    "cos_lr": bool,
    "dropout": float,
    "freeze": int,
    "single_cls": bool,
    "rect": bool,

    # The loss weights. These are the gains on the three terms YOLO's detection
    # loss is summed from, and they are what you reach for when a detector finds
    # the wrong things rather than when it trains badly: raising `box` buys
    # tighter boxes, raising `cls` buys confidence on small or rare classes at
    # the cost of false positives. That trade is the one MARP deliberately wants
    # pushed toward recall, because a biologist rejecting a false positive in a
    # review mosaic is cheap and a missed animal costs re-reviewing video.
    #
    # Safe to accept where `project`, `name`, `resume` and `device` are not:
    # they change how well the run trains, never where its weights land or what
    # the result is taken to mean.
    "box": float,
    "cls": float,
    "dfl": float,

    # Segmentation only, and inert rather than harmful on a detection run.
    # Ultralytics hands `mask_ratio` to its `Format` transform for every task but
    # reads it only under `return_mask`, which is set from the dataset's
    # `use_segments`. So a detect job that sets it trains exactly as if it had
    # not -- it is accepted here because the dashboard offers it on the
    # segmentation form and a job that sets it should have it recorded, not
    # because this engine can currently act on it.
    #
    # An int, not a float: it is a downsample divisor, and Ultralytics lists it
    # among its integer keys.
    "mask_ratio": int,
}


# Defaults for a MARP training run.
# Stated here rather than left to Ultralytics so that what ran is a property of
# this worker's version rather than of whichever library version was installed.
_TRAINING_DEFAULTS: dict[str, Any] = {
    "epochs": 100,
    "patience": 50,
}


# The image size a run falls back to, by what kind of dataset it was given.
#
# **These are different numbers because the tasks are different problems**, and
# one default for both was wrong for whichever it was not chosen for.
#
# 640 is the detection size: ROV frames are large and the animals in them are
# small, so resolution is the whole difficulty.
#
# 224 is what the pretrained `-cls` weights were themselves trained at, so a
# classification run starts from the size its backbone expects. The crops are
# 320x320 mosaic tiles, and 320 is a selection away in the dashboard for anyone
# who wants to compare -- but the worker's own fallback had been 640, which
# upscaled every tile fourfold for no new information and nobody had chosen it.
#
# A job that sets `imgsz` always wins over both; these only apply when it says
# nothing.
_DETECTION_IMGSZ = 640
_CLASSIFY_IMGSZ = 224


# _default_imgsz()
# The image size to use when the job did not name one.
# Inputs: the job params.
# Output: the fallback size for this kind of dataset.
#
# Decided from what is on disk, the same way the data argument is: a data.yaml
# means detection or segmentation, a directory means classification. The job
# cannot declare a task -- the checkpoint owns that -- and this runs before
# anything loads the checkpoint.
def _default_imgsz(params: Mapping[str, Any]) -> int:

    data = _training_data_argument(params)
    if data is not None and data.is_dir():
        return _CLASSIFY_IMGSZ

    # A dataset that is unusable is refused by preflight anyway; falling back
    # to the detection size here keeps the old behaviour for every job that
    # has a data.yaml.
    return _DETECTION_IMGSZ


# The four type names MARP's settings contract accepts.
#
# They are not Python's, and the difference is not cosmetic. MARP validates
# every entry against `real, int, bool, text`, and ONE entry typed "float" makes
# it refuse the WHOLE report with a 400 -- so the attempt records no settings at
# all. Run 205 resolved optimizer=AdamW, lr0=0.002 and momentum=0.9 exactly
# right and stored none of them, because two words did not match (#268).
#
# The worker is the client and MARP's enum is the published contract, so the
# translation belongs on this side. The tracking engine has always done this
# through its own `_SETTING_TYPES`; training was written later and emitted
# `type(value).__name__` instead, which is where the defect came from.
#
# Keyed by Python's type name rather than by isinstance, deliberately: bool is a
# subclass of int, so an isinstance chain reports True as an integer unless bool
# is tested first. A name lookup cannot make that mistake.
_CONTRACT_TYPE_NAMES = {
    "float": "real",
    "int": "int",
    "bool": "bool",
    "str": "text",
}


# _setting_entry()
# Builds one settings-report entry in the shape the coordinator validates.
# Inputs: the value that will actually run, and where it came from.
# Output: `{value, type, source}` with `type` in MARP's vocabulary.
# Use this for EVERY entry, so the vocabulary lives in exactly one place and a
# new report cannot reintroduce Python's names.
def _setting_entry(value: Any, source: str) -> dict[str, Any]:

    contract = _CONTRACT_TYPE_NAMES.get(type(value).__name__)

    # Anything outside the four travels as text rather than being dropped or
    # sent untranslated. A type the contract does not know would otherwise cost
    # the entire report, and the entire report is what run 205 lost.
    if contract is None:
        return {"value": str(value), "type": "text", "source": source}

    return {"value": value, "type": contract, "source": source}


# _run_provenance()
# The facts about a run that only the worker can state, as settings entries.
# Inputs: the device this run resolved to.
# Output: a mapping in report_settings' `{name: {value, type, source}}` shape.
#
# These ride in the settings report rather than in a channel of their own
# because that is the one structured thing an attempt already tells MARP about
# how it ran: the terminal result route records an outcome, a failure reason and
# artifact hashes and nothing else, and the engine's own summary reaches the
# coordinator only inside a log line, where nothing can read a column out of it.
#
# `source` is "worker", and it is the load-bearing part. Neither value is
# something the submitter set. A job may *request* a device, but what it gets is
# whatever resolve_device() makes of that against this machine -- "auto" becomes
# this slot's card or the CPU, and a card this worker does not have is a refusal
# -- so the only device that is a fact is the resolved one. Saying "worker" is
# what lets a reader tell it apart from a request.
def _run_provenance(device: str) -> dict[str, dict[str, Any]]:

    provenance = {
        # Named for MARP's `training_runs.compute_device`, and carrying what
        # that column actually asks for.
        #
        # **The column wants hardware, not a slot.** Its comment in the schema
        # reads 'Hardware used for training (e.g. "RTX 6000 Ada", "A100 GPU")'.
        # This used to send `cuda:0`, which answers a different question -- it
        # says which device index the job was pinned to, not what the model was
        # trained on. Two runs on two different cards both report "cuda:0", so
        # the column could not have done its job even once it was being read.
        "compute_device": _setting_entry(_device_hardware(device), "worker"),
        # The slot is still worth keeping, it is just not the hardware. Which
        # index ran matters when a multi-GPU machine misbehaves on one card.
        "compute_device_index": _setting_entry(device, "worker"),
        # Named for MARP's `training_runs.train_script_commit`. A sha when the
        # worker runs from a checkout, the package version when it runs from an
        # installed wheel; see worker_revision() for why both are answers.
        "train_script_commit": _setting_entry(worker_revision(), "worker"),
    }

    return provenance


# _device_hardware()
# Turns a resolved device string into the name of the thing that will run.
# Inputs: the device string resolve_device() settled on ("cuda:0", "cpu").
# Output: the card's model name, or the device string when there is no name.
#
# Falls back rather than inventing: a CPU run has no card to name, and a driver
# that will not answer get_device_properties leaves describe_cuda_devices()
# reporting an error for that index. In both cases the device string is the most
# specific true thing available, which is what the old behaviour sent anyway.
def _device_hardware(device: str) -> str:

    text = str(device).strip().lower()
    if not text.startswith("cuda"):
        return str(device)

    # "cuda" with no index means this slot's card, which resolve_device() has
    # already turned into "cuda:N" by the time run() calls us. Guard anyway.
    try:
        index = int(text.split(":", 1)[1])
    except (IndexError, ValueError):
        return str(device)

    try:
        for described in describe_cuda_devices():
            if described.get("index") == index and described.get("name"):
                return str(described["name"])
    except Exception:
        # Naming the card is bookkeeping; it must never cost the run.
        pass

    return str(device)


# _resolved_training_settings()
# What Ultralytics decided for itself, read off the built optimizer.
# Inputs: the Ultralytics trainer, once _setup_train() has finished.
# Output: a mapping in report_settings' `{name: {value, type, source}}` shape,
#   carrying only what could actually be read.
#
# **This exists because the requested values are not updated in place.**
# `build_optimizer` resolves `optimizer='auto'` into a name, a learning rate and
# a momentum, and keeps all three in local variables: `trainer.args.optimizer`
# still reads "auto" and `trainer.args.lr0` still reads whatever was configured.
# Reading `args` back would therefore record what was asked for while claiming it
# was what ran, which is the one thing a scientific record must not do.
#
# Worse, `auto` ignores a learning rate the submitter *did* send -- Ultralytics
# says so in its own log line -- so the gap is not only about the Auto default.
#
# Nothing here is reconstructed. Every value is read off the live optimizer, and
# anything that cannot be read is left out rather than filled in from a default
# (#268): an absent column is a known unknown, a guessed one is a wrong fact.
def _resolved_training_settings(trainer: Any) -> dict[str, dict[str, Any]]:

    resolved: dict[str, dict[str, Any]] = {}

    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is None:
        return resolved

    # The optimizer's class is the only honest statement of which one was
    # chosen; `args.optimizer` is still the request.
    resolved["optimizer"] = _setting_entry(type(optimizer).__name__, "resolved")

    groups = getattr(optimizer, "param_groups", None) or []
    if not groups:
        return resolved

    # `initial_lr`, not `lr`. The scheduler stamps `initial_lr` on every group
    # when it is constructed, and then warmup starts moving `lr` per batch, so
    # by the time anything can read it `lr` is no longer the run's lr0.
    first = groups[0]
    if "initial_lr" in first:
        resolved["lr0"] = _setting_entry(float(first["initial_lr"]), "resolved")

    # Momentum lives in two places depending on which optimizer was chosen:
    # SGD-family optimizers have `momentum`, Adam-family ones keep it as the
    # first of `betas`. Read both, report neither if it is neither.
    momentum = first.get("momentum")
    if momentum is None:
        betas = first.get("betas") or ()
        momentum = betas[0] if betas else None
    if momentum is not None:
        resolved["momentum"] = _setting_entry(float(momentum), "resolved")

    return resolved


# How often a live sample leaves the child, in seconds.
#
# Throttled by TIME rather than by batch count, because batches are not a unit
# of time: a 4080 does several a second on small images and one every few
# seconds on large ones, so "every N batches" gives a chart whose density
# depends on the image size. A phase change always samples regardless.
#
# The rate is what a person watching can use, not what the wire can carry --
# events are queued and batched upstream, so this costs little either way.
_SAMPLE_INTERVAL_S = 2.5


# The phase names live samples are filed under.
#
# **Distinct from the per-epoch rows on purpose.** Those already go through
# report_metrics as `(epoch_number, "training", ...)`. If a batch sample used
# the same phase, the two series would be told apart only by the magnitude of
# `step` -- which works on every long run and collapses the first time a
# one-epoch run makes both equal 1. Routing on a word cannot do that.
#
# PENDING: the exact strings belong to MARP's ingest and are not confirmed yet
# (#268). They are here, in one place, so settling them is a two-line change.
_BATCH_PHASE = "training_batch"
_VAL_BATCH_PHASE = "validating_batch"


# _epoch_metrics()
# Everything one finished epoch can honestly say about itself.
# Inputs: the Ultralytics trainer, at `on_fit_epoch_end`.
# Output: a free-form metric mapping, carrying only what was readable.
#
# Three sources, because Ultralytics keeps them in three places and only ever
# brings them together to write its own CSV (`save_metrics`, trainer.py:635):
#
#   trainer.metrics     the validator's output -- precision, recall, mAP50,
#                       mAP50-95, fitness. Already reported before this existed.
#   trainer.tloss       the epoch's running mean loss, reset at every epoch
#                       start, labelled through the trainer's own
#                       `label_loss_items` so the names are the task's own
#                       (`train/box_loss`, and `train/seg_loss` on segmentation).
#   trainer.epoch_time  set immediately before this callback fires.
#
# Nothing is derived. The losses are labelled by Ultralytics' own method rather
# than by a list of names kept here, which would be a second copy that goes
# stale the first time a task has a different loss. A source that is missing --
# `tloss` is None through a NaN-recovery epoch -- contributes nothing rather
# than a zero (#268).
def _epoch_metrics(trainer: Any) -> dict[str, Any]:

    metrics: dict[str, Any] = {
        str(name): value
        for name, value in (getattr(trainer, "metrics", {}) or {}).items()
    }

    # The per-epoch losses, which never reach trainer.metrics.
    tloss = getattr(trainer, "tloss", None)
    labeller = getattr(trainer, "label_loss_items", None)
    if tloss is not None and callable(labeller):
        try:
            metrics.update({str(name): value for name, value in labeller(tloss).items()})
        except Exception:
            # A trainer that labels its losses differently is not a failed
            # epoch. Report what was readable and leave the rest out.
            pass

    # The one key the worker names itself, because Ultralytics keeps the epoch
    # duration as an attribute and never puts it in a mapping. Named for MARP's
    # `epochs.duration_seconds` so the API side is a lookup, not a mapping.
    #
    # **What this duration covers, because both answers surprise somebody:**
    #
    # It INCLUDES that epoch's validation. Ultralytics takes its timestamp after
    # `validate()` has returned (trainer.py:645), so an epoch's duration is
    # training plus the validation that followed it. Anyone comparing these
    # against the training loop alone will find them consistently longer, and
    # that is correct rather than drift.
    #
    # And the FIRST epoch's clock starts at `_do_train` entry, not at the job's.
    # `epoch_time_start` is initialised there (trainer.py:457), which is after
    # `_setup_train` has built the dataloaders, loaded the model and built the
    # optimizer. Those are not the epoch and should not be in it -- but on a
    # large dataset the gap is minutes, so epoch 1's duration is not the time
    # from the job starting, and reading it that way understates a run badly.
    epoch_time = getattr(trainer, "epoch_time", None)
    if epoch_time is not None:
        try:
            metrics["duration_seconds"] = round(float(epoch_time), 3)
        except (TypeError, ValueError):
            pass

    # Where this epoch sat inside its run, as seconds since training began.
    #
    # Offsets rather than wall-clock instants, which is Isaac's decision (#268):
    # "ordering and duration only -- valid within a run, never compared across
    # machines". A volunteer's clock can be wrong by any amount, so an absolute
    # timestamp from this machine would claim an agreement with MARP's other
    # timestamps that nothing checks.
    #
    # What these ARE: durations measured by the trainer. A machine an hour out
    # of sync still measures 42 seconds as 42 seconds, so an offset does not
    # depend on this computer's clock being right -- only on it ticking evenly.
    #
    # What these are NOT: a position on any MARP-wide timeline. If MARP ever
    # places them on one by adding its own stamp for the run, that is MARP's
    # doing and the accuracy of it is MARP's to state. Nothing here asserts it.
    #
    # The end is read off `epoch_time_start`, which by the time this callback
    # runs holds the moment the epoch ENDED -- Ultralytics reuses the one
    # attribute as the next epoch's start (trainer.py:646-647). The start is the
    # end minus the duration, and that subtraction is exact rather than an
    # estimate: `epoch_time` was itself computed as this timestamp minus the
    # previous one, so subtracting it back reproduces the overwritten instant.
    #
    # Deliberately not a clock reading of our own taken in `on_train_epoch_start`
    # -- that would be this process's clock a few milliseconds from the
    # trainer's, and `end - start` would then disagree with `duration_seconds`
    # for no benefit. The identity is worth more than a directly sampled start.
    #
    # **The origin is the start of TRAINING, not of the job.** `train_time_start`
    # is set at `_do_train` entry (trainer.py:458), after `_setup_train` has
    # built the dataloaders, loaded the model and built the optimizer. So epoch
    # 1 starts at roughly 0.0 and the setup that preceded it is not inside these
    # numbers at all. Setup cost is still recoverable -- it is the gap between
    # the attempt's first event and the settings report sent from
    # `on_train_start` -- but it is not here, and a reader expecting epoch 1 to
    # begin at the setup time will be wrong.
    origin = getattr(trainer, "train_time_start", None)
    epoch_end = getattr(trainer, "epoch_time_start", None)
    if origin is not None and epoch_end is not None and "duration_seconds" in metrics:
        try:
            # Rounded before the subtraction so the identity
            # `end_offset_s - start_offset_s == duration_seconds` survives it.
            end_offset = round(float(epoch_end) - float(origin), 3)
            metrics["end_offset_s"] = end_offset
            metrics["start_offset_s"] = round(end_offset - metrics["duration_seconds"], 3)
        except (TypeError, ValueError):
            # Half an interval is not a measurement: if either end cannot be
            # read, neither is reported.
            metrics.pop("end_offset_s", None)

    return metrics


# The metric families this worker can report, by the attribute Ultralytics
# keeps each one on.
#
# `box` and `seg` are the strings MARP's `metric_family` enum accepts, settled
# with the API side -- and they are NOT validated there, so a wrong word would
# not throw. It would store a row nobody renders and a table that silently
# never appears, which is why the spelling is pinned here and tested.
#
# `seg` rather than `mask` deliberately: it is the name of the attribute
# Ultralytics itself uses (`SegmentMetrics.seg`), where `(M)` is only the
# suffix on its metric keys.
#
# A tuple of pairs rather than a hard-coded `.box`, so a segmentation run
# reports two families from the same code and classification can be added as
# one entry rather than a rewrite (#268).
_METRIC_FAMILIES = (("box", "box"), ("seg", "seg"))


# What a classification run reports instead, since it has no box or seg family.
#
# `ClassifyMetrics` carries no per-class anything -- no precision, recall or AP
# per class, only these two aggregates -- so a classify block has no `per_class`
# and its confusion matrix is the only per-class information the run produces.
#
# Read off a real one-epoch run rather than from documentation: the keys are
# `metrics/accuracy_top1` and `metrics/accuracy_top5`, with no `(B)`/`(M)`
# suffix, because that suffix distinguishes box from mask and classification is
# neither.
_CLASSIFY_ACCURACIES = (("accuracy_top1", "top1"), ("accuracy_top5", "top5"))


# How the worker labels the confusion matrix's orientation.
#
# Said in the artifact rather than left to documentation, because Ultralytics'
# `matrix[p][t]` is predicted-by-true -- the transpose of how a person reads
# one -- and a reader that assumes the other way round produces a confident lie
# about which species is confused with which.
_MATRIX_ORIENTATION = "rows_predicted_cols_true"


# The curve labels Ultralytics uses, mapped to the keys MARP stores.
# Matched on the labels Ultralytics itself attaches in `curves_results` rather
# than by position, so a reordering upstream drops nothing silently.
_CURVE_KEYS = {"precision": "precision", "recall": "recall", "f1": "f1"}


# _json_number()
# Turns one metric into something JSON can actually carry.
# Inputs: any value Ultralytics produced.
# Output: a float, or None when there is no number to state.
#
# NaN is the case that matters. Ultralytics yields it for a class it could not
# score, and `json.dumps` writes it as a bare `NaN`, which is not valid JSON --
# so one unmeasured class would cost the entire artifact. None becomes `null`,
# which is what this work says everywhere else: absence, not zero.
def _json_number(value: Any) -> float | None:

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return None if math.isnan(number) or math.isinf(number) else number


# _confusion_matrix_report()
# The confusion matrix as numbers, labelled with its own axes.
# Inputs: the validator, once the final validation has run.
# Output: the matrix block, or None when there is nothing to report.
#
# **The axes carry their own class list, and that is the whole point.**
# `ConfusionMatrix.names` is the model's FULL vocabulary, while the per-class
# metrics are indexed by `ap_class_index`, which holds only the classes that
# appeared in validation. One shared list for both silently shifts every axis
# label after any species that was never evaluated -- and the rare species are
# exactly the ones a reader is looking for.
def _confusion_matrix_report(validator: Any) -> dict[str, Any] | None:

    holder = getattr(validator, "confusion_matrix", None)
    raw = getattr(holder, "matrix", None)
    if raw is None:
        return None

    try:
        rows = [[_json_number(cell) for cell in row] for row in raw.tolist()]
    except Exception:
        return None
    if not rows:
        return None

    # The vocabulary, in index order. Keyed by class id, so sorted by key
    # rather than trusting the dict's insertion order.
    names = getattr(holder, "names", None) or {}
    classes = [str(names[key]) for key in sorted(names)]

    report: dict[str, Any] = {
        "orientation": _MATRIX_ORIENTATION,
        "classes": classes,
        "matrix": rows,
    }

    # Detection's matrix is (n+1) square, the extra row and column being
    # background. Classification's is n square and has none -- so the key is
    # absent rather than null, the same absence rule as everywhere else.
    # Derived from the matrix's own size rather than from the task, so it
    # cannot disagree with the array it describes.
    if len(rows) == len(classes) + 1:
        report["background_index"] = len(classes)

    return report


# _overall_metrics()
# One family's aggregate numbers.
# Inputs: an Ultralytics Metric object (`metrics.box`, `metrics.seg`).
# Output: the aggregate mapping.
#
# `mean_results()` returns exactly four values -- mean precision, mean recall,
# mAP50, mAP50-95 -- and no F1. There is deliberately no `f1` here: Ultralytics
# exposes no aggregate one, and `2PR/(P+R)` of the means is a different
# quantity it never computes. An absent field beats a plausible invention.
#
# Both means are over the EVALUATED classes, not the vocabulary, so "overall"
# does not describe species validation never saw.
def _overall_metrics(family: Any) -> dict[str, Any]:

    overall: dict[str, Any] = {}

    try:
        precision, recall, map50, map5095 = family.mean_results()
    except Exception:
        return overall

    overall["precision"] = _json_number(precision)
    overall["recall"] = _json_number(recall)
    overall["map50"] = _json_number(map50)
    overall["map5095"] = _json_number(map5095)

    # Ultralytics weights fitness [0, 0, 0, 1], so today this equals map5095
    # exactly. Sent anyway: it is the number checkpoints are ranked by, and the
    # weights are Ultralytics' to change.
    try:
        overall["fitness"] = _json_number(family.fitness())
    except Exception:
        pass

    return overall


# _per_class_metrics()
# One row per class that validation actually scored.
# Inputs: the family's Metric object and the validator's metrics holder.
# Output: a list of rows, each naming its own class.
#
# **Two different indexes, read in one loop, and Ultralytics' own code mixes
# them.** `p`, `r`, `f1`, `ap50` and `ap` are indexed by position within the
# evaluated set; `names`, `nt_per_class` and `nt_per_image` are indexed by
# CLASS ID. Zipping them blind attributes one species' instance count to
# another whenever some class went unevaluated. Each row therefore names
# itself, and nothing downstream has to know which index it came from.
#
# A class with no validation instances gets NO ROW -- not a row of zeros.
# "Never evaluated" and "scored zero" are different facts, and on a species
# list whose rare classes are the point, conflating them is the expensive one.
def _per_class_metrics(family: Any, metrics: Any) -> list[dict[str, Any]]:

    rows: list[dict[str, Any]] = []

    # `is None` rather than `or []`: ap_class_index is a numpy array, and
    # `array or []` raises ValueError on anything with more than one element.
    # That is not hypothetical -- it is what Ultralytics actually hands us, and
    # the broad except around this call turned the crash into a report that
    # silently carried no metrics at all.
    raw_ids = getattr(family, "ap_class_index", None)
    class_ids = [] if raw_ids is None else list(raw_ids)
    if not class_ids:
        return rows

    names = getattr(metrics, "names", None) or {}
    support = getattr(metrics, "nt_per_class", None)
    images = getattr(metrics, "nt_per_image", None)

    for position, raw_id in enumerate(class_ids):
        class_id = int(raw_id)
        row: dict[str, Any] = {"class": str(names.get(class_id, class_id))}

        # Indexed by POSITION within the evaluated set.
        for key, attribute in (
            ("precision", "p"), ("recall", "r"), ("f1", "f1"),
            ("ap50", "ap50"), ("ap5095", "ap"),
        ):
            series = getattr(family, attribute, None)
            try:
                row[key] = _json_number(series[position])
            except Exception:
                pass

        # Indexed by CLASS ID. Counted things, so reported as integers.
        for key, series in (("support", support), ("images", images)):
            try:
                count = _json_number(series[class_id])
                if count is not None:
                    row[key] = int(count)
            except Exception:
                pass

        rows.append(row)

    return rows


# _metric_families()
# Every family this run reported, as blocks.
# Inputs: the validator's metrics holder.
# Output: a list of family blocks; empty when none could be read.
# Use this rather than reaching for `.box`: a segmentation run reports two
# families from the same code, and a failure in one must not cost the other.
def _metric_families(metrics: Any) -> list[dict[str, Any]]:

    blocks: list[dict[str, Any]] = []

    for attribute, family_name in _METRIC_FAMILIES:
        family = getattr(metrics, attribute, None)
        if family is None:
            continue

        try:
            block: dict[str, Any] = {
                "family": family_name,
                "overall": _overall_metrics(family),
            }
            # Omitted when empty rather than sent as []: no rows means nothing
            # was scored per class, which is absence, not an empty measurement.
            rows = _per_class_metrics(family, metrics)
            if rows:
                block["per_class"] = rows
        except Exception:
            # One family that cannot be read must not cost the others.
            continue

        if block["overall"] or block.get("per_class"):
            blocks.append(block)

    # Classification has neither family, so without this a classify run would
    # send a confusion matrix and no metrics at all.
    if not blocks:
        classify = _classify_metrics(metrics)
        if classify:
            blocks.append(classify)

    return blocks


# _classify_metrics()
# A classification run's one aggregate block.
# Inputs: the validator's metrics holder.
# Output: the block, or None when this is not a classification run.
#
# No `per_class`: ClassifyMetrics has no per-class quantities at all. The
# confusion matrix is the only per-class information a classify run produces,
# and it is reported separately.
def _classify_metrics(metrics: Any) -> dict[str, Any] | None:

    overall: dict[str, Any] = {}

    for key, attribute in _CLASSIFY_ACCURACIES:
        value = _json_number(getattr(metrics, attribute, None))
        if value is not None:
            overall[key] = value

    if not overall:
        return None

    # Ultralytics ranks classification checkpoints on fitness too.
    fitness = _json_number(getattr(metrics, "fitness", None))
    if fitness is not None:
        overall["fitness"] = fitness

    return {"family": "classify", "overall": overall}


# _evaluated_split()
# Which split the final validation actually ran against.
# Inputs: the trainer.
# Output: "val", "test", or None when the dataset named neither.
# Stated rather than assumed: Ultralytics builds its validation loader from
# `data["val"] or data["test"]`, so a dataset supplying only `test` is
# validated on test, and recording "val" would be a small lie in a record.
def _evaluated_split(trainer: Any) -> str | None:

    data = getattr(trainer, "data", None) or {}
    try:
        if data.get("val"):
            return "val"
        if data.get("test"):
            return "test"
    except Exception:
        pass
    return None


# _final_metrics_report()
# The end-of-run report: what the trained model scores, as numbers.
# Inputs: the trainer, at `on_train_end`.
# Output: the report, or None when there is nothing worth sending.
#
# Assembled section by section, each able to fail alone. The lesson of run 205
# is that an all-or-nothing payload turns one bad value into a total loss, so a
# matrix that cannot be read still leaves the per-class rows, and the reverse.
def _final_metrics_report(trainer: Any) -> dict[str, Any] | None:

    validator = getattr(trainer, "validator", None)
    if validator is None:
        return None

    report: dict[str, Any] = {"schema": 1}

    task = getattr(getattr(trainer, "args", None), "task", None)
    if task:
        report["task"] = str(task)

    split = _evaluated_split(trainer)
    if split:
        report["split"] = split

    try:
        matrix = _confusion_matrix_report(validator)
    except Exception:
        matrix = None
    if matrix:
        report["confusion_matrix"] = matrix

    metrics = getattr(validator, "metrics", None)
    families = _metric_families(metrics) if metrics is not None else []
    if families:
        report["metrics"] = families

    # Nothing measured means nothing to say. An empty report would be a file
    # claiming a run was scored when it was not.
    if "confusion_matrix" not in report and "metrics" not in report:
        return None

    return report


# _curves_report()
# The P, R and F1 curves over confidence, as numbers.
# Inputs: the trainer and the split it evaluated.
# Output: the curves report, or None.
#
# **Three curves, not four.** `curves_results` also carries the PR curve, whose
# x axis is RECALL while the other three are confidence -- Ultralytics reuses
# one `px` array for both meanings and distinguishes them only by the label it
# attaches. Storing the PR curve against a confidence axis would record a
# recall value as a confidence. It is kept as `PR_curve.png`, and precision and
# recall at every threshold reconstruct it anyway.
#
# `confidence` is stored once: all three share the one x axis.
def _curves_report(trainer: Any, split: str | None) -> dict[str, Any] | None:

    metrics = getattr(getattr(trainer, "validator", None), "metrics", None)
    if metrics is None:
        return None

    for attribute, family_name in _METRIC_FAMILIES:
        family = getattr(metrics, attribute, None)
        if family is None:
            continue

        try:
            raw_curves = getattr(family, "curves_results", None)
            results = [] if raw_curves is None else list(raw_curves)
        except Exception:
            continue

        confidence: list[Any] | None = None
        curves: dict[str, Any] = {}

        for entry in results:
            try:
                x_values, y_values, x_label, y_label = entry
            except Exception:
                continue

            # Only the confidence-indexed curves. The PR curve's x is recall.
            if str(x_label).strip().lower() != "confidence":
                continue

            key = _CURVE_KEYS.get(str(y_label).strip().lower())
            if key is None:
                continue

            try:
                # 5 decimals: these are 0-1 metrics interpolated over 1000
                # points, so the sixth place is noise and the file halves.
                curves[key] = [
                    [_rounded(value) for value in series] for series in y_values
                ]
                if confidence is None:
                    confidence = [_rounded(value) for value in x_values]
            except Exception:
                curves.pop(key, None)

        if not curves or confidence is None:
            continue

        names = getattr(metrics, "names", None) or {}
        # Same numpy trap as in _per_class_metrics: `array or []` raises.
        raw_ids = getattr(family, "ap_class_index", None)
        classes = [
            str(names.get(int(c), int(c)))
            for c in ([] if raw_ids is None else list(raw_ids))
        ]

        report: dict[str, Any] = {
            "schema": 1,
            "family": family_name,
            "classes": classes,
            "confidence": confidence,
            "curves": curves,
        }
        if split:
            report["split"] = split
        return report

    return None


# _rounded()
# One curve point, at the precision worth storing.
# Inputs: a value from a curve array.
# Output: the value to 5 decimals, or None when it is not a number.
def _rounded(value: Any) -> float | None:

    number = _json_number(value)
    return None if number is None else round(number, 5)


# _training_data_argument()
# What this job should pass to Ultralytics as `data`, if anything.
# Inputs: the job params.
# Output: the data.yaml for a detection or segmentation run, the dataset root
#   for a classification one, or None when neither is usable.
#
# Two shapes, because Ultralytics takes two. Detection and segmentation are
# given a `data.yaml`; **classification is given a DIRECTORY** and reads its
# class vocabulary from the subfolder names -- there is no names list anywhere
# in a classify dataset.
#
# Decided from what is on disk rather than from a task the job declares,
# because the job does not declare one: the checkpoint decides the task, and
# preflight runs before anything loads the checkpoint.
def _training_data_argument(params: Mapping[str, Any]) -> Path | None:

    data_yaml = params.get("data_yaml")
    if data_yaml and Path(str(data_yaml)).is_file():
        return Path(str(data_yaml))

    root = params.get("dataset_path")
    if root and Path(str(root)).is_dir() and (Path(str(root)) / "train").is_dir():
        return Path(str(root))

    return None


# _why_the_dataset_is_unusable()
# The reason a job cannot train, in words a person can act on.
# Inputs: the job params.
# Output: the refusal message.
#
# **The `train/` case is the one that matters, and it is a refusal rather than
# a fallback on purpose.** Given a directory with images but no `train/`,
# Ultralytics does not fail -- it SILENTLY SPLITS 80/20 at random
# (`split_classify_dataset`, data/utils.py:718-726) and logs it at INFO. MARP
# splits deliberately and temporally, so a random split puts frames from the
# same dive seconds apart on both sides and produces accuracy that looks
# excellent and means nothing, with nothing downstream saying so. Refusing is
# the only honest answer.
def _why_the_dataset_is_unusable(params: Mapping[str, Any]) -> str:

    data_yaml = params.get("data_yaml")
    root = params.get("dataset_path")

    if not data_yaml and not root:
        return "the job carried no dataset; nothing was prepared to train on"

    if root and Path(str(root)).is_dir() and not (Path(str(root)) / "train").is_dir():
        return (
            f"the dataset at {root} has no data.yaml and no train/ directory. A "
            "classification dataset must carry its own train/ split -- Ultralytics "
            "would otherwise invent a random 80/20 one, which is not the split "
            "this dataset was built with"
        )

    if data_yaml:
        return f"the dataset's data.yaml is missing from {data_yaml}"

    return f"the dataset at {root} is not on disk"


# _dataset_root()
# The directory a dataset lives in, whichever shape it is.
# Inputs: the data argument this run will pass to Ultralytics.
# Output: the dataset's own directory.
def _dataset_root(data_path: Path) -> Path:

    return data_path.parent if data_path.is_file() else data_path


# TrainingEngine
# Runs one YOLO training job and publishes the weights it produces.
# Constructed once by the registry; every run happens in its own child process,
# so an instance holds no state between jobs.
class TrainingEngine(BaseEngine):

    # engine_name
    # The name a job spec selects this engine by.
    @property
    def engine_name(self) -> str:

        return "marp_training"

    # describe()
    # Says what this engine is and what it needs, for /status.
    # Inputs: none.
    # Output: JSON-safe mapping.
    def describe(self) -> dict[str, Any]:

        return {
            # `engine`, not `name`: every other engine in this worker reports
            # itself under that key, and /status was listing this one under a
            # key nothing else used -- so a consumer iterating engines by
            # entry["engine"] skipped it or died on it. The suite's own
            # test_status_lists_the_engines_actually_registered was red because
            # of this before the classification work touched the file.
            "engine": self.engine_name,
            "kind": "training",
            "requires_gpu": True,
            # What this engine can run, not what it will run. The TASK IS
            # DECIDED BY THE CHECKPOINT -- Ultralytics reads it off the .pt and
            # ignores any task argument for one -- so the engine does not know
            # until a job names its weights. This used to claim "detect", which
            # was simply false for a segmentation run (#268).
            "tasks": ["detect", "segment", "classify"],
        }

    # preflight()
    # Decides whether this machine can run this training job at all.
    # Inputs: the job spec mapping.
    # Output: none on success; raises JobUnrunnable with a reason otherwise.
    # Use this so a job that cannot run fails immediately and says why, rather
    # than starting and dying somewhere inside Ultralytics (R14).
    def preflight(self, spec: Mapping[str, Any]) -> None:

        params = dict(spec.get("params") or {})

        # Training on a CPU is not a slower version of the job, it is a
        # different job: a run that would take an hour takes days, and the
        # volunteer's machine is unusable throughout. Refuse rather than start.
        if cuda_device_count() <= 0:
            raise JobUnrunnable("training needs a CUDA GPU and this machine reports none")

        # The dataset is fetched by the parent before the child starts, so by
        # the time preflight runs it is either on disk or the job is already
        # broken. Saying which is the point of this check.
        if _training_data_argument(params) is None:
            raise JobUnrunnable(_why_the_dataset_is_unusable(params))

        if not params.get("model_path"):
            raise JobUnrunnable("the job carried no starting weights")

    # run()
    # Runs one training job to completion, cancellation or failure.
    # Inputs: the job context and the job spec.
    # Output: a JSON-safe summary. The weights go out through publish_artifact.
    def run(self, ctx: JobContext, spec: Mapping[str, Any]) -> dict[str, Any]:

        started_at = time.monotonic()
        params = dict(spec.get("params") or {})

        data_path = _training_data_argument(params)
        if data_path is None:
            raise JobUnrunnable(_why_the_dataset_is_unusable(params))
        weights_in = Path(str(params["model_path"]))
        device = resolve_device(params.get("device"), int(params.get("slot_index") or 0))

        settings, ignored = self._training_settings(params)
        total_epochs = int(settings["epochs"])

        ctx.report_progress(0, total_epochs, "epochs", phase="loading_model")
        ctx.log(
            f"training {weights_in.name} on {_dataset_root(data_path).name} for {total_epochs} epochs "
            f"on {device}"
        )

        # Said once, before the first epoch, so an attempt that fails later
        # still carries the record of what it was running (MARP_API#232).
        try:
            ctx.report_settings(
                self.engine_name,
                {
                    **{name: _setting_entry(value, "job")
                       for name, value in settings.items()},
                    **_run_provenance(device),
                },
                ignored,
            )
        except Exception as error:
            # Bookkeeping must never cost a volunteer's GPU run.
            ctx.log(
                f"could not report this run's settings: {type(error).__name__}: {error}",
                level="warning",
            )

        # Imported here, not at module scope: importing ultralytics pulls in
        # torch, and a worker that never trains must not pay for it.
        from ultralytics import YOLO

        model = YOLO(str(weights_in))

        # What this run writes, inside the attempt's own directory so two jobs
        # on one machine cannot collide over a run folder (R16).
        run_root = Path(ctx.checkpoint_dir) / "train"
        run_root.mkdir(parents=True, exist_ok=True)

        state = _RunState(
            ctx=ctx,
            total_epochs=total_epochs,
            run_root=run_root,
            engine_name=self.engine_name,
        )
        self._attach_callbacks(model, state)

        ctx.report_progress(0, total_epochs, "epochs", phase="training")

        try:
            model.train(
                data=str(data_path),
                device=device,
                project=str(run_root),
                name="run",
                exist_ok=True,
                **settings,
            )
        except _StopTraining:
            # Raised out of a callback when the operator stopped the job. The
            # weights published so far are the run's honest partial result.
            state.stopped_early = True
            ctx.log("stop requested; keeping the best weights published so far", level="warning")

        # The final best, published unconditionally -- it must never be the one
        # the rate limiter happened to skip (R11).
        best = state.best_weights_path()
        published: dict[str, Any] | None = None
        if best is not None and best.is_file():
            sha256 = ctx.publish_artifact(best, "weights")
            published = {
                "role": "weights",
                "sha256": sha256,
                "path": str(best),
                "size_bytes": best.stat().st_size,
            }
            ctx.log(f"published final weights sha256={sha256}")
        else:
            ctx.log("training produced no best weights to publish", level="warning")

        ctx.report_progress(state.epochs_done, total_epochs, "epochs", phase="publishing")

        return {
            "engine": self.engine_name,
            "device": device,
            # The same value the settings report carries, under the same name.
            # The settings report is what MARP reads; this is here so the
            # terminal log line a failure is diagnosed from says which build
            # produced it, without having to go and find the settings report.
            "train_script_commit": worker_revision(),
            "dataset": str(_dataset_root(data_path)),
            "epochs_requested": total_epochs,
            "epochs_completed": state.epochs_done,
            "checkpoints_published": state.publishes,
            "best": state.best_metric,
            "metrics": state.last_metrics,
            "elapsed_s": round(time.monotonic() - started_at, 3),
            "weights": published,
            "settings": settings,
            "stopped_early": state.stopped_early,
        }

    # _training_settings()
    # Builds the arguments this run will pass to Ultralytics.
    # Inputs: the job's params mapping.
    # Output: (settings actually used, params that nothing honours).
    # Use this so the allow-list is applied in exactly one place and the ignored
    # keys can be reported rather than silently dropped.
    def _training_settings(self, params: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:

        settings: dict[str, Any] = dict(_TRAINING_DEFAULTS)
        ignored: dict[str, Any] = {}

        # Set before the job's own params are applied, so a job that names
        # `imgsz` still overrides it below.
        settings["imgsz"] = _default_imgsz(params)

        for name, value in params.items():
            # The worker's own keys are not submitter settings and are not
            # reported as ignored ones.
            if name.startswith("_") or name in {
                "data_yaml", "model_path", "model_sha256", "model_class_names",
                "device", "slot_index", "slot_count", "dataset_path",
            }:
                continue

            caster = _ALLOWED_TRAINING_PARAMS.get(name)
            if caster is None:
                ignored[name] = value
                continue

            try:
                settings[name] = caster(value)
            except (TypeError, ValueError):
                ignored[name] = value

        return settings, ignored

    # _attach_callbacks()
    # Wires Ultralytics' epoch hooks to the job context.
    # Inputs: the YOLO model and the run state.
    # Output: none.
    # Use this before train(). The epoch boundary is the only place a training
    # run yields, so it is where progress, metrics, stop and pause all live.
    def _attach_callbacks(self, model: Any, state: "_RunState") -> None:

        # `on_train_start` rather than any earlier hook: it is the first point
        # at which Ultralytics has finished _setup_train() and the optimizer
        # exists, and it is still before epoch 1. `on_pretrain_routine_start`
        # fires from the trainer's constructor, where nothing is resolved yet.
        model.add_callback("on_train_start", state.on_train_start)

        # The live ones. `on_train_batch_end` and `on_val_batch_end` fire once
        # per batch -- many times a second on a fast card -- so both are
        # throttled inside the handler rather than here.
        #
        # `on_val_batch_end` is handed the VALIDATOR, not the trainer: it has
        # its own `batch_i` and `dataloader`, which is why the validator pause
        # can have a bar at all.
        model.add_callback("on_train_epoch_start", state.on_epoch_start)
        model.add_callback("on_train_batch_end", state.on_batch_end)
        model.add_callback("on_val_batch_end", state.on_val_batch_end)

        # `on_train_end` fires after final_eval() has validated the best
        # weights and after the plots are written (trainer.py:671-675), so it
        # is the first moment the run's own scores and files both exist. A
        # stopped run never reaches it, which is correct: there was no final
        # validation, so there is nothing to report.
        model.add_callback("on_train_end", state.on_train_end)
        model.add_callback("on_fit_epoch_end", state.on_epoch_end)


# _StopTraining
# Raised inside a callback to unwind a run the operator stopped.
# A private exception rather than JobStopped, because it never leaves this
# module: run() catches it and turns it into a partial result.
class _StopTraining(RuntimeError):
    pass


# _RunState
# What one training run knows about itself while it is running.
# Separate from the engine because the callbacks need somewhere to keep count,
# and the engine instance is shared by the registry.
class _RunState:

    # __init__()
    # Builds the state for one run.
    # Inputs: the job context, the epoch total, where the run writes, and the
    #   engine name the settings reports are filed under.
    # Output: initialized _RunState.
    def __init__(
        self,
        ctx: JobContext,
        total_epochs: int,
        run_root: Path,
        engine_name: str = "marp_training",
    ) -> None:

        self._ctx = ctx
        self._total = total_epochs
        self._run_root = run_root
        self._engine_name = engine_name

        self.epochs_done = 0
        self.publishes = 0

        # Live-sample bookkeeping. `_global_step` is monotonic across the whole
        # run so batch points share one x axis; `_batch_in_epoch` resets each
        # epoch because that is what the console line counts.
        self._global_step = 0
        self._batch_in_epoch = 0
        self._last_sample_at = 0.0
        self.stopped_early = False
        self.best_metric: float | None = None
        self.last_metrics: dict[str, Any] = {}

        # When weights were last sent up, so the rate limit is measured from a
        # real publish rather than from the start of the run.
        self._last_publish_at = 0.0

    # on_train_start()
    # Reports what Ultralytics resolved, once, before the first epoch.
    # Inputs: the Ultralytics trainer.
    # Output: none.
    #
    # A second settings report rather than more fields on the first one, because
    # the first is sent before the model is even loaded and none of this exists
    # yet. Keeping the first where it is matters: an attempt that dies during
    # setup -- a broken dataset, an OOM building the model -- still carries the
    # record of what it was trying to run, which is why MARP_API#232 put it
    # there. This one says what the run settled on and is only reachable once
    # setup succeeded.
    def on_train_start(self, trainer: Any) -> None:

        # Bookkeeping must never cost a volunteer's GPU run. Same rule as the
        # first report: a fault building the record is a warning, not a failure.
        try:
            resolved = _resolved_training_settings(trainer)
            if resolved:
                self._ctx.report_settings(self._engine_name, resolved, {})
        except Exception as error:
            self._ctx.log(
                f"could not report what this run resolved: {type(error).__name__}: {error}",
                level="warning",
            )


    # on_train_end()
    # Publishes what the finished run measured, and the files it drew.
    # Inputs: the Ultralytics trainer, after the final validation.
    # Output: none.
    #
    # Three kinds of thing leave here, and they are separate artifacts on
    # purpose. The numbers are what MARP charts and compares between runs; the
    # images are what the trainer actually produced and are kept as a fallback;
    # the curves are a megabyte of their own and a reader who does not want them
    # should not have to parse them to skip them.
    #
    # **Nothing here may fail the run.** Training has finished and the weights
    # are already published by the time this runs, so a fault in bookkeeping
    # would throw away a result that is complete. Every piece is attempted
    # independently and every failure is a warning.
    def on_train_end(self, trainer: Any) -> None:

        split = None
        try:
            split = _evaluated_split(trainer)
        except Exception:
            pass

        # The scores. Role is lower case and exact: MARP looks this one up by
        # name, where the plot roles below are opaque to it.
        try:
            report = _final_metrics_report(trainer)
            if report:
                self._publish_json(report, "final_metrics")
        except Exception as error:
            self._ctx.log(
                f"could not publish this run's metrics: {type(error).__name__}: {error}",
                level="warning",
            )

        # The curves. Also looked up by name, also exact.
        try:
            curves = _curves_report(trainer, split)
            if curves:
                self._publish_json(curves, "metrics_curves")
        except Exception as error:
            self._ctx.log(
                f"could not publish this run's curves: {type(error).__name__}: {error}",
                level="warning",
            )

        # The plots Ultralytics drew.
        try:
            self._publish_run_files(trainer)
        except Exception as error:
            self._ctx.log(
                f"could not publish this run's plots: {type(error).__name__}: {error}",
                level="warning",
            )


    # _publish_json()
    # Writes one report beside the run and hands it over.
    # Inputs: the report and the role to publish it under.
    # Output: none.
    # Written to the attempt's own directory rather than the run directory, so
    # it cannot collide with anything Ultralytics writes there.
    def _publish_json(self, report: dict, role: str) -> None:

        path = Path(self._ctx.checkpoint_dir) / f"{role}.json"

        # allow_nan=False turns a stray NaN into an exception here rather than
        # into a file MARP cannot parse. _json_number() should have caught them
        # all; this is the check that says so.
        path.write_text(
            json.dumps(report, indent=1, allow_nan=False), encoding="utf-8"
        )

        sha256 = self._ctx.publish_artifact(path, role)
        self._ctx.log(f"published {role} sha256={sha256}")


    # _publish_run_files()
    # Hands over the plots and the CSV Ultralytics wrote for this run.
    # Inputs: the trainer, for its save_dir.
    # Output: none.
    #
    # **The role is the real filename stem, verbatim.** `PR_curve`, not
    # `pr_curve`: MARP stores the role as sent and serves it back under that
    # name, and the dashboard keys off it, so lower-casing would make the record
    # disagree with the run directory for no gain. There is no allow-list, which
    # is the point -- a plot a later Ultralytics adds arrives rather than being
    # dropped by a filter neither of us remembered to update.
    #
    # Top level only. `weights/` sits below this directory and holds best.pt and
    # last.pt, which are around 130 MB each and are published separately.
    def _publish_run_files(self, trainer: Any) -> None:

        run_dir = None
        save_dir = getattr(trainer, "save_dir", None)
        if save_dir:
            run_dir = Path(str(save_dir))
        if run_dir is None or not run_dir.is_dir():
            best = self.best_weights_path()
            run_dir = best.parent.parent if best is not None else None
        if run_dir is None or not run_dir.is_dir():
            return

        # Roles must be unique within an attempt -- the runner files artifacts
        # into a mapping keyed by role, so a second one of the same name would
        # replace the first in the record rather than sitting beside it.
        published: set[str] = set()

        for path in sorted(run_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".csv"}:
                continue

            role = path.stem
            if role in published:
                self._ctx.log(
                    f"two run files share the name {role!r}; kept the first",
                    level="warning",
                )
                continue

            try:
                self._ctx.publish_artifact(path, role)
            except Exception as error:
                # One plot that could not be handed over is not a failed run.
                self._ctx.log(
                    f"could not publish {path.name}: {type(error).__name__}: {error}",
                    level="warning",
                )
                continue

            published.add(role)

        if published:
            self._ctx.log(f"published {len(published)} run files")


    # on_epoch_start()
    # Resets the per-epoch batch counter and marks the phase change.
    # Inputs: the Ultralytics trainer.
    # Output: none.
    def on_epoch_start(self, trainer: Any) -> None:

        self._batch_in_epoch = 0

        # A phase change always samples, however recently the last one went:
        # it is the point of the chart that explains a flat stretch.
        self._emit_sample(trainer, force=True)


    # on_batch_end()
    # Reports one live training sample, at most every few seconds.
    # Inputs: the Ultralytics trainer.
    # Output: none.
    def on_batch_end(self, trainer: Any) -> None:

        # Counted here rather than read off the trainer: Ultralytics keeps the
        # batch index in a local variable inside its own loop and does not
        # expose it. Counting the callback is exact, since it fires once per
        # batch, and it is the only honest way to get the number.
        self._batch_in_epoch += 1
        self._global_step += 1

        self._emit_sample(trainer)


    # on_val_batch_end()
    # Reports one live validation sample, at most every few seconds.
    # Inputs: the VALIDATOR -- not the trainer.
    # Output: none.
    # Validation is a visible pause on the console, so it gets its own bar.
    def on_val_batch_end(self, validator: Any) -> None:

        now = time.monotonic()
        if (now - self._last_sample_at) < _SAMPLE_INTERVAL_S:
            return
        self._last_sample_at = now

        try:
            sample: dict[str, Any] = {"at": time.time()}

            batch = getattr(validator, "batch_i", None)
            if batch is not None:
                # batch_i is zero-based, and a person counts from one.
                sample["batch"] = int(batch) + 1

            loader = getattr(validator, "dataloader", None)
            if loader is not None:
                sample["batches_total"] = len(loader)

            if len(sample) > 1:
                self._ctx.report_metrics(self._global_step, _VAL_BATCH_PHASE, sample)
        except Exception:
            # A sample is worth nothing next to the run producing it.
            pass


    # _emit_sample()
    # Builds and sends one live training sample.
    # Inputs: the trainer, and whether to bypass the throttle.
    # Output: none.
    #
    # **Every field is attempted separately and a failure drops only that
    # field.** One unreadable number must never cost the sample, and no sample
    # may ever cost the run -- training has real GPU hours in it and this is a
    # chart (#268).
    def _emit_sample(self, trainer: Any, force: bool = False) -> None:

        now = time.monotonic()
        if not force and (now - self._last_sample_at) < _SAMPLE_INTERVAL_S:
            return
        self._last_sample_at = now

        try:
            sample = self._live_sample(trainer)
            if sample:
                self._ctx.report_metrics(self._global_step, _BATCH_PHASE, sample)
        except Exception:
            pass


    # _live_sample()
    # What one live sample says, as numbers.
    # Inputs: the trainer.
    # Output: a JSON-safe mapping; empty when nothing could be read.
    #
    # Everything here is measured or derived from measurements. `it/s` and the
    # epoch ETA are not reported by Ultralytics at all, but elapsed-since-epoch
    # -start and the batch counts are real, so the division is arithmetic on
    # measurements rather than an estimate dressed as one.
    def _live_sample(self, trainer: Any) -> dict[str, Any]:

        sample: dict[str, Any] = {
            "at": time.time(),
            "epoch": self.epochs_done + 1,
            "epochs_total": self._total,
            "batch": self._batch_in_epoch,
        }

        # Batches per epoch, for the denominator of the bar.
        loader = getattr(trainer, "train_loader", None)
        if loader is not None:
            try:
                sample["batches_total"] = len(loader)
            except Exception:
                pass

        # The running mean losses, under the names the task itself uses --
        # box/cls/dfl for a detector, a single `loss` for classify.
        tloss = getattr(trainer, "tloss", None)
        labeller = getattr(trainer, "label_loss_items", None)
        if tloss is not None and callable(labeller):
            try:
                for name, value in labeller(tloss).items():
                    number = _json_number(value)
                    if number is not None:
                        sample[str(name)] = number
            except Exception:
                pass

        # The same number the console's GPU_mem column shows, in the same unit.
        #
        # `_get_memory()` is what trainer.py:594 formats into that column, and
        # it already handles cuda, mps and cpu -- reaching for
        # `accelerator.memory_reserved()` directly would be a second copy of
        # that branching, wrong on two of the three. Private, so guarded: if a
        # later Ultralytics drops it the field goes absent, which is the honest
        # failure rather than a zero.
        #
        # Reads 0 on a CPU run, which is not a gap -- it is what the console
        # prints, because no GPU memory is in use.
        #
        # **Sent as BYTES, although the console shows gigabytes.** At gigabyte
        # precision a 700 MB reservation rounds to 0.0, which reads as "no
        # memory" when it means "less than half a gigabyte" -- the same
        # gap-that-looks-like-a-measurement this work keeps guarding against.
        # The reader can scale; it cannot un-round.
        #
        # The multiplication is exact rather than approximate: `_get_memory()`
        # divides the byte count by 2**30, and multiplying a float64 back by
        # the same power of two recovers the original integer bit for bit.
        memory = getattr(trainer, "_get_memory", None)
        if callable(memory):
            try:
                gigabytes = _json_number(memory())
                if gigabytes is not None:
                    sample["gpu_memory_bytes"] = int(round(gigabytes * (2 ** 30)))
            except Exception:
                pass

        args = getattr(trainer, "args", None)
        imgsz = _json_number(getattr(args, "imgsz", None))
        if imgsz is not None:
            sample["imgsz"] = int(imgsz)

        self._add_pace(trainer, sample)

        return sample


    # _add_pace()
    # Adds iterations per second and the epoch's ETA, when they are knowable.
    # Inputs: the trainer and the sample being built.
    # Output: none; the sample is extended in place.
    #
    # Separate because it is the only part that divides, and division is where
    # a plausible-looking wrong number comes from. Nothing is added before the
    # first batch of an epoch has finished, because a rate over zero elapsed
    # time is not slow -- it is undefined.
    def _add_pace(self, trainer: Any, sample: dict[str, Any]) -> None:

        started = _json_number(getattr(trainer, "epoch_time_start", None))
        if started is None or self._batch_in_epoch <= 0:
            return

        elapsed = time.time() - started
        if elapsed <= 0:
            return

        rate = self._batch_in_epoch / elapsed
        sample["iterations_per_second"] = round(rate, 4)

        total = sample.get("batches_total")
        if isinstance(total, int) and total > self._batch_in_epoch and rate > 0:
            remaining = (total - self._batch_in_epoch) / rate
            sample["epoch_eta_seconds"] = round(remaining, 1)


    # best_weights_path()
    # Where Ultralytics keeps the best weights for this run.
    # Inputs: none.
    # Output: the path, or None when the run has not written one yet.
    def best_weights_path(self) -> Path | None:

        # Ultralytics composes `<project>/<name>/weights/best.pt`. Looked up
        # rather than remembered, because the trainer may move to `run2` if a
        # directory already existed, and the glob finds whichever it chose.
        candidates = sorted(self._run_root.glob("*/weights/best.pt"))
        return candidates[-1] if candidates else None

    # on_epoch_end()
    # Reports one finished epoch, publishes a new best, and honours stop/pause.
    # Inputs: the Ultralytics trainer.
    # Output: none; raises _StopTraining when the job has been stopped.
    #
    # `on_fit_epoch_end` rather than `on_train_epoch_end`: the fit hook runs
    # after validation, so the metrics exist and `best.pt` has already been
    # updated if this epoch improved. The train hook fires before either, and
    # would report a best that is one epoch stale.
    def on_epoch_end(self, trainer: Any) -> None:

        # Ultralytics' epoch index is zero-based, so the first finished epoch is 1.
        reported = int(getattr(trainer, "epoch", self.epochs_done) or 0) + 1

        # **The hook fires once more than there are epochs.** After the last one
        # Ultralytics validates the best weights again and calls this with the
        # index already advanced, so a two-epoch run arrives here three times. The
        # first real run of this engine reported "3/2 epochs" and would have
        # written an epoch 3 that never happened -- a row in `epochs` and a
        # `total_epochs` that both disagree with the run.
        #
        # Those final metrics are the run's result and worth keeping; they are
        # simply not another epoch. Held as the latest metrics, which is what the
        # summary reports, and not counted.
        if reported > self._total or reported <= self.epochs_done:
            raw = _epoch_metrics(trainer)
            if raw:
                self.last_metrics = raw
            return

        self.epochs_done = reported

        # Ultralytics' metric names vary by task and version, which is exactly
        # why report_metrics takes a free-form mapping.
        self.last_metrics = _epoch_metrics(trainer)

        fitness = getattr(trainer, "fitness", None)
        if fitness is not None:
            try:
                self.best_metric = float(getattr(trainer, "best_fitness", fitness) or fitness)
            except (TypeError, ValueError):
                pass

        self._ctx.report_progress(self.epochs_done, self._total, "epochs", phase="training")
        self._ctx.report_metrics(self.epochs_done, "training", self.last_metrics)

        self._maybe_publish(trainer)

        # A pause blocks here with the model loaded and the dataset open, so
        # resuming costs nothing and no epoch is repeated (#57 R5).
        self._ctx.wait_while_paused()

        # Cancellation arrives at the epoch boundary and nowhere else. Raising
        # rather than returning is what unwinds Ultralytics' own loop.
        if self._ctx.should_stop():
            raise _StopTraining()

    # _maybe_publish()
    # Sends the best weights up, if this epoch improved on them and it is time.
    # Inputs: the Ultralytics trainer.
    # Output: none.
    #
    # Two conditions, and both matter. Only a new best is worth sending, because
    # an epoch that did not improve produces weights nobody would register or
    # run. And not more often than the interval, because early epochs improve on
    # almost every pass and each file is around 130 MB on a volunteer's upload.
    def _maybe_publish(self, trainer: Any) -> None:

        improved = bool(getattr(trainer, "best_fitness", None) is not None
                        and getattr(trainer, "fitness", None) is not None
                        and trainer.fitness >= trainer.best_fitness)
        if not improved:
            return

        now = time.monotonic()
        if self._last_publish_at and (now - self._last_publish_at) < _PUBLISH_INTERVAL_S:
            return

        best = self.best_weights_path()
        if best is None or not best.is_file():
            return

        try:
            sha256 = self._ctx.publish_artifact(best, "checkpoint")
        except Exception as error:
            # A checkpoint that could not be handed over is not a failed run.
            # The run keeps going and the next improvement tries again.
            self._ctx.log(
                f"could not publish a checkpoint: {type(error).__name__}: {error}",
                level="warning",
            )
            return

        self._last_publish_at = now
        self.publishes += 1
        self._ctx.log(f"published checkpoint at epoch {self.epochs_done} sha256={sha256}")
