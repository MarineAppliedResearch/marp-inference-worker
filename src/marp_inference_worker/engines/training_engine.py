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
from marp_inference_worker.system.device import cuda_device_count, resolve_device

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
    "imgsz": 640,
    "patience": 50,
}


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

    return {
        # Named for MARP's `training_runs.compute_device`. The resolved device,
        # which may or may not be the one the job named.
        "compute_device": _setting_entry(device, "worker"),
        # Named for MARP's `training_runs.train_script_commit`. A sha when the
        # worker runs from a checkout, the package version when it runs from an
        # installed wheel; see worker_revision() for why both are answers.
        "train_script_commit": _setting_entry(worker_revision(), "worker"),
    }


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
            "name": self.engine_name,
            "kind": "training",
            "requires_gpu": True,
            "task": "detect",
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
        data_yaml = params.get("data_yaml")
        if not data_yaml:
            raise JobUnrunnable("the job carried no dataset; nothing was prepared to train on")
        if not Path(str(data_yaml)).is_file():
            raise JobUnrunnable(f"the dataset's data.yaml is missing from {data_yaml}")

        if not params.get("model_path"):
            raise JobUnrunnable("the job carried no starting weights")

    # run()
    # Runs one training job to completion, cancellation or failure.
    # Inputs: the job context and the job spec.
    # Output: a JSON-safe summary. The weights go out through publish_artifact.
    def run(self, ctx: JobContext, spec: Mapping[str, Any]) -> dict[str, Any]:

        started_at = time.monotonic()
        params = dict(spec.get("params") or {})

        data_yaml = Path(str(params["data_yaml"]))
        weights_in = Path(str(params["model_path"]))
        device = resolve_device(params.get("device"), int(params.get("slot_index") or 0))

        settings, ignored = self._training_settings(params)
        total_epochs = int(settings["epochs"])

        ctx.report_progress(0, total_epochs, "epochs", phase="loading_model")
        ctx.log(
            f"training {weights_in.name} on {data_yaml.parent.name} for {total_epochs} epochs "
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
                data=str(data_yaml),
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
            "dataset": str(data_yaml.parent),
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
