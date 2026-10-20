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
}


# Defaults for a MARP training run.
# Stated here rather than left to Ultralytics so that what ran is a property of
# this worker's version rather than of whichever library version was installed.
_TRAINING_DEFAULTS: dict[str, Any] = {
    "epochs": 100,
    "imgsz": 640,
    "patience": 50,
}


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
                {name: {"value": value, "type": type(value).__name__, "source": "job"}
                 for name, value in settings.items()},
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

        state = _RunState(ctx=ctx, total_epochs=total_epochs, run_root=run_root)
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
    # Inputs: the job context, the epoch total, and where the run writes.
    # Output: initialized _RunState.
    def __init__(self, ctx: JobContext, total_epochs: int, run_root: Path) -> None:

        self._ctx = ctx
        self._total = total_epochs
        self._run_root = run_root

        self.epochs_done = 0
        self.publishes = 0
        self.stopped_early = False
        self.best_metric: float | None = None
        self.last_metrics: dict[str, Any] = {}

        # When weights were last sent up, so the rate limit is measured from a
        # real publish rather than from the start of the run.
        self._last_publish_at = 0.0

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
            raw = dict(getattr(trainer, "metrics", {}) or {})
            if raw:
                self.last_metrics = {str(k): v for k, v in raw.items()}
            return

        self.epochs_done = reported

        # Ultralytics' metric names vary by task and version, which is exactly
        # why report_metrics takes a free-form mapping.
        raw = dict(getattr(trainer, "metrics", {}) or {})
        self.last_metrics = {str(k): v for k, v in raw.items()}

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
