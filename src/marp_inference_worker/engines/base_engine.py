# base_engine.py
# Created: 2026-07-07
# Author: Isaac Travers
#
# Engine contract for the MARP Inference Worker.
# This file defines the one contract every engine is invoked through: run(ctx, spec).
# The context object is the engine's only channel to the outside world, so an engine
# learns nothing about MARP, service tokens, Jellyfin or the coordinator's routes.
# Engine-specific model loading and prediction code belongs in the engine modules.

# ABC tools define the abstract engine base class.
from abc import ABC, abstractmethod

# Path types the checkpoint and artifact locations handed to an engine.
from pathlib import Path

# Protocol lets the context be a structural type, so a test can pass a fake context
# without importing the real job machinery.
from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

# ModelSpec types the frame-inference capability's load_model, and is imported
# only for type checking. At runtime it would be a cycle: the models package
# imports model_manager, which imports engine_registry, which imports this file.
# Importing base_engine first then failed on a partially initialized module --
# a real breakage that only appeared when a test imported this module before
# anything else had pulled the package in.
if TYPE_CHECKING:
    from marp_inference_worker.models.model_spec import ModelSpec


# JobUnrunnable
# Raised by preflight() when this machine cannot run a job at all.
# The runner turns this into an immediate terminal failure with the reason attached,
# rather than letting the job start and die part way through (R14).
class JobUnrunnable(RuntimeError):
    pass


# JobStopped
# Raised inside an engine's run() when should_stop() went true and the engine
# unwound. The runner reports this as a cancellation rather than a failure.
class JobStopped(RuntimeError):
    pass


# JobContext
# The six calls an engine may make, and nothing else.
# Inputs: supplied by the job runner in the child process.
# Output: a structural type engines can depend on without importing the runner.
# Use this as the type of the ctx argument to run(); anything satisfying it will do.
@runtime_checkable
class JobContext(Protocol):

    # params
    # The job's free-form parameter mapping, straight from the job spec.
    # Engines read their own settings from here and ignore keys they do not know.
    @property
    def params(self) -> Mapping[str, Any]: ...

    # checkpoint_dir
    # A writable directory private to this job attempt.
    # Engines put run outputs, weights and scratch files here.
    @property
    def checkpoint_dir(self) -> Path: ...

    # resume_from
    # A previously published checkpoint to continue from, or None for a fresh start.
    # Milestone 1 always hands None; training in Milestone 2 is what needs it.
    @property
    def resume_from(self) -> Path | None: ...

    # log()
    # Records one line of engine progress narrative for the operator and coordinator.
    # Inputs: message text and an optional severity level.
    # Output: none.
    def log(self, message: str, level: str = "info") -> None: ...

    # report_progress()
    # Reports how far through the work the engine is.
    # Inputs: units finished, total units expected, and what a unit is ("frames").
    # Output: none.
    def report_progress(self, done: int, total: int | None, unit: str) -> None: ...

    # report_metrics()
    # Reports engine metrics for one step of one phase.
    # Inputs: step number, phase name, and a free-form mapping of metric values.
    # Output: none. The mapping is deliberately not schema'd -- Ultralytics' metric
    # names differ by task and version, so pinning them here would be wrong.
    def report_metrics(self, step: int, phase: str, metrics: Mapping[str, Any]) -> None: ...

    # publish_artifact()
    # Hands a finished file to the runner, which hashes it and offers it upstream.
    # Inputs: local path and a role name such as "results" or "checkpoint".
    # Output: the sha256 the runner recorded for that file.
    def publish_artifact(self, path: Path, kind: str) -> str: ...

    # should_stop()
    # True once the job has been told to stop. Engines must check it inside their
    # work loop; it is how cancellation arrives, because nothing can call into the
    # engine from outside.
    # Inputs: none.
    # Output: True when the engine should unwind and return.
    def should_stop(self) -> bool: ...


# BaseEngine
# The shared interface for every engine the worker can dispatch a job to.
# Engines are constructed inside the job's own child process, so an engine instance
# may hold model handles and tracker state without any cross-job locking.
class BaseEngine(ABC):

    # engine_name
    # Returns the public engine name used in a job spec's `engine` field.
    # Inputs: none.
    # Output: engine name string.
    # Use this so the registry can match a job spec to an engine class.
    @property
    @abstractmethod
    def engine_name(self) -> str:

        # Subclasses must define their public engine name.
        raise NotImplementedError

    # describe()
    # Returns what this engine is and what it needs, for the worker's /status.
    # Inputs: none.
    # Output: JSON-safe dictionary. Should say the engine's name, its version if it
    # has one, and whether it requires a GPU.
    # Use this so /status reports the real engine list rather than a literal (R12).
    @abstractmethod
    def describe(self) -> dict[str, Any]:

        # Subclasses must describe themselves.
        raise NotImplementedError

    # preflight()
    # Decides, before any work starts, whether this job can run on this machine.
    # Inputs: the job spec mapping.
    # Output: none on success; raises JobUnrunnable with a reason otherwise.
    # Use this so a job that cannot run fails immediately and says why (R14).
    @abstractmethod
    def preflight(self, spec: Mapping[str, Any]) -> None:

        # Subclasses must decide whether they can run the job.
        raise NotImplementedError

    # run()
    # Runs one job to completion, cancellation or failure.
    # Inputs: a JobContext and the job spec mapping.
    # Output: a JSON-safe summary dictionary. Bulk results go out through
    # ctx.publish_artifact(), never in the return value (R11).
    # Use this as the single entry point for all job execution.
    @abstractmethod
    def run(self, ctx: JobContext, spec: Mapping[str, Any]) -> dict[str, Any]:

        # Subclasses must implement the job body.
        raise NotImplementedError


# FrameInferenceCapable
# Declares that an engine can also answer the worker's single-frame API routes.
# This capability is separate from run() because most jobs never need it and the
# frame routes exist for GUI tools and debugging, not for coordinator work.
# Callers test for it with isinstance() rather than probing for method names.
class FrameInferenceCapable(ABC):

    # load_model()
    # Loads a cached model artifact into this engine runtime.
    # Inputs: validated ModelSpec and local cached artifact path.
    # Output: JSON-safe metadata describing the loaded model.
    # Use this when /models/load prepares a model for the frame routes.
    @abstractmethod
    def load_model(self, model_spec: "ModelSpec", artifact_path: Path) -> Any:

        # Subclasses must implement actual model-loading behavior.
        raise NotImplementedError

    # infer_frame()
    # Runs inference on one image source and returns normalized detections.
    # Inputs: model ID, image source, confidence, label map, and image flag.
    # Output: normalized inference result dictionary.
    # Use this for the JSON frame-inference route.
    @abstractmethod
    def infer_frame(
        self,
        model_id: str,
        image_source: str,
        confidence: float,
        class_names_by_id: dict[int, str],
        return_annotated_image: bool = False,
    ) -> dict[str, Any]:

        # Subclasses must implement frame inference.
        raise NotImplementedError

    # infer_frame_image()
    # Runs inference and returns annotated image bytes.
    # Inputs: model ID, image source, confidence, and label map.
    # Output: encoded image bytes.
    # Use this for the binary frame-image route.
    @abstractmethod
    def infer_frame_image(
        self,
        model_id: str,
        image_source: str,
        confidence: float,
        class_names_by_id: dict[int, str],
    ) -> bytes:

        # Subclasses must implement annotated image inference.
        raise NotImplementedError
