# engine_registry.py
# Created: 2026-07-07
# Author: Isaac Travers
#
# Engine registry for the MARP Inference Worker.
# This file maps public engine names to engine instances, and is the only place
# that knows which engines exist. Keep engine selection here so the job runner
# and the model manager do not import or branch across every runtime directly.
#
# Two kinds of thing are registered. A job engine implements run(ctx, spec) and
# is what the runner dispatches to. A frame-inference engine answers the worker's
# own single-frame routes, which exist for GUI tools and debugging. An engine may
# be both; MockEngine is.

# The engine contract and the frame capability, used for the typed lookups.
from marp_inference_worker.engines.base_engine import BaseEngine, FrameInferenceCapable

# MockEngine implements the job contract without an ML runtime.
from marp_inference_worker.engines.mock_engine import MockEngine

# TrackingEngine runs MARP's detect -> track -> reduce pipeline.
from marp_inference_worker.engines.tracking_engine import TrackingEngine

# UltralyticsEngine serves the single-frame routes.
from marp_inference_worker.engines.ultralytics_engine import UltralyticsEngine


# Registered engine instances keyed by public engine name.
# Engines are constructed here at import, which is cheap: every ML import inside
# them is lazy, so registering TrackingEngine does not load torch.
_ENGINES_BY_NAME: dict[str, object] = {
    "mock": MockEngine(),
    "ultralytics": UltralyticsEngine(),
    "marp_tracking": TrackingEngine(),
}


# get_engine()
# Returns the engine registered under a public engine name.
# Inputs: engine name string.
# Output: the registered engine instance.
# Use this when the caller does not care which capabilities the engine has.
def get_engine(engine_name: str) -> object:

    # Normalize so API input casing does not affect lookup.
    normalized_engine_name = engine_name.lower()

    # Return the matching engine when it has been registered.
    if normalized_engine_name in _ENGINES_BY_NAME:
        return _ENGINES_BY_NAME[normalized_engine_name]

    # Fail clearly when an unsupported engine is requested.
    raise ValueError(f"Unsupported inference engine: {engine_name}")


# get_job_engine()
# Returns an engine that can run a job through run(ctx, spec).
# Inputs: engine name string from a job spec's `engine` field.
# Output: a BaseEngine.
# Use this from the job runner. The isinstance check is the point: an engine
# that does not implement the job contract is refused by name here, rather than
# failing later on a missing method.
def get_job_engine(engine_name: str) -> BaseEngine:

    # Look it up through the shared accessor so normalization happens once.
    engine = get_engine(engine_name)

    # A frame-only engine is a legitimate registration but cannot take a job.
    if not isinstance(engine, BaseEngine):
        raise ValueError(f"Engine cannot run jobs: {engine_name}")

    return engine


# get_frame_engine()
# Returns an engine that can answer the single-frame inference routes.
# Inputs: engine name string from ModelSpec.engine.
# Output: a FrameInferenceCapable engine.
# Use this from the model manager. This replaced a pair of hasattr() probes for
# "infer_frame" and "infer_frame_image": the capability is now declared by the
# class and checked once, so an engine cannot half-implement it and a typo in a
# method name cannot read as "this engine does not support frame inference".
def get_frame_engine(engine_name: str) -> FrameInferenceCapable:

    # Look it up through the shared accessor so normalization happens once.
    engine = get_engine(engine_name)

    # Say which capability is missing, not just that something went wrong.
    if not isinstance(engine, FrameInferenceCapable):
        raise ValueError(f"Engine does not support frame inference: {engine_name}")

    return engine


# job_engine_names()
# Lists the engines on this worker that can take a job.
# Inputs: none.
# Output: sorted list of engine names.
# Use this when polling, so the coordinator only offers work this worker can run.
def job_engine_names() -> list[str]:

    # Sorted so the reported list is stable between calls.
    return sorted(
        name for name, engine in _ENGINES_BY_NAME.items() if isinstance(engine, BaseEngine)
    )


# describe_engines()
# Describes every job engine on this worker.
# Inputs: none.
# Output: list of describe() mappings.
# Use this in /status and in enrolment, so the engine list reported is the one
# actually registered rather than a literal that goes stale (R12).
def describe_engines() -> list[dict[str, object]]:

    # Only job engines describe themselves; the frame routes report separately.
    return [
        engine.describe()
        for _, engine in sorted(_ENGINES_BY_NAME.items())
        if isinstance(engine, BaseEngine)
    ]
